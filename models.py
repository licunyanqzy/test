import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import line_profiler


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list, activation="relu", batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module):
    def __init__(
            self, obs_len, action_encoder_input_dim, action_encoder_hidden_dim,
    ):
        super(ActionEncoder, self).__init__()
        self.obs_len = obs_len
        self.action_encoder_input_dim = action_encoder_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim

        self.actionEncoderLSTM = nn.LSTMCell(self.action_encoder_input_dim, self.action_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.action_encoder_input_dim)

    def init_hidden(self, batch):
        return (        # torch.zeros or torch.randn ?
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
        )

    def forward(self, obs_traj):    # 先embedding,后LSTM,可能存在维度问题 ?
        batch = obs_traj.shape[1]
        action_encoder_hidden_state, action_encoder_cell_state = self.init_hidden(batch)

        obs_traj_embedding = self.inputEmbedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.action_encoder_input_dim)

        for i, input_data in enumerate(
            obs_traj_embedding[: self.obs_len].chunk(
                obs_traj_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            action_encoder_hidden_state, action_encoder_cell_state = self.actionEncoderLSTM(
                input_data.squeeze(0), (action_encoder_hidden_state, action_encoder_cell_state)
            )

        return action_encoder_hidden_state


class GoalEncoder(nn.Module):
    def __init__(
            self, obs_len, goal_encoder_input_dim, goal_encoder_hidden_dim
    ):
        super(GoalEncoder, self).__init__()
        self.obs_len = obs_len
        self.goal_encoder_input_dim = goal_encoder_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim

        self.goalEncoderLSTM = nn.LSTMCell(self.goal_encoder_input_dim, self.goal_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.goal_encoder_input_dim)

        self.hidden2goal = nn.Linear(goal_encoder_hidden_dim, 2)    # for experiment_goal

    def init_hidden(self, batch):
        return (
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda()
        )

    def forward(self, obs_goal):    # 先embedding,后LSTM,可能存在维度问题 ?
        batch = obs_goal.shape[1]
        goal_encoder_hidden_state, goal_encoder_cell_state = self.init_hidden(batch)

        obs_goal_embedding = self.inputEmbedding(obs_goal.contiguous().view(-1, 2))
        obs_goal_embedding = obs_goal_embedding.view(-1, batch, self.goal_encoder_input_dim)

        for i, input_data in enumerate(
            obs_goal_embedding[: self.obs_len].chunk(
                obs_goal_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            goal_encoder_hidden_state, goal_encoder_cell_state = self.goalEncoderLSTM(
                input_data.squeeze(0), (goal_encoder_hidden_state, goal_encoder_cell_state)
            )

        output = goal_encoder_hidden_state
        # output = self.hidden2goal(goal_encoder_hidden_state)    # for experiment_goal
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out

        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h, idx):
        # 维度可能存在问题 ?
        # idx是否有用 ?

        n = h.size()[0]

        # print(h.size())               [22,48]
        # print(self.w.size())          [4,48,48]
        # print(self.a_src.size())      [4,48,1]
        # print(self.a_dst.size())      [4,48,1]

        h_prime = torch.matmul(h, self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1)     # 作用 ?

        # print("\n")
        # print(attn_src.size())    [4,22,1]
        # print(attn_dst.size())    [4,22,1]
        # print(attn.size())        [4,22,22]
        # print("\n")

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        output_mean = torch.mean(output, dim=0)

        # print(attn.size())            [4,22,22]
        # print(output.size())          [4,22,48]
        # print(output_mean.size())     [22,48]
        # os._exit(0)

        if self.bias is not None:
            return output_mean + self.bias
        else:
            return output_mean

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class InteractionGate(nn.Module):
    def __init__(self, distance_embedding_dim, action_decoder_hidden_dim, ):
        super(InteractionGate, self).__init__()
        self.distance_embedding_dim = distance_embedding_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.distEmbedding = nn.Linear(1, self.distance_embedding_dim)
        self.gateStep1 = nn.Linear(
            distance_embedding_dim * 2 + action_decoder_hidden_dim * 2, action_decoder_hidden_dim
        )
        self.gateStep2 = nn.Sigmoid()

    def cal_dist(self, goal, action):
        n = goal.size()[0]
        distance_other = torch.zeros([n, n]).cuda()
        distance_self = torch.zeros([n, n])
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance_other[i, j] = torch.norm(action[i] - goal[i])
                    distance_self[i] = distance_other[i, j]
                else:
                    a1, a2, g1, g2 = action[i], action[j], goal[i], goal[j]
                    dn = (a1[0] - a1[0]) * (g1[1] - g2[1]) - (a1[1] - a2[1]) * (g1[0] - g2[0])
                    n1 = (a1[0] * a2[1] - a1[1] * a2[0]) * (g1[0] - g2[0]) - (a1[0] - a2[0]) * (
                                g1[0] * g2[1] - g1[0] * g2[0])
                    n2 = (a1[0] * a2[1] - a1[1] * a2[0]) * (g1[1] - g2[1]) - (a1[1] - a2[0]) * (
                                g1[0] * g2[1] - g1[0] * g2[0])

                    d1 = torch.norm(a1 - g1)
                    d2 = torch.norm(a2 - g2)
                    if not dn == 0:
                        p1 = n1 / dn
                        p2 = n2 / dn
                        p = torch.tensor([p1, p2]).cuda()
                        if (a1[0] - p1) * (p1 - g1[0]) > 0:
                            d1 = torch.norm(a1 - p)
                            d2 = torch.norm(a2 - p)

                    distance_other[i, j] = d1
                    distance_other[j, i] = d2
        return distance_self, distance_other

    def forward(self, action_hidden_state, goal_hidden_state, goal, action, ):
        num = goal.size()[0]
        distance_self, distance_other = self.cal_dist(goal, action)      # [21,21]
        distance_other_embedding = self.distEmbedding(distance_other.contiguous().view(-1, 1))
        distance_other_embedding = distance_other_embedding.view(num, num, self.distance_embedding_dim)
        distance_self_embedding = self.distEmbedding(distance_self.contiguous().view(-1 ,1))
        distance_self_embedding = distance_self_embedding.view(num, num, self.distance_embedding_dim)

        temp_action = action_hidden_state.repeat(num, 1).view(num, num, -1)
        temp_goal = goal_hidden_state.repeat(num, 1).view(num, num, -1)     # [21,21,48]
        mask_goal = torch.ones([num, num]).cuda()
        mask_goal = torch.triu(mask_goal, 0) - torch.triu(mask_goal, 1)    # [21,21]
        mask_goal = mask_goal.unsqueeze(2).expand(num, num, self.action_decoder_hidden_dim)
        mask_action = torch.ones([num, num]).cuda()
        mask_action = torch.triu(mask_action, 1) + torch.tril(mask_action, -1)
        mask_action = mask_action.unsqueeze(2).expand(num, num, self.action_decoder_hidden_dim)
        goal_action = temp_goal * mask_goal + temp_action * mask_action

        gate = self.gateStep1(torch.cat(
            [temp_action, goal_action, distance_self_embedding, distance_other_embedding], dim=0
        ))
        gate = self.gateStep2(gate)     # [21,21,48]
        output = goal_action * gate

        return output


class GAT(nn.Module):
    def __init__(
            self, distance_embedding_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout=0.2, alpha=0.2,
    ):
        super(GAT, self).__init__()
        self.n_heads = 4

        self.interactionGate = InteractionGate(
            distance_embedding_dim=distance_embedding_dim, action_decoder_hidden_dim=action_decoder_hidden_dim
        )

        self.gatLayer = GraphAttentionLayer(
            n_head=self.n_heads, f_in=action_decoder_hidden_dim,
            f_out=action_decoder_hidden_dim, attn_dropout=dropout,
        )

    def forward(self, action_hidden_state, goal_hidden_state, goal, action):
        # 只能进行一层GAT, 如何实现两层 ? to be continued ...
        n = action_hidden_state.size()[0]

        gated_hidden_state = self.interactionGate(
            action_hidden_state, goal_hidden_state, goal, action
        )   # [21,21,48]

        gate_layer_input = torch.cat([action_hidden_state.unsqueeze(1), gated_hidden_state], dim=1)
        output = self.gatLayer(gate_layer_input)    # to be continued ...

        return output


class Attention(nn.Module):
    """
    交互模块, 刻画interaction
    input: hidden_state
    output: hidden_state'
    """
    def __init__(
            self, distance_embedding_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout, alpha
    ):
        super(Attention, self).__init__()
        self.gat = GAT(
            distance_embedding_dim, action_decoder_hidden_dim, n_units, n_heads, dropout, alpha
        )

    def forward(
            self, action_decoder_hidden_state, goal_decoder_hidden_state, goal, action, seq_start_end
    ):
        outputs = []
        for start, end in seq_start_end.data:
            curr_action_hidden_state = action_decoder_hidden_state[start:end, :]     # [1413, 48]
            curr_goal_hidden_state = goal_decoder_hidden_state[start:end, :]     # [1413, 48]
            curr_goal = goal[start:end, :]       # [1413, 2]
            curr_action = action[start:end, :]       # [1413, 2]
            curr_graph_embedding = self.gat(
                curr_action_hidden_state, curr_goal_hidden_state, curr_goal, curr_action
            )
            outputs.append(curr_graph_embedding)

        output_hidden_state = torch.cat(outputs, dim=0)
        return output_hidden_state


class Decoder(nn.Module):
    def __init__(
            self, obs_len, pred_len, n_units, n_heads, dropout, alpha, distance_embedding_dim,
            action_decoder_input_dim, action_decoder_hidden_dim, goal_decoder_input_dim, goal_decoder_hidden_dim
    ):
        super(Decoder, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.goal_decoder_input_dim = goal_decoder_input_dim
        self.goal_decoder_hidden_dim = goal_decoder_hidden_dim
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim
        self.distance_embedding_dim = distance_embedding_dim

        self.goalDecoderLSTM = nn.LSTMCell(self.goal_decoder_input_dim, self.goal_decoder_hidden_dim)
        self.hidden2goal = nn.Linear(self.goal_decoder_hidden_dim, 2)
        self.goalEmbedding = nn.Linear(2, self.goal_decoder_input_dim)

        self.actionDecoderLSTM = nn.LSTMCell(self.action_decoder_input_dim, self.action_decoder_hidden_dim)
        self.actionEmbedding = nn.Linear(2, self.action_decoder_input_dim)
        self.hidden2action = nn.Linear(self.action_decoder_hidden_dim, 2)

        self.attention = Attention(  # 是否存在问题 ?
            self.distance_embedding_dim, self.action_decoder_hidden_dim, n_units, n_heads, dropout, alpha
        )

    def forward(
            self, goal_input_hidden_state, action_input_hidden_state,
            action_real, goal_real, teacher_forcing_ratio, seq_start_end,
    ):
        goal_decoder_hidden_state = goal_input_hidden_state
        goal_decoder_cell_state = torch.zeros_like(goal_decoder_hidden_state).cuda()
        action_decoder_hidden_state = action_input_hidden_state
        action_decoder_cell_state = torch.zeros_like(action_decoder_hidden_state).cuda()

        pred_goal = []
        pred_action = []
        action_output = action_real[self.obs_len - 1]
        goal_output = goal_real[self.obs_len - 1]

        batch = action_real.shape[1]
        action_real_embedding = self.actionEmbedding(action_real.contiguous().view(-1, 2))    # [28260, 16]
        action_real_embedding = action_real_embedding.view(-1, batch, self.action_decoder_input_dim)  # [20,1413,16]

        if self.training:
            for i, input_data in enumerate(
              action_real_embedding[-self.pred_len:].chunk(
                  action_real_embedding[-self.pred_len:].size(0), dim=0
              )
            ):
                teacher_forcing = random.random() < teacher_forcing_ratio
                if teacher_forcing:
                    goal_input_data = input_data
                    action_input_data = input_data
                else:
                    goal_input_data = self.goalEmbedding(action_output)
                    action_input_data = self.actionEmbedding(action_output)

                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    goal_input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)
                pred_goal += [goal_output]

                # 两种思路: 现为先LSTM后interaction, 是否需要调整为先interaction后LSTM ?

                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    action_input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )

                action_decoder_hidden_state = self.attention(
                    action_decoder_hidden_state, goal_decoder_hidden_state,
                    goal_output, action_output, seq_start_end
                )

                action_output = self.hidden2action(action_decoder_hidden_state)
                pred_action += [action_output]
        else:
            for i in range(self.pred_len):
                input_data = self.goalEmbedding(action_output)
                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)
                pred_goal += [goal_output]

                # 两种思路: 现为先LSTM后interaction, 是否需要调整为先interaction后LSTM ?

                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )

                action_decoder_hidden_state = self.attention(
                    action_decoder_hidden_state, goal_decoder_hidden_state,
                    goal_output, action_output, seq_start_end
                )

                action_output = self.hidden2action(action_decoder_hidden_state)
                pred_action += [action_output]

        pred_goal_output = torch.stack(pred_goal)
        pred_action_output = torch.stack(pred_action)
        return pred_goal_output, pred_action_output


class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_input_dim, action_encoder_hidden_dim,
            goal_input_dim, goal_encoder_hidden_dim, distance_embedding_dim,
            n_units, n_heads, dropout, alpha,
            noise_dim=(8,), noise_type="gaussian",
    ):
        super(TrajectoryPrediction, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.action_input_dim = action_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim
        self.goal_input_dim = goal_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.goal_decoder_hidden_dim = self.goal_encoder_hidden_dim + self.noise_dim[0]
        self.action_decoder_hidden_dim = self.action_encoder_hidden_dim + self.noise_dim[0]

        self.actionEncoder = ActionEncoder(
            obs_len=self.obs_len,
            action_encoder_input_dim=self.action_input_dim,
            action_encoder_hidden_dim=self.action_encoder_hidden_dim,
        )
        self.goalEncoder = GoalEncoder(
            obs_len=self.obs_len,
            goal_encoder_input_dim=self.goal_input_dim,
            goal_encoder_hidden_dim=self.goal_encoder_hidden_dim,
        )
        self.Decoder = Decoder(
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            goal_decoder_input_dim=self.goal_input_dim,
            goal_decoder_hidden_dim=self.goal_decoder_hidden_dim,
            action_decoder_input_dim=self.action_input_dim,
            action_decoder_hidden_dim=self.action_decoder_hidden_dim,
            distance_embedding_dim=distance_embedding_dim,
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha,
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end-start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
            self, input_action, input_goal, seq_start_end, teacher_forcing_ratio=0.5,
    ):
        goal_encoder_hidden_state = self.goalEncoder(input_action)
        action_encoder_hidden_state = self.actionEncoder(input_action)

        # add noise 噪声加的是否正确 ?
        goal_hidden_state_noise = self.add_noise(goal_encoder_hidden_state, seq_start_end)
        action_hidden_state_noise = self.add_noise(action_encoder_hidden_state, seq_start_end)

        pred_goal, pred_action = self.Decoder(
            goal_hidden_state_noise, action_hidden_state_noise,
            input_action, input_goal, teacher_forcing_ratio, seq_start_end,
        )

        return pred_goal, pred_action


import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def add_noise(shape, noise_type):
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

        obs_traj_embedding = self.inputEmbedding(obs_traj.contiguous().view(-1,2))
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

        return goal_encoder_hidden_state


class GoalDecoder(nn.Module):
    def __init__(
            self, pred_len, goal_decoder_input_dim, goal_decoder_hidden_dim,
    ):
        super(GoalDecoder, self).__init__()
        self.pred_len = pred_len
        self.goal_decoder_input_dim = goal_decoder_input_dim
        self.goal_decoder_hidden_dim = goal_decoder_hidden_dim

        self.goalDecoderLSTM = nn.LSTMCell(self.goal_decoder_input_dim, self.goal_decoder_hidden_dim)
        self.hidden2pos = nn.Linear(self.goal_decoder_hidden_dim, 2)
        self.inputEmbedding = nn.Linear(2, self.goal_decoder_input_dim)

    def forward(self, goal_real, input_hidden_state, teacher_forcing_ratio):    # 尚缺少加噪声的部分 ?
        pred_goal = []
        output = goal_real[-self.pred_len-1]
        batch = goal_real.shape[1]
        goal_decoder_hidden_state = input_hidden_state
        goal_decoder_cell_state = torch.zeros_like(goal_decoder_hidden_state).cuda()

        goal_real_embedding = self.inputEmbedding(goal_real.contiguous().view(-1, 2))
        goal_real_embedding = goal_real_embedding.view(-1, batch, self.goal_decoder_input_dim)

        if self.training:
            for i, input_data in enumerate(
                    goal_real_embedding[-self.pred_len:].chunk(
                        goal_real_embedding[-self.pred_len:].size(0), dim=0
                    )
            ):
                teacher_forcing = random.random() < teacher_forcing_ratio
                input_data = input_data if teacher_forcing else output
                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                output = self.hidden2pos(goal_decoder_hidden_state)
                pred_goal += [output]
        else:
            for i in range(self.pred_len):
                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    output, (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                output = self.hidden2pos(goal_decoder_hidden_state)
                pred_goal += [output]

        # outputs = torch.stack(pred_goal)    # 是否需要这步操作 ?
        return pred_goal


class GoalAttention(nn.Module):
    def __init__(self, action_hidden_state_dim):
        super(GoalAttention, self).__init__()
        self.fc = nn.Linear(2, action_hidden_state_dim, bias=True)
        self.weight = nn.Softmax()

    def forward(self, action_decoder_hidden_state, goal):   # 维度问题,goal是否包含人数这一维度(即1413),此处计算时是否需要加dim= ?
        return action_decoder_hidden_state.mul(self.weight(self.fc(goal)))


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
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

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

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


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GraphAttention(nn.Module):    # 存在的问题: attention未考虑相关位置,action的余弦距离等 ?
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.gat = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, action, seq_start_end):
        graphAtt = []
        for start, end in seq_start_end.data:
            curr_action = action[:, start:end, :]
            curr_graph_embedding = self.gat(curr_action)
            graphAtt.append(curr_graph_embedding)
        graphAtt = torch.cat(graphAtt, dim=1)
        return graphAtt


class ActionDecoder(nn.Module):
    def __init__(
            self, pred_len, action_decoder_input_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout, alpha,
    ):
        super(ActionDecoder, self).__init__()
        self.pred_len = pred_len
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.inputEmbedding = nn.Linear(2, self.action_decoder_input_dim)
        self.goalAttention = GoalAttention(
            action_hidden_state_dim=self.action_encoder_hidden_dim,
        )
        self.graphAttention = GraphAttention(   # 是否存在问题 ?
            n_units, n_heads, dropout, alpha
        )
        self.actionDecoderLSTM = nn.LSTMCell(self.action_decoder_input_dim, self.action_decoder_hidden_dim)
        self.hidden2pos = nn.Linear(self.action_decoder_hidden_dim, 2)

    def forward(self, action_real, action_encoder_hidden_state, pred_goal, seq_start_end, teacher_forcing_ratio):
        pred_action = []
        output = action_real[-self.pred_len-1]
        batch = action_real.shape[1]
        action_decoder_hidden_state = action_encoder_hidden_state
        action_decoder_cell_state = torch.zeros_like(action_decoder_hidden_state).cuda()

        action_real_embedding = self.inputEmbedding(action_real.contiguous().view(-1, 2))
        action_real_embedding = action_real_embedding.view(-1, batch, self.action_decoder_input_dim)

        if self.training:
            for i, input_data in enumerate(
                action_real_embedding[-self.pred_len :].chunk(
                    action_real_embedding[-self.pred_len :].size(0), dim=0
                )
            ):
                teacher_forcing = random.random() < teacher_forcing_ratio
                input_data = input_data if teacher_forcing else output
                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(    # LSTM
                    input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )
                action_decoder_hidden_state = self.goalAttention(action_decoder_hidden_state, pred_goal[i])   # goal attention
                action_decoder_hidden_state = self.graphAttention(action_decoder_hidden_state, seq_start_end)     # graph attention
                output = self.hidden2pos(action_decoder_hidden_state)
                pred_action += [output]
        else:
            for i in range(self.pred_len):
                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(  # LSTM
                    output, (action_decoder_hidden_state, action_decoder_cell_state)
                )
                action_decoder_hidden_state = self.goalAttention(action_decoder_hidden_state, pred_goal[i])  # goal attention
                action_decoder_hidden_state = self.graphAttention(action_decoder_hidden_state, seq_start_end)  # graph attention
                output = self.hidden2pos(action_decoder_hidden_state)
                pred_action += [output]

        return pred_action


class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_encoder_input_dim, action_encoder_hidden_dim,
            goal_encoder_input_dim, goal_encoder_hidden_dim,
            goal_decoder_input_dim, goal_decoder_hidden_dim,
            action_decoder_input_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout, alpha,
            noise_dim=(8,), noise_type="gaussian",
    ):
        super(TrajectoryPrediction, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.action_encoder_input_dim = action_encoder_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim
        self.goal_encoder_input_dim = goal_encoder_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim
        self.goal_decoder_input_dim = goal_decoder_input_dim
        self.goal_decoder_hidden_dim = goal_decoder_hidden_dim
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.actionEncoder = ActionEncoder(
            obs_len=self.obs_len,
            action_encoder_input_dim=self.action_encoder_input_dim,
            action_encoder_hidden_dim=self.action_encoder_hidden_dim,
        )
        self.goalEncoder = GoalEncoder(
            obs_len=self.obs_len,
            goal_encoder_input_dim=self.goal_encoder_input_dim,
            goal_encoder_hidden_dim=self.goal_encoder_hidden_dim,
        )
        self.goalDecoder = GoalDecoder(
            pred_len=self.pred_len,
            goal_decoder_input_dim=self.goal_decoder_input_dim,
            goal_decoder_hidden_dim=self.goal_decoder_hidden_dim,
        )
        self.actionDecoder = ActionDecoder(
            pred_len=self.pred_len,
            action_decoder_input_dim=self.action_decoder_input_dim,
            action_decoder_hidden_dim=self.action_decoder_hidden_dim,
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha,
        )

    def forward(self, input_traj, input_goal, seq_start_end, teacher_forcing_ratio, training_step):     # 缺少 add noise...

        goal_encoder_hidden_state = self.goalEncoder(input_goal)

        pred_goal = self.goalDecoder(input_goal, goal_encoder_hidden_state, teacher_forcing_ratio)  # 输入input_goal有问题,需要分training_step ?

        if training_step == 1:  # training goal encoder-decoder
            return pred_goal
        else:   # training action encoder-decoder with goal attention
            action_encoder_hidden_state = self.actionEncoder(input_traj)
            pred_action = self.actionDecoder(
                input_traj, action_encoder_hidden_state, pred_goal, seq_start_end, teacher_forcing_ratio
            )
            return pred_action

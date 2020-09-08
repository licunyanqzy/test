import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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


class AutomaticWeightLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)    # 初始值如何设置 ?
        self.params = nn.Parameter(params)

    def forward(self, loss):
        weight_loss = 0
        for i in range(len(loss)):
            weight_loss += 0.5 / (self.params[i] ** 2) * loss[i] + torch.log(self.params[i])
        return weight_loss


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
    """
    交互模块, 刻画其他人的影响
    """
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


class GoalAttention(nn.Module):
    """
    goal 指导生成 action
    """
    def __init__(self, action_hidden_state_dim):
        super(GoalAttention, self).__init__()
        self.fc = nn.Linear(2, action_hidden_state_dim, bias=True)
        self.weight = nn.Softmax()

    def forward(self, action_decoder_hidden_state, goal):   # 维度问题,goal是否包含人数这一维度(即1413),此处计算时是否需要加dim= ?
        return action_decoder_hidden_state.mul(self.weight(self.fc(goal)))


class ActionAttention(nn.Module):
    """
    action 反作用于 goal, 约束下一时刻的goal
    """
    def __init__(self, goal_hidden_state_dim):
        super(ActionAttention, self).__init__()
        self.fc = nn.Linear(2, goal_hidden_state_dim, bias=True)
        self.weight = nn.Softmax()

    def forward(self, goal_decoder_hidden_state, action):   # 是否存在类似维度问题 ?
        return goal_decoder_hidden_state.mul(self.weight(self.fc(action)))


class Decoder(nn.Module):
    def __init__(
            self, obs_len, pred_len, n_units, n_heads, dropout, alpha,
            action_decoder_input_dim, action_decoder_hidden_dim, goal_decoder_input_dim, goal_decoder_hidden_dim
    ):
        super(Decoder, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.goal_decoder_input_dim = goal_decoder_input_dim
        self.goal_decoder_hidden_dim = goal_decoder_hidden_dim
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        # add noise 之后, 维度需要随之变化, to be continued...

        self.goalDecoderLSTM = nn.LSTMCell(self.goal_decoder_input_dim, self.goal_decoder_hidden_dim)
        self.hidden2goal = nn.Linear(self.goal_decoder_hidden_dim, 2)
        self.goalEmbedding = nn.Linear(2, self.goal_decoder_input_dim)

        self.actionDecoderLSTM = nn.LSTMCell(self.action_decoder_input_dim, self.action_decoder_hidden_dim)
        self.actionEmbedding = nn.Linear(2, self.action_decoder_input_dim)
        self.hidden2action = nn.Linear(self.action_decoder_hidden_dim, 2)

        self.goalAttention = GoalAttention(
            action_hidden_state_dim=self.action_encoder_hidden_dim,
        )
        self.actionAttention = ActionAttention(
            goal_hidden_state_dim=self.goal_decoder_hidden_dim,
        )
        self.graphAttention = GraphAttention(  # 是否存在问题 ?
            n_units, n_heads, dropout, alpha
        )

    def forward(
            self, teacher_forcing_ratio, seq_start_end,
            goal_real, goal_input_hidden_state,
            action_real, action_input_hidden_state,
    ):

        goal_decoder_hidden_state = goal_input_hidden_state
        goal_decoder_cell_state = torch.zeros_like(goal_decoder_hidden_state).cuda()
        action_decoder_hidden_state = action_input_hidden_state
        action_decoder_cell_state = torch.zeros_like(action_decoder_hidden_state).cuda()

        pred_goal = []
        goal_output = goal_real[self.obs_len - 1]
        pred_action = []
        action_output = action_real[self.obs_len - 1]
        # 需要维护 pred_traj, pred_action 两个列表, to be continued...
        pred_traj = []

        batch = action_real.shape[1]
        action_real_embedding = self.actionEmbedding(action_real.contiguous().view(-1, 2))
        action_real_embedding = action_real_embedding.view(-1, batch, self.action_decoder_input_dim)
        action_real_embedding_chunk = action_real_embedding[-self.pred_len:].chunk(
            action_real_embedding[-self.pred_len:].size(0), dim=0
        )
        goal_real_embedding = self.inputEmbedding(goal_real.contiguous().view(-1, 2))
        goal_real_embedding = goal_real_embedding.view(-1, batch, self.goal_decoder_input_dim)
        goal_real_embedding_chunk = goal_real_embedding[-self.pred_len:].chunk(
            goal_real_embedding[-self.pred_len:].size(0), dim=0
        )

        if self.training:
            for i in range(self.pred_len):
                action_input_data = action_real_embedding_chunk[i]
                goal_input_data = goal_real_embedding_chunk[i]

                teacher_forcing = random.random() < teacher_forcing_ratio
                if teacher_forcing:
                    action_input_data = action_input_data
                    goal_input_data = goal_input_data
                else:
                    action_input_data = action_output
                    goal_input_data = goal_output

                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    goal_input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)
                pred_goal += [goal_output]
                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    action_input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )
                action_decoder_hidden_state = self.goalAttention(action_decoder_hidden_state, goal_output)
                action_decoder_hidden_state = self.graphAttention(action_decoder_hidden_state, seq_start_end)
                action_output = self.hidden2action(action_decoder_hidden_state)
                pred_action += [action_output]
                goal_decoder_hidden_state = self.actionAttention(goal_decoder_hidden_state, action_output)
        else:
            for i in range(self.pred_len):
                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    goal_output.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)
                pred_goal += [goal_output]
                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    action_output.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )
                action_decoder_hidden_state = self.goalAttention(action_decoder_hidden_state, goal_output)
                action_decoder_hidden_state = self.graphAttention(action_decoder_hidden_state, seq_start_end)
                action_output = self.hidden2action(action_decoder_hidden_state)
                pred_action += [action_output]
                goal_decoder_hidden_state = self.actionAttention(goal_decoder_hidden_state, action_output)

        pred_goal_output = torch.stack(pred_goal)
        pred_action_output = torch.stack(pred_action)
        return pred_goal_output, pred_action_output


class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_encoder_input_dim, action_encoder_hidden_dim,
            goal_encoder_input_dim, goal_encoder_hidden_dim,
            goal_decoder_input_dim, action_decoder_input_dim,
            # action_decoder_hidden_dim, goal_decoder_hidden_dim,
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
        self.action_decoder_input_dim = action_decoder_input_dim
        # self.goal_decoder_hidden_dim = goal_decoder_hidden_dim
        # self.action_decoder_hidden_dim = action_decoder_hidden_dim
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.goal_decoder_hidden_dim = self.goal_encoder_hidden_dim + self.noise_dim[0]
        self.action_decoder_hidden_dim = self.action_encoder_hidden_dim + self.noise_dim[0]

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
        self.Decoder = Decoder(
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            goal_decoder_input_dim=self.goal_decoder_input_dim,
            goal_decoder_hidden_dim=self.goal_decoder_hidden_dim,
            action_decoder_input_dim=self.action_decoder_input_dim,
            action_decoder_hidden_dim=self.action_decoder_hidden_dim,
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
        goal_encoder_hidden_state = self.goalEncoder(input_goal)
        action_encoder_hidden_state = self.actionEncoder(input_action)

        # add noise 噪声加的是否正确 ?
        goal_hidden_state_noise = self.add_noise(goal_encoder_hidden_state, seq_start_end)
        action_hidden_state_noise = self.add_noise(action_encoder_hidden_state, seq_start_end)

        pred_goal, pred_action = self.Decoder(
            input_goal, goal_hidden_state_noise,
            input_action, action_hidden_state_noise,
            teacher_forcing_ratio, seq_start_end,
        )

        return pred_goal, pred_action




















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


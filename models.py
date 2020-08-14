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

    def forward(self, action_decoder_hidden_state, goal):
        return action_decoder_hidden_state.mul(self.weight(self.fc(goal)))


class GraphAttention(nn.Module):
    def __init__(self, ):
        super(GraphAttention, self).__init__()

    def forward(self, action, seq_start_end):

        for start, end in seq_start_end.data:
            curr_action = action[:, start:end, :]




class ActionDecoder(nn.Module):
    def __init__(
            self, pred_len, action_encoder_hidden_state, action_decoder_input_dim, action_decoder_hidden_dim
    ):
        super(ActionDecoder, self).__init__()
        self.pred_len = pred_len
        self.action_encoder_hidden_state = action_encoder_hidden_state
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.inputEmbedding = nn.Linear(2, self.action_decoder_input_dim)
        self.goalAttention = GoalAttention(
            action_hidden_state_dim=self.action_encoder_hidden_dim,
        )
        self.graphAttention = GraphAttention(   # 实例化 GraphAttention...

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
                action_decoder_hidden_state = self.graphAttention(action_decoder_hidden_state, )     # graph attention...
                output = self.hidden2pos(action_decoder_hidden_state)
                pred_action += [output]
        else:
            for i in range(self.pred_len):



class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_encoder_input_dim, action_encoder_hidden_dim,
            goal_encoder_input_dim, goal_encoder_hidden_dim,
            goal_decoder_input_dim, goal_decoder_hidden_dim,
            dropout, noise_dim=(8,), noise_type="gaussian",
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

    def forward(self, input_traj, input_goal, seq_start_end, teacher_forcing_ratio, training_step):

        goal_encoder_hidden_state = self.goalEncoder(input_goal)

        pred_goal = self.goalDecoder(input_goal, goal_encoder_hidden_state, teacher_forcing_ratio)  # 输入input_goal有问题,需要分training_step ?

        if training_step == 1:  # training goal encoder-decoder
            return pred_goal
        else:   # training action encoder-decoder with goal attention
            action_encoder_hidden_state = self.actionEncoder(input_traj)


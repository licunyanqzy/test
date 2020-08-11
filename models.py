import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self, obs_len, action_encoder_input_dim, action_encoder_hidden_dim, embedding_dim,
    ):
        super(ActionEncoder, self).__init__()
        self.obs_len = obs_len
        self.action_encoder_input_dim = action_encoder_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim
        self.embedding_dim = embedding_dim

        self.actionEncoderLSTM = nn.LSTMCell(self.action_encoder_input_dim, self.action_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        return (        # torch.zeros or torch.randn ?
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
        )

    def forward(self, obs_traj):    # 先embedding,后LSTM,可能存在维度问题 ?
        batch = obs_traj.shape[1]
        action_lstm_hidden_state, action_lstm_cell_state = self.init_hidden(batch)

        obs_traj_embedding = self.inputEmbedding(obs_traj.contiguous().view(-1,2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)

        for i, input_data in enumerate(
            obs_traj_embedding[: self.obs_len].chunk(
                obs_traj_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            action_lstm_hidden_state, action_lstm_cell_state = self.actionEncoderLSTM(
                input_data.squeeze(0), (action_lstm_hidden_state, action_lstm_cell_state)
            )

        return action_lstm_hidden_state


class GoalEncoder(nn.Module):
    def __init__(
            self, obs_len, goal_encoder_input_dim, goal_encoder_hidden_dim, embedding_dim
    ):
        super(GoalEncoder, self).__init__()
        self.obs_len = obs_len
        self.goal_encoder_input_dim = goal_encoder_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim
        self.embedding_dim = embedding_dim

        self.goalEncoderLSTM = nn.LSTMCell(self.goal_encoder_input_dim, self.goal_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda()
        )

    def forward(self, obs_goal):    # 先embedding,后LSTM,可能存在维度问题 ?
        batch = obs_goal.shape[1]
        goal_lstm_hidden_state, goal_lstm_cell_state = self.init_hidden(batch)

        obs_goal_embedding = self.inputEmbedding(obs_goal.contiguous().view(-1, 2))
        obs_goal_embedding = obs_goal_embedding.view(-1, batch, self.embedding_dim)

        for i, input_data in enumerate(
            obs_goal_embedding[: self.obs_len].chunk(
                obs_goal_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            goal_lstm_hidden_state, goal_lstm_cell_state = self.goalEncoderLSTM(
                input_data.squeeze(0), (goal_lstm_hidden_state, goal_lstm_cell_state)
            )

        return goal_lstm_hidden_state


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

    def forward(self, ):



class Attention(nn.Module):
    def __init__(self, ):
        super(Attention, self).__init__()



class ActionDecoder(nn.Module):
    def __init__(self, ):
        super(ActionDecoder, self).__init__()



class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_encoder_input_dim, action_encoder_hidden_dim, action_embedding_dim,
            goal_encoder_input_dim, goal_encoder_hidden_dim, goal_embedding_dim,
            dropout, noise_dim=(8,), noise_type="gaussian",
    ):
        super(TrajectoryPrediction, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.action_encoder_input_dim = action_encoder_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim
        self.action_embedding = action_embedding_dim
        self.goal_encoder_input_dim = goal_encoder_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim
        self.goal_embedding_dim = goal_embedding_dim

        self.actionEncoder = ActionEncoder(
            obs_len=self.obs_len,
            action_encoder_input_dim=self.action_encoder_input_dim,
            action_encoder_hidden_dim=self.action_encoder_hidden_dim,
            embedding_dim=self.action_embedding
        )
        self.goalEncoder = GoalEncoder(
            obs_len=self.obs_len,
            goal_encoder_input_dim=self.goal_encoder_input_dim,
            goal_encoder_hidden_dim=self.goal_encoder_hidden_dim,
            embedding_dim=self.goal_embedding_dim
        )






    def forward(self, training_step, ):

















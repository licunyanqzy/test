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
            self, obs_len, action_lstm_input_dim, action_lstm_hidden_dim,
    ):
        super(ActionEncoder, self).__init__()
        self.obs_len = obs_len
        self.action_lstm_input_dim = action_lstm_input_dim
        self.action_lstm_hidden_dim = action_lstm_hidden_dim

        self.actionLSTM = nn.LSTMCell(self.action_lstm_input_dim, self.action_lstm_hidden_dim)

    def init_hidden(self, batch):
        return (        # torch.zeros or torch.randn ?
            torch.randn(batch, self.action_lstm_hidden_dim).cuda(),
            torch.randn(batch, self.action_lstm_hidden_dim).cuda(),
        )

    def forward(self, obs_traj):
        batch = obs_traj.shape[1]
        action_lstm_hidden_state, action_lstm_cell_state = self.init_hidden(batch)

        for i, input_data in enumerate(
            obs_traj[: self.obs_len].chunk(
                obs_traj[: self.obs_len].size(0), dim=0
            )
        ):
            action_lstm_hidden_state, action_lstm_cell_state = self.actionLSTM(
                input_data.squeeze(0), (action_lstm_hidden_state, action_lstm_cell_state)
            )

        return action_lstm_hidden_state


class GoalEncoder(nn.Module):
    def __init__(
            self, obs_len, goal_lstm_input_dim, goal_lstm_hidden_dim,
    ):
        super(GoalEncoder, self).__init__()
        self.obs_len = obs_len
        self.goal_lstm_input_dim = goal_lstm_input_dim
        self.goal_lstm_hidden_dim = goal_lstm_hidden_dim

        self. goalLSTM = nn.LSTMCell(self.goal_lstm_input_dim, self.goal_lstm_hidden_dim)

    def init_hidden(self, batch):
        return (
            torch.randn(batch, self.goal_lstm_hidden_dim).cuda(),
            torch.randn(batch, self.goal_lstm_hidden_dim).cuda()
        )

    def forward(self, obs_goal):
        batch = obs_goal.shape[1]
        goal_lstm_hidden_state, goal_lstm_cell_state = self.init_hidden(batch)

        for i, input_data in enumerate(
            obs_goal[: self.obs_len].chunk(
                obs_goal[: self.obs_len].size(0), dim=0
            )
        ):
            goal_lstm_hidden_state, goal_lstm_cell_state = self.goalLSTM(
                obs_goal, (goal_lstm_hidden_state, goal_lstm_cell_state)
            )

        return goal_lstm_hidden_state



class Attention(nn.Module):
    def __init__(self, ):
        super(Attention, self).__init__()



class ActionDecoder(nn.Module):




class GoalDecoder(nn.Module):
    def __init__(self, ):
        super(GoalDecoder, self).__init__()



class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_lstm_input_dim, action_lstm_hidden_dim, goal_lstm_input_dim, goal_lstm_hidden_dim,
            dropout, noise_dim=(8,), noise_type="gaussian",
    ):
        super(TrajectoryPrediction, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.actionEncoder = ActionEncoder()
        self.goalEncoder = GoalEncoder()
        self.attention = Attention()





    def forward(self, ):

















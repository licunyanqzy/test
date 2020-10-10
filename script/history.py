import torch
import torch.nn as nn


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


class InteractionGate(nn.Module):
    def __init__(self, distance_embedding_dim, action_decoder_hidden_dim):
        super(InteractionGate, self).__init__()
        self.embedding = nn.Linear(1, distance_embedding_dim)
        self.fc = nn.Linear(
            distance_embedding_dim * 2 + action_decoder_hidden_dim * 2, action_decoder_hidden_dim
        )
        self.gate = nn.Sigmoid()

    def cal_distance(self, a1, a2, g1, g2):
        dn = (a1[0]-a1[0]) * (g1[1]-g2[1]) - (a1[1]-a2[1]) * (g1[0]-g2[0])
        n1 = (a1[0]*a2[1] - a1[1]*a2[0]) * (g1[0]-g2[0]) - (a1[0]-a2[0]) * (g1[0]*g2[1] - g1[0]*g2[0])
        n2 = (a1[0]*a2[1] - a1[1]*a2[0]) * (g1[1]-g2[1]) - (a1[1]-a2[0]) * (g1[0]*g2[1] - g1[0]*g2[0])

        d1 = torch.norm(a1 - g1)
        d2 = torch.norm(a2 - g2)

        if not dn == 0:
            p1 = n1 / dn
            p2 = n2 / dn
            if (a1[0] - p1) * (p1 - g1[0]) > 0:
                d1 = torch.norm(a1 - [p1, p2])
                d2 = torch.norm(a2 - [p1, p2])

        return d1.unsqueeze(0), d2.unsqueeze(0)

    def forward(self, action_hidden_state, goal_hidden_state, goal, position, idx):
        n = goal.size()[0]

        for i in range(n):
            d1, d2 = self.cal_distance(position[idx], goal[i], position[idx], position[i])
            if i == idx:
                continue

            d1_embedding = self.embedding(d1)
            d2_embedding = self.embedding(d2)
            fc_input = torch.cat(
                [action_hidden_state[idx], action_hidden_state[i], d1_embedding, d2_embedding], dim=0
            )
            action_hidden_state[i] = self.gate(self.fc(fc_input))

        d = torch.norm(position[idx] - goal[idx]).unsqueeze(0)
        d_embedding = self.embedding(d)
        fc_input = torch.cat(
            [action_hidden_state[idx], goal_hidden_state, d_embedding, d_embedding], dim=0
        )
        goal_hidden_state = self.gate(self.fc(fc_input))

        return action_hidden_state, goal_hidden_state


class GAT(nn.Module):
    def __init__(
            self, distance_embedding_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout=0.2, alpha=0.2,
    ):
        super(GAT, self).__init__()
        self.n_heads = 4

        self.embedding = nn.Linear(1, distance_embedding_dim)
        self.interactionGate = nn.Sequential(
            nn.Linear(distance_embedding_dim * 2 + action_decoder_hidden_dim * 2, action_decoder_hidden_dim),
            nn.Sigmoid()
        )

        self.gatLayer = GraphAttentionLayer(
            n_head=self.n_heads, f_in=action_decoder_hidden_dim,
            f_out=action_decoder_hidden_dim, attn_dropout=dropout,
        )

    def cal_dist(self, goal, action):
        n = goal.size()[0]
        distance = torch.zeros([n, n]).cuda()
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance[i, j] = torch.norm(action[i] - goal[i])
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

                    distance[i, j] = d1
                    distance[j, i] = d2
        return distance

    def forward(self, action_hidden_state, goal_hidden_state, goal, action):
        # 只能进行一层GAT, 如何实现两层 ? to be continued ...

        n = action_hidden_state.size()[0]
        outputs = []
        distance = self.cal_dist(goal, action)

        for idx in range(n):
            curr_action_hidden_state = []
            for i in range(n):
                if i == idx:
                    d_embedding = self.embedding(distance[i, i].unsqueeze(0))
                    gate = self.interactionGate(
                        torch.cat([
                            action_hidden_state[idx], goal_hidden_state[idx], d_embedding, d_embedding
                        ], dim=0)
                    )
                    curr_goal_hidden_state = goal_hidden_state[idx] * gate
                    curr_action_hidden_state += [action_hidden_state[idx]]
                else:
                    d1_embedding = self.embedding(distance[idx, i].unsqueeze(0))
                    d2_embedding = self.embedding(distance[i, idx].unsqueeze(0))
                    gate = self.interactionGate(
                        torch.cat([
                            action_hidden_state[idx], action_hidden_state[i], d1_embedding, d2_embedding
                        ], dim=0)
                    )
                    gated_hidden_state = action_hidden_state[i] * gate
                    curr_action_hidden_state += [gated_hidden_state]

            curr_action_hidden_state = torch.stack(curr_action_hidden_state)
            gat_input = torch.cat([curr_action_hidden_state, curr_goal_hidden_state.unsqueeze(0)])
            # gat_input = torch.cat([action_hidden_state, goal_hidden_state[idx].unsqueeze(0)])
            gat_output = self.gatLayer(gat_input, idx)
            gat_output = F.elu(gat_output)
            outputs += [gat_output[idx]]

        outputs = torch.stack(outputs)
        return outputs
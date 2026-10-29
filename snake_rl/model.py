"""Q-network and the trainer that updates it."""

import os

import numpy as np
import torch
import torch.nn as nn


class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        os.makedirs(model_folder_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_folder_path, file_name))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(np.array(action), dtype=torch.float32)
        reward = torch.tensor(np.array(reward), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)

        if len(state.shape) == 1:
            # single sample: (x,) -> (1, x)
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            done = (done,)

        # 1) predicted Q values for the current state
        pred = self.model(state)

        # 2) Bellman target: Q_new = r + gamma * max(Q(s', a')), only for the action taken
        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        loss = self.loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

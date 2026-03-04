import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file contains the code for our A2C Network, which makes decisions for the
tracing agent. It takes in the state information produced in the step function of
tracing_env, and outputs probabilities for 4 actions (up, right, down, left)

This code does not need to be modified, and its predictions can be accessed via
the 'act' function

Ported Tensorflow v1 to pytorch. -- Andrew Chang, Feb 25th
"""

class PolicyEstimator_RNN(nn.Module):
    def __init__(self, state_size, action_size, target=None, device='cpu'):
        super(PolicyEstimator_RNN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.target_mode = (target is not None)
        self.device = device

        #local_map shape: (None, 5, 625)
        self.lm_dense1 = nn.Linear(625, 100)
        self.lm_dense2 = nn.Linear(100, 100)

        #state_shape: (None, 5, state_size)
        self.state_dense1 = nn.Linear(self.state_size, 64)
        self.state_dense2 = nn.Linear(64, 10)

        self.rnn_cell = nn.LSTM(110, 100)

        self.output = nn.Linear(5*100, self.action_size)
        self.action_probs = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, local_maps=None):
        """
        state:
            target_mode=False -> (batch, 5, state_size)
            target_mode=True  -> (batch, state_size)

        local_map:
            (batch, 5, 625)
        """
        if self.target_mode:
            x = F.relu(self.state_dense1(state))
            x = F.relu(self.state_dense2(x))
            return self.output(x)

        # Possible Model Improvement area: change the local map model into a CNN or something else with spatial encoding. --Andrew Chang, Feb 25 2026

        lm = F.relu(self.lm_dense1(local_maps))
        lm = F.relu(self.lm_dense2(lm))

        s = F.relu(self.state_dense1(state))
        s = F.relu(self.state_dense2(s))

        x = torch.concat((s, lm), dim=2) # (batch, 5, 110)

        rnn_out, (h_n, c_n) = self.rnn_cell(x)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)

        output = self.output(rnn_out)

        action_probs = self.action_probs(output)
        action_probs = torch.squeeze(action_probs)

        return action_probs 

    def predict(self, states, local_map, sess=None):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(states).to(self.device)
            if local_map is not None:
                local_map = torch.FloatTensor(local_map).to(self.device)
                action_probs = self.forward(state, local_map)
            else:
                action_probs = self.forward(state)
        return action_probs.cpu().numpy()

    def update(self, state, target, local_maps=None):
        self.train()
        state = torch.FloatTensor(state).to(self.device)
        target = torch.FloatTensor(target).to(self.device)

        if local_maps is not None:
            local_map = torch.FloatTensor(local_map).to(self.device)
            output = self.forward(state, local_map)
        else:
            output = self.forward(state)

        loss = F.mse_loss(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, name):
        torch.save(self.state_dict(), name)



class ValueEstimator_RNN(nn.Module):
    def __init__(self, state_size, action_size, target=None, device='cpu'):
        super(ValueEstimator_RNN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.target_mode = (target is not None)
        self.device = device

        #local_map shape: (None, 5, 625)
        self.lm_dense1 = nn.Linear(625*5, 100)
        self.lm_dense2 = nn.Linear(100, 100)

        #state_shape: (None, 5, state_size)
        self.state_dense1 = nn.Linear(self.state_size, 64)
        self.state_dense2 = nn.Linear(64, 10)

        self.rnn_cell = nn.LSTM(110, 100)

        self.output = nn.Linear(5*110, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, states, local_maps):
        """
        state:
            target_mode=False -> (batch, 5, state_size)
            target_mode=True  -> (batch, state_size)

        local_map:
            (batch, 5, 625)
        """
        if self.target_mode:
            x = F.relu(self.state_dense1(state))
            x = F.relu(self.state_dense2(x))
            return self.output(x)

        # Possible Model Improvement area: change the local map model into a CNN or something else with spatial encoding. --Andrew Chang, Feb 25 2026

        flattened_lm = torch.flatten(local_maps)
        lm = F.relu(self.lm_dense1(flattened_lm))
        lm = F.relu(self.lm_dense2(lm))

        s = F.relu(self.state_dense1(state))
        s = F.relu(self.state_dense2(s))

        x = torch.concat((s, lm), dim=2) # (batch, 5, 110)

        rnn_out, (h_n, c_n) = self.rnn_cell(x)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)

        output = self.output(rnn_out)

        value_estimate = torch.squeeze(output)
        return value_estimate 

    def predict(self, states, target, local_maps):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if local_map is not None:
                local_map = torch.FloatTensor(local_map).to(self.device)
                value_estimate = self.forward(state, local_map)
            else:
                value_estimate = self.forward(state)
        return value_estimate.cpu().numpy()

    def update(self, state, target, local_maps=None):
        self.train()
        state = torch.FloatTensor(state).to(self.device)
        target = torch.FloatTensor(target).to(self.device)

        if local_maps is not None:
            local_map = torch.FloatTensor(local_map).to(self.device)
            value_estimate = self.forward(state, local_map)
        else:
            value_estimate = self.forward(state)

        loss = F.mse_loss(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, name):
        torch.save(self.state_dict(), name)

class A2CAgent:
    def __init__(self, state_size, action_size, target=None):
        self.state_size = state_size
        self.policy = PolicyEstimator_RNN(state_size, action_size, target)
        self.value = ValueEstimator_RNN(state_size, action_size, target)
        self.memory = deque(maxlen=2000)

    def act(self, state, local_map):
        action_probs = self.policy.predict(state, local_map)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def memorize(self, state, local_map, action, reward, next_state, next_local_map, done):
        self.memory.append((state, local_map, action, reward, next_state, next_local_map, done))

    def update(self, states, local_maps, pol_target, val_target, action):
        self.policy.update(states, pol_target, action, local_maps)
        self.value.update(states, val_target, local_maps)

    def load(self, policy_name, value_name):
        self.policy = torch.load(policy_name)
        self.value = torch.load(value_name)

    def save(self, policy_name, value_name, episode=None):
        self.policy.save_weights(policy_name)
        self.value.save_weights(value_name)


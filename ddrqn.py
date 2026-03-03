import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
"""
This file contains the code for our implementation of the DDRQN architecture. This network is used for
the search agent's predictions. This code does not need to be changed. To access its predictions, use the
'act' function

Ported Tensorflow v1 to pytorch. -- Andrew Chang, Feb 25th
"""

class DDRQNModel(nn.Module):
    def __init__(self, state_size, action_size, target=None):
        self.state_size = state_size
        self.action_size = action_size
        self.target_mode = (target is not None) #When target is None, do not receive local map \
                # If not, receive local map.

# Possible Model Improvement area: change the local map model into a CNN or something else with spatial encoding. --Andrew Chang, Feb 25 2026
        if not self.target_mode:
            self.lm_dense1 = nn.Linear(625, 100) # lm_dense1, lm_dense2 receives local map
            self.lm_dense2 = nn.Linear(100, 100)

        # If self.target_mode is True, input state is (batch, state_size).
        # If self.target_mode is False, state is (batch, 5, state_size).
        self.state_dense1 = nn.Linear(state_size, 64)
        self.state_dense2 = nn.Linear(64, 10)

        if self.target_mode:
            self.output = nn.Linear(10, self.action_size)
        else:
            self.rnn_cell = nn.LSTM(110, 110, batch_first=True)
            self.output = nn.Linear(5*110, self.action_size)

        self.optimizer = optimizer.Adam(self.parameters(), lr=0.001)

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
        flattened_lm = torch.flatten(local_maps)
        lm = F.relu(self.lm_dense1(flattened_lm))
        lm = F.relu(self.lm_dense2(lm))

        s = F.relu(self.state_dense1(state))
        s = F.relu(self.state_dense2(s))

        x = torch.concat((s, lm), dim=2) # (batch, 5, 110)

        rnn_out, (h_n, c_n) = self.rnn_cell(x)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)

        return self.output(rnn_out)

    def predict(self, state, local_maps=None):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if local_map is not None:
                local_map = torch.FloatTensor(local_map).to(self.device)
                q_values = self.forward(state, local_map)
            else:
                q_values = self.forward(state)
        return q_values.cpu().numpy()

    def fit(self, state, target, local_maps=None):
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

        return loss.item()

    def load_weights(self, name):
        torch.save(self.state_dict(), name)

    def save_weights(self, name, sess, episode=None):
        self.saver.save(sess, name, global_step=episode)


class DDRQNAgent:
    def __init__(self, state_size, action_size, scope, session, target=None):
        self.scope = scope
        self.model = DDRQNModel(state_size, action_size, target)
        self.target_model = DDRQNModel(state_size, action_size, target)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.sess = session

        self.load_weight_dir = "Weights/"
        self.save_weight_dir = "Weights_full/"
        self.temp_weight_dir = "Weights_temp/"

        self.make_dirs()

        self.update_target_model()

        if target is None:
            self.isTarget = False
        else:
            self.isTarget = True


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        if self.isTarget:
            for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
                target = self.model.predict(state)
                target_next = self.model.predict(next_state)
                target_val = self.target_model.predict(next_state)
                if done:
                    target[0][action] = reward
                else:
                    a = np.argmax(target_next[0])
                    target[0][action] = reward + self.gamma * target_val[0][a]

                self.model.fit(state, target)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


        else:
            for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
                target = self.model.predict(state, local_map)
                target_next = self.model.predict(next_state, next_local_map)
                target_val = self.target_model.predict(next_state, next_local_map)
                if done:
                    target[0][action] = reward
                else:
                    a = np.argmax(target_next[0])
                    target[0][action] = reward + self.gamma * target_val[0][a]

                self.model.fit(state, target, local_map)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state, local_maps=None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, local_maps)
        return np.argmax(act_values[0])  # returns action

    def memorize(self, state, local_map, action, reward, next_state, next_local_map, done):
        self.memory.append((state, local_map, action, reward, next_state, next_local_map, done))

    def make_dirs(self):
        if not os.path.exists(self.load_weight_dir):
            os.makedirs(self.load_weight_dir)

        if not os.path.exists(self.save_weight_dir):
            os.makedirs(self.save_weight_dir)

        if not os.path.exists(self.temp_weight_dir):
            os.makedirs(self.temp_weight_dir)

    def load(self, name, name2):
        self.model.load_weights(self.load_weight_dir + name, self.sess)
        self.target_model.load_weights(self.load_weight_dir + name2, self.sess)

    def save(self, name, name2, episode=None):
        self.model.save_weights(self.save_weight_dir + name, self.sess, episode)
        self.target_model.save_weights(self.save_weight_dir + name2, self.sess, episode)

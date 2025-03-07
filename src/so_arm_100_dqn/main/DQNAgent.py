"""
Educational Use License

Copyright (c) 2025 UP Student IA Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, and distribute the Software solely for educational and
non-commercial purposes, subject to the following conditions:

[Include the full text as above or a reference to the LICENSE file]

This license is intended for educational purposes only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class SimpleDQN(nn.Module):

    def __init__(self, state_dim, num_actions, hidden_dim=120):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, action_dim, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.999, learning_rate=1e-3, 
                 memory_size=5000, batch_size=32, dq_trainer = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.dq_trainer = dq_trainer

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = SimpleDQN(state_dim, num_actions).to(self.device)
        self.target_net = SimpleDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0

    def get_action(self, state):

        if random.random() < self.epsilon:
            return self.dq_trainer.generate_action()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            best_action_index = torch.argmax(q_values).item()
            
            #           joint 0         |           joint 1          |           joint 2          |
            #      +             -      |      +             -       |      +             -       |
            #      Q0     |      Q1     |      Q2     |      Q3      |      Q4     |      Q5      |
            #      a             b              c             d             f              e      |
            # 
            # best_action_index = argmax (a,b,c,d,e)
            #
            #  best_action_index = b
            # e.g b = Q0
            #
            # action_list = [1,0,0,0,0,0]
            #
            #
            # state (theta0, theta1, theta2)
            #
            # action (theta0, theta1, theta2)
            #
            # theta0 (Q0, Q1)
            # theta1 (Q2, Q3)
            # theta2 (Q4, Q5)
            #
            # Action [(Q0 Q1), (Q2 Q3), (Q4 Q5)]
            # 
            #  
            action_list = [0, 0, 0, 0, 0, 0]
            return action_list

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        actions = torch.clamp(actions, min=0, max=self.num_actions - 1)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1).to(self.device)


        if actions.dim() == 3:
            actions = actions.unsqueeze(1)

        # Obtener valores de Q de la red de política
        q_values = self.policy_net(states)  # [batch_size, num_actions]
        print("q_values", q_values)
        # Seleccionar los valores Q correspondientes a las acciones tomadas
        action_indices = torch.argmax(q_values, dim=1, keepdim=True)  # [batch_size, 1]
        current_q = q_values.gather(1, action_indices)

        # Calcular Q-target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)

        # Calcular pérdida y optimizar
        loss = nn.SmoothL1Loss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # with torch.no_grad():
        #     next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # target_q = rewards + self.gamma * next_q
        # actions = actions.squeeze(1)  # Convierte [batch_size, 1, action_dim] → [batch_size, action_dim]
        # actions = actions.unsqueeze(-1)  # Convierte [batch_size, action_dim] → [batch_size, action_dim, 1]

        # loss = nn.SmoothL1Loss()(self.policy_net(states).gather(1, actions), target_q)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.steps_done += 1

        if self.steps_done % 10000 == 0:
            print(f"Step {self.steps_done}: Updating target network...")
            self.target_net.load_state_dict(self.policy_net.state_dict())

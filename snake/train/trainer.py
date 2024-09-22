import random
from collections import deque
import numpy as np
import torch 
from torch import nn
import torch.optim as optim

from snake.train.game import SnakeGameAI


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 预测 Q 值
        pred = self.model(state)

        target = pred.clone()
        # 遍历batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, trainer: QTrainer, memory_size = 100_000, batch_size = 1000):
        self.n_games = 0
        self.epsilon = 1 # 探索率
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.trainer = trainer
    
    def get_state(self, game: SnakeGameAI):
        return game.get_simple_state()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 如果超过 MAX_MEMORY，会自动移除最早的

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # 随机抽样
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state, random_factor=0.01):
        # 随机探索和模型预测之间的折中
        self.trainer.model.train()
        epsilon = max(random_factor, self.epsilon/(1+self.n_games))  # Decay epsilon
        
        final_move = [0, 0, 0]
        if random.random() < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.trainer.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def predict(self, state):
        # 不包含随机的推理
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        self.trainer.model.eval()
        prediction = self.trainer.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


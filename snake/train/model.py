import os
from torch import nn
import torch
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, output_size=3):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # QLearning中输出的实际是Q值，也就是对每个动作的奖励分数，这个分数会和直接奖励相加，作为目标Q值，所以不需要激活函数
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        model_path = f'{model_folder_path}/{file_name}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.state_dict(), model_path)

    @classmethod
    def load(cls, file_name='model.pth'):
        model_folder_path = './model'
        model_path = f'{model_folder_path}/{file_name}'
        model = cls()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

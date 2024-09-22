import pygame

from snake.train.game import SnakeGameAI
from snake.train.model import Linear_QNet
from snake.train.trainer import Agent, QTrainer


def eval():
    pygame.init()
    model = Linear_QNet.load("high.pth")
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    agent = Agent(trainer)
    game = SnakeGameAI(frameout=10000)
    game_over = False
    while not game_over:
        # 获取当前状态
        state_old = agent.get_state(game)

        # 获取动作
        final_move = agent.predict(state_old)

        # 执行动作并获取新状态
        reward, game_over, score = game.play_step(final_move, speed=60)

if __name__ == '__main__':
    eval()

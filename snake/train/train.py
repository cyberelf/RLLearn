import pygame
import tqdm

from snake.train.game import SnakeGameAI
from snake.train.model import Linear_QNet
from snake.train.trainer import Agent, QTrainer


def train():
    pygame.init()
    model = Linear_QNet(11, 256, 3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    agent = Agent(trainer)
    game = SnakeGameAI()

    for i in tqdm.tqdm(range(1000)):
        # 获取当前状态
        state_old = agent.get_state(game)

        # 获取动作
        final_move = agent.get_action(state_old)

        # 执行动作并获取新状态
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 训练短期记忆
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 记忆经验
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 训练长记忆，绘制结果
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score)
        
            # 保存模型
            if agent.n_games > 50:
                model.save(f"checkpoints/{agent.n_games}.pth")
                break
    
    model.save("model/model.pth")

if __name__ == '__main__':
    train()

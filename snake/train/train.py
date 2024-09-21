import pygame
import tqdm
from torch.utils.tensorboard import SummaryWriter

from snake.train.game import SnakeGameAI
from snake.train.model import Linear_QNet
from snake.train.trainer import Agent, QTrainer


def train(continue_training=False):
    pygame.init()
    if continue_training:
        model = Linear_QNet.load("model.pth")
    else:
        model = Linear_QNet(11, 256, 3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    agent = Agent(trainer)
    game = SnakeGameAI()

    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    writer = SummaryWriter('logs')

    try:
        for i in tqdm.tqdm(range(1000)):
            done = False
            while not done:
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

            # 训练长记忆，绘制结果
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score)
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            writer.add_scalar('Score', score, agent.n_games)
            writer.add_scalar('Mean Score', mean_score, agent.n_games)
            if score > record:
                record = score
                model.save("high.pth")

            # 保存模型
            if agent.n_games > 1 and agent.n_games % 50 == 0:
                model.save(f"{agent.n_games}.pth")
    except KeyboardInterrupt:
        print('Training interrupted')
    finally:
        model.save("model.pth")

if __name__ == '__main__':
    train(False)

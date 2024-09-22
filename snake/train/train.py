from datetime import date
import datetime
import random
from re import T
import pygame
from torch import rand
import tqdm
from torch.utils.tensorboard import SummaryWriter

from snake.train.game import SnakeGameAI
from snake.train.model import Linear_QNet
from snake.train.trainer import Agent, QTrainer


def train(continue_training=False):
    pygame.init()
    if continue_training:
        model = Linear_QNet.load("high.pth")
        game = SnakeGameAI()
        # TODO: get n_games to set a proper epsilon
    else:
        model = Linear_QNet()
        game = SnakeGameAI(frameout=2000)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    agent = Agent(trainer)

    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    writer = SummaryWriter(f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    # 要记忆的最小分数
    memory_score_th = 10

    try:
        for i in tqdm.tqdm(range(1000)):
            done = False
            long_term_candidate = []
            while not done:
                # 获取当前状态
                state_old = agent.get_state(game)

                # 获取动作
                final_move = agent.get_action(state_old, random_factor=0)

                # 执行动作并获取新状态
                reward, done, score = game.play_step(final_move)
                state_new = agent.get_state(game)

                # 训练短期记忆
                agent.train_short_memory(state_old, final_move, reward, state_new, done)

                # 记忆经验
                long_term_candidate.append((state_old, final_move, reward, state_new, done))
            
            # 只记忆分数大于 memory_score_th 的经验
            if score >= memory_score_th:
                for exp in long_term_candidate:
                    agent.remember(*exp)
                # if score > 2 * memory_score_th:
                #     memory_score_th = score // 2
            else:
                for exp in long_term_candidate[-200:]:
                    agent.remember(*exp)
                    
            # 绘制结果
            agent.n_games += 1
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

            # 训练长期记忆
            agent.train_long_memory()

    except KeyboardInterrupt:
        print('Training interrupted')
    finally:
        model.save("model.pth")



if __name__ == '__main__':
    train(False)
    # train(True)

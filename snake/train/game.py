import sys
import numpy as np
import pygame
import os
import torch.nn.functional as F

from snake.game import Game
from snake.spirits.consts import RED
from snake.spirits.obstacle import Obstacle

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

class SnakeGameAI(Game):
    def __init__(self, frameout=500):
        super().__init__()
        self.frame_iteration = 0  # 用于防止无限循环
        self.frame_noscore = 0  # 没有吃到食物的帧数
        self.failed = False  # 游戏是否结束
        # self.screen = pygame.Surface((640, 480))
        self.score = 0
        self.frameout = frameout

    def reset(self):
        super().reset()
        # 增加障碍物
        self.obstacles = [Obstacle(position=(x, y)) for x in range(270, 330, 10) for y in range(170, 230, 10)]
        # self.obstacles = [Obstacle(position=(x, 200)) for x in range(0, 270, 10)]
        # self.obstacles += [Obstacle(position=(x, 200)) for x in range(430, 630, 10)]
        self.frame_iteration = 0
        self.frame_noscore = 0
    
    def play_step(self, action, speed=500):
        self.frame_iteration += 1
        self.frame_noscore += 1
        # 只处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        old_score = self.score
        # 处理动作
        self._move(action)  # 更新蛇的方向和位置
        
        # 更新游戏状态
        self.update()
        
        # 检查游戏是否结束
        reward = 0
        game_over = False
        if self.failed:
            game_over = True
            self.failed = False
            reward = -10
            return reward, game_over, old_score
        
        if self.frame_noscore > self.frameout:
            self.game_over()
            game_over = True
            reward = -10
            return reward, game_over, old_score
        
        # 奖励机制
        if self.score > old_score:
            reward = self.score - old_score
            self.frame_noscore = 0
        else:
            reward = 0
        
        # 更新UI
        self.draw()
        self.show_debug()
        
        self.clock.tick(speed)
        
        return reward, game_over, self.score
    
    def snake_at_edge_count(self):
        # 判断蛇附近有几个边缘，作为一个风险指标
        snake = self.player_snake
        head = snake.get_head_position()
        point_l = (head[0] - 10, head[1])
        point_r = (head[0] + 10, head[1])
        point_u = (head[0], head[1] - 10)
        point_d = (head[0], head[1] + 10)

        edge_dirs = [self.is_collision(point) for point in [point_l, point_r, point_u, point_d]]
        if self.length > 1:
            return edge_dirs.count(True) - 1
        else:
            return edge_dirs.count(True)

    
    def _move(self, action):
        # action 是一个长度为3的列表：[直行，右转，左转]
        movement_action = action[:3]
        if movement_action[1]:
            self.player_snake.turn_right()
        elif movement_action[2]:
            self.player_snake.turn_left()

    def game_over(self):
        # 重载游戏结束的逻辑
        self.failed = True
        self.reset()
        self.frame_iteration = 0
        # pygame.time.wait(3000)

    def show_debug(self):
        # 显示调试信息
        font = pygame.font.SysFont('arial', 12)
        pos = font.render(f'Pos: {self.player_snake.positions[0]}', True, RED)
        len = font.render(f'Len: {self.player_snake.length}', True, RED)
        dir = font.render(f'Dir: {self.player_snake.direction}', True, RED)
        state = font.render(f'Dir: {self.get_simple_state()}', True, RED)
        self.screen.blit(pos, (10, 40))
        self.screen.blit(len, (10, 55))
        self.screen.blit(dir, (10, 70))
        self.screen.blit(state, (10, 85))
        pygame.display.flip()
    
    def get_simple_state(self):
        snake = self.player_snake
        head = snake.get_head_position()
        point_l = (head[0] - 10, head[1])
        point_r = (head[0] + 10, head[1])
        point_u = (head[0], head[1] - 10)
        point_d = (head[0], head[1] + 10)

        dir_l = snake.direction == "LEFT"
        dir_r = snake.direction == "RIGHT"
        dir_u = snake.direction == "UP"
        dir_d = snake.direction == "DOWN"

        collisions = [self.is_collision(point) for point in [point_l, point_r, point_u, point_d]]
        dirs = [dir_l, dir_r, dir_u, dir_d]
        dir_lefts = [dir_u, dir_d, dir_r, dir_l]
        dir_rights = [dir_d, dir_u, dir_l, dir_r]

        danger_straight = collisions[dirs.index(True)]
        danger_left = collisions[dir_lefts.index(True)]
        danger_right = collisions[dir_rights.index(True)]

        state = [
            # 危险的移动
            danger_straight,
            danger_right,
            danger_left,

            # 移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物位置
            self.foods[0].position[0] < head[0],  # 食物在左边
            self.foods[0].position[0] > head[0],  # 食物在右边
            self.foods[0].position[1] < head[1],  # 食物在上方
            self.foods[0].position[1] > head[1]   # 食物在下方
        ]
        return np.array(state, dtype=int)
    
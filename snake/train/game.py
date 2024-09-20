import sys
import pygame
import os

from snake.game import Game

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

class SnakeGameAI(Game):
    def __init__(self):
        super().__init__()
        self.frame_iteration = 0  # 用于防止无限循环
        self.failed = False  # 游戏是否结束
        self.display = pygame.Surface((640, 480))
        self.clock = pygame.time.Clock()
        self.reset()
    
    def play_step(self, action):
        self.frame_iteration += 1
        # 处理事件（可以忽略，因为AI控制）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        old_score = self.score
        # 处理动作
        self._move(action)  # 更新蛇的方向和位置
        self.player_snake.move()
        
        # 更新游戏状态
        self.update()
        
        # 检查游戏是否结束
        reward = 0
        game_over = False
        if self.failed:
            game_over = True
            self.failed = False
            reward = -10
            return reward, game_over, self.score
        
        # 奖励机制
        if self.score > old_score:
            reward = 10
        else:
            reward = 0
        
        # 更新UI
        self.draw()
        self.clock.tick(60)
        
        return reward, game_over, self.score
    
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
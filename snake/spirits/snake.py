import pygame
from snake.spirits.bullet import Bullet
from snake.spirits.consts import GREEN


class Snake:
    def __init__(self, position, direction, color=GREEN):
        self.positions = [position]  # 蛇的身体部分列表，包含各个方块的坐标
        self.direction = direction   # 当前方向
        self.color = color           # 蛇的颜色
        self.length = 1              # 初始长度
        self.bullets = []            # 存储蛇发射的子弹

    def move(self):
        x, y = self.positions[0]
        if self.direction == 'UP':
            y -= 10
        elif self.direction == 'DOWN':
            y += 10
        elif self.direction == 'LEFT':
            x -= 10
        elif self.direction == 'RIGHT':
            x += 10
        new_head = (x, y)
        self.positions.insert(0, new_head)
        if len(self.positions) > self.length:
            self.positions.pop()

    def change_direction(self, direction):
        # 防止蛇直接反向移动
        opposite_directions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if direction != opposite_directions.get(self.direction):
            self.direction = direction

    def turn_right(self):
        if self.direction == 'UP':
            self.direction = 'RIGHT'
        elif self.direction == 'DOWN':
            self.direction = 'LEFT'
        elif self.direction == 'LEFT':
            self.direction = 'UP'
        elif self.direction == 'RIGHT':
            self.direction = 'DOWN'
    
    def turn_left(self):
        if self.direction == 'UP':
            self.direction = 'LEFT'
        elif self.direction == 'DOWN':
            self.direction = 'RIGHT'
        elif self.direction == 'LEFT':
            self.direction = 'DOWN'
        elif self.direction == 'RIGHT':
            self.direction = 'UP'

    def grow(self):
        self.length += 1

    def draw(self, surface):
        for pos in self.positions:
            pygame.draw.rect(surface, self.color, pygame.Rect(pos[0], pos[1], 10, 10))

    def check_collision(self, obj_positions):
        # 检查是否与给定的对象列表发生碰撞
        return self.positions[0] in obj_positions

    def check_self_collision(self):
        # 检查是否撞到自己
        return self.positions[0] in self.positions[1:]

    def get_head_position(self):
        return self.positions[0]
    
    def shoot(self):
        # 发射子弹
        head_pos = self.positions[0]
        bullet = Bullet(self, position=head_pos, direction=self.direction)
        self.bullets.append(bullet)
        return bullet

import weakref
import pygame


class Bullet:
    def __init__(self, owner, position, direction, speed=15, color=(255, 165, 0)):
        """
        初始化子弹。

        参数：
        - owner: 子弹的所有者
        - position: 子弹的初始位置 (x, y)。
        - direction: 子弹的移动方向，值为 'UP'、'DOWN'、'LEFT' 或 'RIGHT'。
        - speed: 子弹的速度，默认为15像素/帧。
        - color: 子弹的颜色，默认为橙色。
        """
        # owner may die 
        self.owner = weakref.ref(owner)
        self.position = list(position)
        self.direction = direction
        self.speed = speed
        self.color = color
        self.size = 5  # 子弹的大小

    def move(self):
        # 根据方向移动子弹
        if self.direction == 'UP':
            self.position[1] -= self.speed
        elif self.direction == 'DOWN':
            self.position[1] += self.speed
        elif self.direction == 'LEFT':
            self.position[0] -= self.speed
        elif self.direction == 'RIGHT':
            self.position[0] += self.speed

    def draw(self, surface):
        # 绘制子弹为一个小矩形
        pygame.draw.rect(surface, self.color, pygame.Rect(self.position[0], self.position[1], self.size, self.size))

    def is_off_screen(self, screen_width, screen_height):
        # 检查子弹是否移出屏幕
        x, y = self.position
        return x < 0 or x > screen_width or y < 0 or y > screen_height

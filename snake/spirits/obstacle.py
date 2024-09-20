import pygame
from snake.spirits.consts import BLUE


class Obstacle:
    def __init__(self, position, color=BLUE):
        self.position = position
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, pygame.Rect(self.position[0], self.position[1], 10, 10))

import random

import pygame
from snake.spirits.consts import RED


class Food:
    def __init__(self, screen_width, screen_height, color=RED):
        self.position = self.random_position(screen_width, screen_height)
        self.color = color

    def random_position(self, screen_width, screen_height):
        x = random.randint(0, (screen_width - 10) // 10) * 10
        y = random.randint(0, (screen_height - 10) // 10) * 10
        return (x, y)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, pygame.Rect(self.position[0], self.position[1], 10, 10))

    def respawn(self, screen_width, screen_height, occupied_positions):
        while True:
            self.position = self.random_position(screen_width, screen_height)
            if self.position not in occupied_positions:
                break

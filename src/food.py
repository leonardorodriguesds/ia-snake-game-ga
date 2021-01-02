import pygame
from colors import *

class Food:
    def __init__(self, x, y, width=10,height=10, game=None, game_x = 0):
        self.x, self.y = x, y
        self.width, self.height=width, height
        self.time = 600
        self.game = game
        self.game_x = game_x

    def draw(self):
        self.time -= 1
        pygame.draw.rect(self.game, red, [self.game_x + self.x, self.y, self.width, self.height])
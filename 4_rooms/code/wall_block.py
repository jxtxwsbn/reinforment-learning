import pygame
from pygame.sprite import Sprite
class Wall_block(Sprite):
    def __init__(self,screen,x,y):
        super(Wall_block,self).__init__()
        self.screen = screen
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x,self.y,50,50)
        self.color = (0,0,0)
    def draw(self):
        pygame.draw.rect(self.screen,self.color,self.rect)

def drawGrid(screen):
    blockSize = 50 #Set the size of the grid block
    for x in range(11):
        for y in range(11):
            rect = pygame.Rect(x*blockSize, y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(screen, (0,0,0), rect, 1)

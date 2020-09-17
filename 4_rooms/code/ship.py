import pygame

class Ship():
    def __init__(self, screen):
        """Initialize the ship and set its starting position."""
        self.screen = screen
        # Load the ship image and get its rect.
        self.image = pygame.image.load('images/bee.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        # Start each new ship at the bottom left of the screen.
        self.rect.left = self.screen_rect.left
        self.rect.bottom = self.screen_rect.bottom

    def blitme(self):
        """Draw the ship at its current location."""
        self.screen.blit(self.image, self.rect)

class Key():
    def __init__(self,screen,x=10,y=10):
        self.screen = screen
        self.key = pygame.image.load('images/flower.jpg')
        self.rect = self.key.get_rect()
        self.screen_rect = screen.get_rect()
        # Start each new key at the top right of the screen.
        self.rect.centerx = x*50+25
        self.rect.centery = (10-y)*50+25

    def blitme(self):
        self.screen.blit(self.key, self.rect)

import sys
import pygame
from ship import Ship,Key
from wall_block import Wall_block,drawGrid
from pygame.sprite import Group
from game_function import step2


def run_game():
    #Initialize game and create a screen object.
    pygame.init()
    #550 pixel wide and 550 pixel high
    screen = pygame.display.set_mode((550, 550))
    pygame.display.set_caption("11X11 grid map")
    #make a ship
    ship = Ship(screen)
    print('initial position of the ship: ','(0,0)')
    #make the goal
    key = Key(screen)
    # Set the background color.
    bg_color = (245, 245, 245)
    #make a brick
    bricks = Group()
    brick_coor = [(0,5),(2,5),(3,5),(4,5),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(5,7),
                  (5,8),(5,10),(6,6),(7,6),(9,6),(10,6)]
    for (x,y) in brick_coor:
        brick = Wall_block(screen,50*x,50*y)
        bricks.add(brick)
    #Start the main loop for the game.

    while True:
        #Watch for keyboard and mouse events.
        step2(ship,key,bricks)
        #Redraw screen during each pass through the loop
        screen.fill(bg_color)
        drawGrid(screen)
        ship.blitme()
        key.blitme()
        for brick in bricks.sprites():
            brick.draw()
        #Make the most recently drawn screen visible.
        pygame.display.flip()

run_game()
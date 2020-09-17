import pygame
import sys
import numpy as np
import random

def step2(ship,key,bricks):
    #the results for the action
    change = np.array([0,0])
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            rand1 = random.uniform(0,1)
            #print(rand1)
            if event.key == pygame.K_RIGHT:
                step_number = 0
                if ship.rect.right <= ship.screen_rect.right:
                    # move the ship to the right
                    #ship.rect.centerx += 50
                    if rand1<0.8:
                        change= np.array([50,0])
                    elif 0.8<rand1<0.9:
                        change = np.array([0,50])
                    else:
                        change = np.array([0,-50])
            elif event.key == pygame.K_LEFT:
                if ship.rect.left >= ship.screen_rect.left:
                    # move the ship to the left
                    #ship.rect.centerx -= 50
                    if rand1<0.8:
                        change= np.array([-50,0])
                    elif 0.8<rand1<0.9:
                        change = np.array([0,50])
                    else:
                        change = np.array([0,-50])
            elif event.key == pygame.K_UP:
                if ship.rect.top >= ship.screen_rect.top:
                    #ship.rect.centery -= 50
                    if rand1<0.8:
                        change= np.array([0,-50])
                    elif 0.8<rand1<0.9:
                        change = np.array([50,0])
                    else:
                        change = np.array([-50,0])
            elif event.key == pygame.K_DOWN:
                if ship.rect.bottom <= ship.screen_rect.bottom:
                    #ship.rect.centery += 50
                    if rand1 < 0.8:
                        change = np.array([0, 50])
                    elif 0.8 < rand1 < 0.9:
                        change = np.array([50, 0])
                    else:
                        change = np.array([-50, 0])

            if np.array_equal(change,np.array([50,0])):
                action = 'Right'
            elif np.array_equal(change,np.array([-50,0])):
                action= 'Left'
            elif np.array_equal(change,np.array([0,50])):
                action= 'Down'
            elif np.array_equal(change,np.array([0,-50])):
                action= 'Up'
            print('take action: {}'.format(action))
            position = ship.rect.topleft
            ship.rect.topleft = tuple(np.asarray(position) + change)
            if pygame.sprite.spritecollideany(ship, bricks) or ship.rect.right > ship.screen_rect.right\
                    or ship.rect.left < ship.screen_rect.left or ship.rect.top < ship.screen_rect.top or \
                    ship.rect.bottom > ship.screen_rect.bottom:

                ship.rect.topleft = position
                print('hit the wall')
            (x,y) = ship.rect.topleft
            print('Current position: ({},{})'.format(int(x/50),10-int(y/50)))
            #print('current position: ','(',int(x/50),10-int(y/50),')')
            if pygame.sprite.collide_rect(ship, key):
                ship.rect.topleft = (0, 500)
                print('+1,get the reward','\n','Initialize the position')

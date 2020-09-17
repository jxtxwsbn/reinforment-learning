import numpy as np
import pygame
import random
import sys
from wall_block import drawGrid

def update_screen(screen,bg_color,ship,key,bricks):
    screen.fill(bg_color)
    drawGrid(screen)
    ship.blitme()
    key.blitme()
    for brick in bricks.sprites():
        brick.draw()
    # Make the most recently drawn screen visible.
    pygame.display.flip()
    # time.sleep(0.0001)
    # print('step: ',step)
    # time.sleep(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()


def movement(action):
    #the dymanic of the environment
    rand1 = random.uniform(0,1)
    if action=='stay':
        change =np.array([0,0])

    if action == 'r':
        if rand1 < 0.8:
            change = np.array([50, 0])
        elif 0.8 < rand1 < 0.9:
            change = np.array([0, 50])
        else:
            change = np.array([0, -50])

    if action == 'l':
        if rand1 < 0.8:
            change = np.array([-50, 0])
        elif 0.8 < rand1 < 0.9:
            change = np.array([0, 50])
        else:
            change = np.array([0, -50])

    if action == 'up':
        if rand1 < 0.8:
            change = np.array([0, -50])
        elif 0.8 < rand1 < 0.9:
            change = np.array([50, 0])
        else:
            change = np.array([-50, 0])

    if action == 'd':
        if rand1 < 0.8:
            change = np.array([0, 50])
        elif 0.8 < rand1 < 0.9:
            change = np.array([50, 0])
        else:
            change = np.array([-50, 0])

    return change

def env_feedback(ship,key,bricks,change):
    #return the reward and current position
    position = ship.rect.topleft
    ship.rect.topleft = tuple(np.asarray(position) + change)
    if pygame.sprite.spritecollideany(ship, bricks) or ship.rect.right > ship.screen_rect.right \
            or ship.rect.left < ship.screen_rect.left or ship.rect.top < ship.screen_rect.top or \
            ship.rect.bottom > ship.screen_rect.bottom:
        ship.rect.topleft = position
        #print('hit the wall')
    (x, y) = ship.rect.topleft
    current_position = (int(x / 50), 10 - int(y / 50))
    #print('Current position: ({},{})'.format(int(x / 50), 10 - int(y / 50)))
    # print('current position: ','(',int(x/50),10-int(y/50),')')
    if pygame.sprite.collide_rect(ship, key):
        (t_x, t_y) = ship.rect.topleft
        target = (int(t_x / 50), 10 - int(t_y / 50))
        ship.rect.topleft = (0, 500)
        #print('+1,get the reward in postion: {}'.format(target), '\n', 'Initialize the position to (0,0)')
        reward =1
    else:
        reward =0
        target=None
    return reward,current_position,target



#===========================================================#
#                          4 planers                        #
#===========================================================#

def random_planer():
    rand = random.uniform(0, 1)
    if rand < 0.25:
        action = 'r'
    elif 0.25 < rand < 0.5:
        action = 'l'
    elif 0.5 < rand < 0.75:
        action = 'up'
    else:
        action = 'd'
    return action

def worse_planer():

    return 'up'

def better_planer():
    rand = random.uniform(0,1)
    if rand < 0.3:
        action = 'r'
    elif 0.3 < rand < 0.5:
        action = 'l'
    elif 0.5 < rand < 0.8:
        action = 'up'
    else:
        action = 'd'
    return action

def learning_planer(total_return,target):
    if total_return < 1:
        rand = random.uniform(0, 1)
        if rand < 0.25:
            action = 'r'
        elif 0.25 < rand < 0.5:
            action = 'l'
        elif 0.5 < rand < 0.75:
            action = 'up'
        else:
            action = 'd'
    else:
        rand2 = random.uniform(0, 1)
        right_m = target[0] - 0
        up_m = target[1] - 0
        right_p = right_m / (right_m + up_m)
        #print(right_p)
        if rand2 < 0.7 * right_p:
            action = 'r'
        elif 0.7 * right_p < rand2 < 0.7:
            action = 'up'
        elif 0.7 < rand2 < 0.85:
            action = 'l'
        elif 0.85 < rand2 < 1:
            action = 'd'

    return action

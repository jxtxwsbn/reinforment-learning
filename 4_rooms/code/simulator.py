import sys
import pygame
from ship import Ship,Key
from wall_block import Wall_block
from pygame.sprite import Group
import matplotlib.pyplot as plt
from environment_model import movement,random_planer,worse_planer,better_planer,learning_planer,env_feedback,update_screen
import random
import argparse

def run_game(args):
    #Initialize game and create a screen object.
    pygame.init()
    #550 pixel wide and 550 pixel high
    screen = pygame.display.set_mode((550, 550))
    pygame.display.set_caption("11X11 grid map")
    # make a blocks
    bricks = Group()
    brick_coor = [(0,5),(2,5),(3,5),(4,5),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(5,7),
                  (5,8),(5,10),(6,6),(7,6),(9,6),(10,6)]
    for (x,y) in brick_coor:
        brick = Wall_block(screen,50*x,50*y)
        bricks.add(brick)
    #make a ship
    ship = Ship(screen)
    print('initial position of the ship: ','(0,0)')
    #make the goal
    x=random.randint(0,10)
    y=random.randint(0,10)
    key = Key(screen,x=10,y=10)
    while pygame.sprite.spritecollideany(key, bricks):
        x = random.randint(0, 10)
        y = random.randint(0, 10)
    #print(x,y)
    # Set the background color.
    bg_color = (230, 230, 230)
    #Start the main loop for the game.
    step =0
    total_return=0
    record_list = []
    target = None
    while True:
        #planar that map state to actions
        #random planer
        if args.planer=='random':
            action = random_planer()
        if args.planer=='worse':
            action = worse_planer()
        #better planer
        if args.planer =='better':
            action = better_planer()
        if args.planer=='learning':
            action = learning_planer(total_return,target)
            #print('please use the arrow in the keyboard to move the ship!')

        #the dynamic property of the environment
        change = movement(action)

        #inteact with the enironment
        reward,current_position,target_postion=env_feedback(ship,key,bricks,change)

        if target_postion!=None:
            target=target_postion

        total_return = reward + total_return
        step = step + 1
        record_list.append(total_return)

        #UI update
        update_screen(screen, bg_color, ship, key, bricks)
        #Redraw screen during each pass through the loop
        if step > (10000-1):
            print(total_return)
            #record = np.array(record_list)
            plt.plot(record_list,color="r", linestyle="-", linewidth=1)
            plt.show()
            break

    sys.exit()
if __name__=='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--planer',type=str,default='random')
    args = parser.parse_args()
    for i in range(10):
        run_game(args)

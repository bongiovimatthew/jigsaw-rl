from pygame.locals import *
from random import randint
import pygame
import time
from Environment.env import Environment, ActionSpace
from enum import Enum
import os
import pygame.surfarray as surfarray
from PIL import Image
import numpy as np

class Apple:
    x = 0
    y = 0
    step = 44
 
    def __init__(self, x, y):
        self.x = x * self.step
        self.y = y * self.step
 
    def draw(self, surface, image):
        surface.blit(image,(self.x, self.y)) 

    def place(self, player):
        counter = 0
        touchesSnake = True 
        while counter < 1000000 and touchesSnake:
            self.x = randint(2, 9) * 44
            self.y = randint(2, 9) * 44
            touchesSnake = False

            for i in range(0, player.length):
                if self.x == player.x[i] and self.y == player.y[i]:
                    touchesSnake = True
                    break

            counter += 1
        return self
 
class Player:
    
    step = 44
    direction = 0
    length = 3
 
    #updateCountMax = 2
    updateCountMax = 0
    updateCount = 0
 
    def __init__(self, length):
       self.x = [0]
       self.y = [0]

       self.length = length
       for i in range(0,2000):
           self.x.append(-100)
           self.y.append(-100)
 
       # initial positions, no collision.
       self.x[1] = 1*44
       self.x[2] = 2*44
 
    def update(self):
 
        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:
 
            # update previous positions
            for i in range(self.length-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]
 
            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step
 
            self.updateCount = 0
 
    def moveRight(self):
        if self.direction == 1: 
            return
        self.direction = 0
 
    def moveLeft(self):
        if self.direction == 0: 
            return
        self.direction = 1
 
    def moveUp(self):
        if self.direction == 3: 
            return
        self.direction = 2
 
    def moveDown(self):
        if self.direction == 2: 
            return
        self.direction = 3 
 
    def draw(self, surface, image, headImage):
        for i in range(1, self.length):
            surface.blit(image, (self.x[i],self.y[i])) 
        surface.blit(headImage, (self.x[0],self.y[0]))
 
class Game:

    def isOutOfBounds(self, x1, y1, bsize):
        if x1 >= 550:
            return True 

        if y1 >= 750: 
            return True 

        if x1 < 0 or y1 < 0: 
            return True 
        return False

    def isCollision(self, x1, y1, x2, y2, bsize):
        # if x1 >= x2 and x1 <= x2 + bsize:
        #     if y1 >= y2 and y1 <= y2 + bsize:
        #         return True
        if x1 == x2 and y1 == y2: 
            return True
        return False

class Actions(Enum):
    ACTION_NOOP = 0 
    ACTION_UP = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3
    ACTION_LEFT = 4
    
class SnakeEnvironment(Environment):

    MAX_ACTIONS_NUM = 5

    def __init__(self):
        self.windowWidth = 800
        self.windowHeight = 600
        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))
        self._display_surf = None

    def setupEnvironment(self):
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        
        basePath = os.path.dirname(__file__)
        self._image_surf = pygame.image.load(os.path.join(basePath, "block.jpg")).convert()
        self._apple_surf = pygame.image.load(os.path.join(basePath, "apple.jpg")).convert()
        self._head_surf = pygame.image.load(os.path.join(basePath, "head.jpg")).convert()

        self._running = True
        self.game = Game()
        self.player = Player(3) 
        self.apple = Apple(5, 5)

    def reset(self):
        pygame.quit()
        
        self._running = False
        self.game = None
        self.player = None
        self.apple = None 
        self._display_surf = None

        pygame.init()
        self.setupEnvironment()
        return self.render()

    def action(self):
        return self.action_space

    def _convert_state(self, action):
        
        if (action == Actions.ACTION_RIGHT.value):
            self.player.moveRight()
        if (action == Actions.ACTION_LEFT.value):
            self.player.moveLeft()
        if (action == Actions.ACTION_UP.value):
            self.player.moveUp()
        if (action == Actions.ACTION_DOWN.value):
            self.player.moveDown()

        self.player.update()

        return self.render()

    def step(self, action):
        next_state = self._convert_state(action)

        #imgDisp = Image.fromarray(next_state, 'RGB')
        #imgDisp.show()
        reward, done, info = self.check_state()

        if done: 
            print("------------------------------------------------------")
            print("Game Over: Score: {0}".format(self.player.length - 3))
            print("------------------------------------------------------")
            print()

        return (next_state, reward, done, info)

    def check_state(self):
        reward = 1
        info = {}
        done = False

         # does snake eat apple?
        if self.game.isCollision(self.apple.x, self.apple.y, self.player.x[0], self.player.y[0], 44):
            self.apple = self.apple.place(self.player)
            self.player.length = self.player.length + 1
            reward = 5

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.isCollision(self.player.x[0], self.player.y[0], self.player.x[i], self.player.y[i], 40):
                reward = -5
                done = True

        if self.game.isOutOfBounds(self.player.x[0], self.player.y[0], 44):
            reward = -5 
            done = True

        return (reward, done, info)

    def render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf, self._head_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        return surfarray.array3d(self._display_surf)
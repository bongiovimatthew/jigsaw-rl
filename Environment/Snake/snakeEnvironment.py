from random import randint
from Environment.env import Environment, ActionSpace
from enum import Enum
import os
from PIL import Image
import numpy as np


class Apple:

    def __init__(self, x, y):
        self.step = 50

        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.paste(image, (self.x, self.y))

    def place(self, player):
        counter = 0
        touchesSnake = True
        while counter < 1000000 and touchesSnake:
            self.x = randint(0, (SnakeEnvironment.WINDOW_WIDTH / self.step) - 1) * self.step
            self.y = randint(0, (SnakeEnvironment.WINDOW_HEIGHT / self.step) - 1) * self.step
            touchesSnake = False

            for i in range(0, player.length):
                if self.x == player.x[i] and self.y == player.y[i]:
                    touchesSnake = True
                    break

            counter += 1
        return self


class Player:

    direction = 0
    length = 3

    #updateCountMax = 2
    updateCountMax = 0
    updateCount = 0

    def __init__(self, length):
        self.x = [0]
        self.y = [0]

        self.length = length
        self.step = 50 

        for i in range(0, 2000):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        self.x[0] = 2 * self.step
        self.x[1] = 1 * self.step
        self.x[2] = 0

        self.y[0] = 0
        self.y[1] = 0
        self.y[2] = 0

    def update(self):

        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:

            # update previous positions
            for i in range(self.length-1, 0, -1):
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
            surface.paste(image, (self.x[i], self.y[i]))
        surface.paste(headImage, (self.x[0], self.y[0]))

    def draw(self, body_surface, head_surface, image, headImage):
        for i in range(1, self.length):
            body_surface.paste(image, (self.x[i], self.y[i]))
        head_surface.paste(headImage, (self.x[0], self.y[0]))


class Game:

    def isOutOfBounds(self, x1, y1):
        if x1 >= SnakeEnvironment.WINDOW_WIDTH:
            return True

        if y1 >= SnakeEnvironment.WINDOW_HEIGHT:
            return True

        if x1 < 0 or y1 < 0:
            return True
        return False

    def isCollision(self, x1, y1, x2, y2,):
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
    WINDOW_HEIGHT = 400
    WINDOW_WIDTH = 400

    def __init__(self):
        self.action_space = ActionSpace(range(self.MAX_ACTIONS_NUM))
        self._display_surf = None
        self.setupEnvironment()

    def setupEnvironment(self):
        self._display_surf = Image.new('RGB', (SnakeEnvironment.WINDOW_WIDTH, SnakeEnvironment.WINDOW_HEIGHT), (0, 0, 0))

        basePath = os.path.dirname(__file__)
        apple_path = os.path.join(basePath, "apple.jpg")
        block_path = os.path.join(basePath, "block.jpg")
        head_path = os.path.join(basePath, "head.jpg")

        self._image_surf = Image.open(block_path, 'r')
        self._apple_surf = Image.open(apple_path, 'r')
        self._head_surf = Image.open(head_path, 'r')

        self._running = True
        self.game = Game()
        self.player = Player(3)
        self.apple = Apple(0, 3)
        self.movesCount = 0
        self.numberOfTimesExecutedEachAction = list(range(self.MAX_ACTIONS_NUM))

    def reset(self):
        self._running = False
        self.game = None
        self.player = None
        self.apple = None
        self._display_surf = None
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
        self.movesCount += 1
        #imgDisp = Image.fromarray(next_state, 'RGB')
        # imgDisp.show()
        self.numberOfTimesExecutedEachAction[action] += 1

        reward, done, info = self.check_state()

        if done:
            print("------------------------------------------------------")
            print("Game Over: Score: {0}".format(self.player.length - 3))
            print("Num Moves: {0}".format(self.movesCount))
            print("------------------------------------------------------")
            print()

        return (next_state, reward, done, info)

    def check_state(self):
        reward = 0.25
        info = {}
        done = False
        info['oldScore'] = self.player.length
        info['movesCount'] = self.movesCount
        info["negativeRewardCount"] = -1 
        info["zeroRewardCount"] = -1
        info["positiveRewardCount"] = -1
        info['numberOfTimesExecutedEachAction'] = self.numberOfTimesExecutedEachAction

        # does snake eat apple?
        if self.game.isCollision(self.apple.x, self.apple.y, self.player.x[0], self.player.y[0]):
            self.apple = self.apple.place(self.player)
            self.player.length = self.player.length + 1
            reward = 1  #+ 2 * self.player.length

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.isCollision(self.player.x[0], self.player.y[0], self.player.x[i], self.player.y[i]):
                reward = 0
                done = True

        if self.game.isOutOfBounds(self.player.x[0], self.player.y[0]):
            reward = 0
            done = True

        info['score'] = self.player.length

        return (reward, done, info)

    def render(self):
#        self._display_surf.paste( (0, 0, 0), [0, 0, self._display_surf.size[0], self._display_surf.size[1]])

        body_surface = Image.new('L', (SnakeEnvironment.WINDOW_WIDTH, SnakeEnvironment.WINDOW_HEIGHT))
        head_surface = Image.new('L', (SnakeEnvironment.WINDOW_WIDTH, SnakeEnvironment.WINDOW_HEIGHT))
        apple_surface = Image.new('L', (SnakeEnvironment.WINDOW_WIDTH, SnakeEnvironment.WINDOW_HEIGHT))

        self.player.draw(body_surface, head_surface, self._image_surf, self._head_surf)
        self.apple.draw(apple_surface, self._apple_surf)

        body_array = np.array(body_surface)
        head_array = np.array(head_surface)
        apple_array = np.array(apple_surface)

        full_image = np.dstack((body_array, head_array, apple_array))
        return full_image
        #return np.array(self._display_surf)
        #return np.rot90(np.flipud(surfarray.array3d(self._display_surf)), axes=(1,0))

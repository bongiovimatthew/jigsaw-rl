from pygame.locals import *
from random import randint
import pygame
import time
import os


class Apple:
    x = 0
    y = 0
    step = 44

    def __init__(self, x, y):
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))

    def place(self, player):
        counter = 0
        touchesSnake = True
        while counter < 1000000 and touchesSnake:
            self.x = randint(1, 17) * 44
            self.y = randint(1, 12) * 44
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

    updateCountMax = 2
    updateCount = 0

    def __init__(self, length):
        self.x = [0]
        self.y = [0]
        self.length = length
        for i in range(0, 2000):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        self.x[1] = 1*44
        self.x[2] = 2*44

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
            surface.blit(image, (self.x[i], self.y[i]))
        surface.blit(headImage, (self.x[0], self.y[0]))


class Game:

    def isOutOfBounds(self, x1, y1, bsize):
        if x1 >= 750:
            return True

        if y1 >= 550:
            return True

        if x1 < 0 or y1 < 0:
            return True
        return False

    def isCollision(self, x1, y1, x2, y2, bsize):
        if x1 == x2 and y1 == y2:
            return True
        return False


class App:

    windowWidth = 800
    windowHeight = 600
    player = 0
    apple = 0

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self._head_surf = None
        self.game = Game()
        self.player = Player(3)
        self.apple = Apple(5, 5)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('Snake')
        self._running = True
        basePath = os.path.dirname(__file__)
        self._image_surf = pygame.image.load(os.path.join(basePath, "block.jpg")).convert()
        self._apple_surf = pygame.image.load(os.path.join(basePath, "apple.jpg")).convert()
        self._head_surf = pygame.image.load(os.path.join(basePath, "head.jpg")).convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        self.player.update()
        reward = 0

        # does snake eat apple?
        if self.game.isCollision(self.apple.x, self.apple.y, self.player.x[0], self.player.y[0], 44):
            self.apple = self.apple.place(self.player)
            self.player.length = self.player.length + 1
            reward = 1
            #print("Reward +1")

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.isCollision(self.player.x[0], self.player.y[0], self.player.x[i], self.player.y[i], 40):
                reward = -1
                done = True
                #print("Reward -1")
                self._running = False

        if self.game.isOutOfBounds(self.player.x[0], self.player.y[0], 44):
            reward = -1
            done = True
            #print("Reward -1")
            self._running = False

        # if reward == 0:
            #print("Reward 0")
        pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf, self._head_surf)
        self.apple.draw(self._display_surf, self._apple_surf)

        # Use this for generating the env state image
        #pygame.image.save(self._display_surf, "screenshot.jpg")

        pygame.display.flip()

    def on_cleanup(self):
        print("------------------------------------------------------")
        print("Game Over: Score: {0}".format(self.player.length - 3))
        print("------------------------------------------------------")
        print()

        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while(self._running):
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if (keys[K_RIGHT]):
                self.player.moveRight()

            if (keys[K_LEFT]):
                self.player.moveLeft()

            if (keys[K_UP]):
                self.player.moveUp()

            if (keys[K_DOWN]):
                self.player.moveDown()

            if (keys[K_ESCAPE]):
                self._running = False

            self.on_loop()
            self.on_render()

            # Remove this unless the human is playing
            time.sleep(50.0 / 1000.0)

        self.on_cleanup()


if __name__ == "__main__":
    replay = 5

    while replay > 0:
        theApp = App()
        theApp.on_execute()
        replay -= 1

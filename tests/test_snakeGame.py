from Environment.Snake.snakeEnvironment import SnakeEnvironment
from PIL import Image
from enum import Enum

class Actions(Enum):
    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3
    ACTION_LEFT = 4

class SnakeGameTests:

    def TestGenerateAndDisplayPuzzle():
        env = SnakeEnvironment()
        imageData = env.render()
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        for i in range(0, 2):
            print("NOOP")
            imageData, _, _, _ = env.step(Actions.ACTION_NOOP.value) 
            imgDisp = Image.fromarray(imageData, 'RGB')
            imgDisp.show()

        print("DOWN")
        imageData, _, _, _ = env.step(Actions.ACTION_DOWN.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("NOOP")
        imageData, _, _, _ = env.step(Actions.ACTION_NOOP.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("RIGHT")
        imageData, _, _, _ = env.step(Actions.ACTION_RIGHT.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("NOOP")
        imageData, _, _, _ = env.step(Actions.ACTION_NOOP.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("UP")
        imageData, _, _, _ = env.step(Actions.ACTION_UP.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("NOOP")
        imageData, _, _, _ = env.step(Actions.ACTION_NOOP.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("LEFT")
        imageData, _, _, _ = env.step(Actions.ACTION_LEFT.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()

        print("NOOP")
        imageData, _, _, _ = env.step(Actions.ACTION_NOOP.value) 
        imgDisp = Image.fromarray(imageData, 'RGB')
        imgDisp.show()
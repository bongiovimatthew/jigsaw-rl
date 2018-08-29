from pynput.keyboard import Key, Listener
from enum import Enum
import PIL as pl
from PIL import ImageTk, Image

from Environment.snakeEnvironment import SnakeEnvironment


class Actions(Enum):
    ACTION_NOOP = 0 
    ACTION_UP = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3
    ACTION_LEFT = 4

class HumanSnakeAgent: 
    def __init__(self):
        self.env = SnakeEnvironment()
        self.s_t = self.env.reset()
        self.R = 0
        self.negativeRewardCount = 0
        self.zeroRewardCount = 0
        self.positiveRewardCount = 0
        self.averageRewards = 0
        self.averageScore = 0
        self.slidingWindowAverageScore = 0
        self.numStepsForRunningMetrics = 0
        self.slidingWindowScoresArray = []

        startImg = self.env.render()
        self.displayUpdatedBoard(startImg)

    def displayUpdatedBoard(self, imgData):
        img = pl.Image.fromarray(imgData, 'RGB')
        img.show(title="Move")

    def on_press(self, key):
        #print('{0} pressed'.format(key))
        return 

    def on_release(self, key):
        if key == Key.esc:
            # Stop listener
            return False

        action = self.getActionFromUserInput(key)

        print("Action:  {0}", action)
        self.s_t, reward, done, info = self.env.step(action)
        #info = self.update_and_get_metrics(info)

        imgData = self.env.render()

        if (done):
            imgData = self.env.reset()

        self.displayUpdatedBoard(imgData)
        
    def getActionFromUserInput(self, input):
        if input == Key.up:
            return Actions.ACTION_UP.value
        if input == Key.down:
            return Actions.ACTION_DOWN.value
        if input == Key.left:
            return Actions.ACTION_LEFT.value
        if input == Key.right:
            return Actions.ACTION_RIGHT.value
        if input == Key.space:
            return Actions.ACTION_NOOP.value
        return
    
    def run(self):

        print("Input Move (up/down/left/right/spacebar/Esc): ")
        self.env.start_run_loop()
        
        # with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
        #     listener.join() 

    def update_and_get_metrics(self, info):
        rewards = info["rewards"]
        score = info["score"]
        self.numStepsForRunningMetrics += 1

        if rewards < 0:
            self.negativeRewardCount += 1

        elif rewards == 0:
            self.zeroRewardCount += 1

        elif rewards > 0:
            self.positiveRewardCount += 1

        self.averageRewards = ((self.averageRewards * (self.t - 1)) + rewards) / self.t
        self.averageScore = ((self.averageScore * (self.t - 1)) + score) / self.t

        self.slidingWindowAverageScore = 0
        self.slidingWindowScoresArray.append(score)

        if len(self.slidingWindowScoresArray) > 50:
            self.slidingWindowScoresArray.pop(0)

        self.slidingWindowAverageScore = sum(self.slidingWindowScoresArray) / len(self.slidingWindowScoresArray)

        info["negativeRewardCount"] = self.negativeRewardCount
        info["zeroRewardCount"] = self.zeroRewardCount
        info["positiveRewardCount"] = self.positiveRewardCount
        info["averageRewards"] = self.averageRewards
        info["averageScore"] = self.averageScore
        info["slidingWindowAverageScore"] = self.slidingWindowAverageScore

        return info

    def reset_running_metrics(self):
        self.negativeRewardCount = 0
        self.positiveRewardCount = 0
        self.zeroRewardCount = 0
        self.averageRewards = 0
        self.numStepsForRunningMetrics = 0

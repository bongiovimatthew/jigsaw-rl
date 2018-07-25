from pynput.keyboard import Key, Listener
from Environment.env import PuzzleEnvironment
from enum import Enum

import PIL as pl
from PIL import ImageTk, Image

from threading import Thread
from time import sleep

from tkinter import *

class Actions(Enum):
    ACTION_CYCLE = 0 
    ACTION_TRANS_UP = 1
    ACTION_TRANS_RIGHT = 2
    ACTION_TRANS_DOWN = 3
    ACTION_TRANS_LEFT = 4
    ACTION_ROT90_1 = 5
    ACTION_ROT90_2 = 6
    ACTION_ROT90_3 = 7

class HumanAgent: 
    def __init__(self):
        self.env = PuzzleEnvironment()
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
        # Create a window
        #self.window = tkinter.Tk()

        # Create a canvas that can fit the above image
        #width = 228
        #height = 228 
        #self.canvas = tkinter.Canvas(self.window, width = width, height = height)
        #self.canvas.pack()

        img = pl.Image.fromarray(imgData, 'RGB')
        img.show(title="Move")
        #sleep(2)
        #img.destroy()

        # root = Tk()
        # root_panel = Frame(root)
        # root_panel.pack(side="bottom", fill="both", expand="yes")

        # img_tk = ImageTk.PhotoImage(img)
        # img_panel = Label(root_panel)
        # img_panel.configure(image=img_tk)

        # root.mainloop()
        # sleep(2)
        # root.destroy()

        #self.canvas.create_image(0, 0, image=img, anchor=tkinter.NW)
        #self.window.mainloop()
        #sleep(2)
        #self.window.root.destroy()

    def on_press(self, key):
        #print('{0} pressed'.format(key))
        return 

    def on_release(self, key):
        if key == Key.esc:
            # Stop listener
            return False

        action = self.getActionFromUserInput(key)

        self.s_t, reward, done, info = self.env.step(action)
        #info = self.update_and_get_metrics(info)

        imgData = self.env.render()

        #thread = Thread(target = self.displayUpdatedBoard(imgData), args = (10, ))
        #thread.start()
        #thread.join()

        self.displayUpdatedBoard(imgData)
        
    def getActionFromUserInput(self, input):
        if input == Key.up:
            return Actions.ACTION_TRANS_UP.value
        if input == Key.down:
            return Actions.ACTION_TRANS_DOWN.value
        if input == Key.left:
            return Actions.ACTION_TRANS_LEFT.value
        if input == Key.right:
            return Actions.ACTION_TRANS_RIGHT.value
        if input == Key.space:
            return Actions.ACTION_CYCLE.value
        if str(input) == "'r'": 
            return Actions.ACTION_ROT90_1.value
        return
    
    def run(self):

        print("Input Move (up/down/left/right/spacebar/Esc): ")
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join() 

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
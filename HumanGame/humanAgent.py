from pynput.keyboard import Listener
from enum import Enum
import PIL as pl
from PIL import ImageTk, Image

from Environment.env import PuzzleEnvironment

from threading import Thread
from time import sleep

from tkinter import *
import cv2 
import pdb
class Actions(Enum):
    ACTION_TRANS_UP = 0
    ACTION_TRANS_RIGHT = 1
    ACTION_TRANS_DOWN = 2
    ACTION_TRANS_LEFT = 3
    ACTION_ROT90_1 = 4
    ACTION_CYCLE = 5

class Key():
    RIGHT = 2555904
    LEFT = 2424832
    UP = 2490368
    DOWN = 2621440
    SPACE = 32
    NEXT = 110 
class HumanAgent: 
    def __init__(self):
        self.env = PuzzleEnvironment(2)
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

        self.startImg = self.env.render()
        cv2.namedWindow('puzzle',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('puzzle', 600,600)
        # self.displayUpdatedBoard(self.startImg)

    def displayUpdatedBoard(self, imgData):

        cv2.imshow('puzzle',imgData)
        cv2.waitKey(500) 
    def on_press(self, key):
        #print('{0} pressed'.format(key))
        return 

    def play(self):
        imgData = self.startImg
        while True:
            cv2.imshow('puzzle',imgData)
            key = cv2.waitKey(500)
            if key == 27: # exit on ESC
                return False
            action = self.getActionFromUserInput(key)
            
            if action != -1 :
                self.s_t, reward, done, info = self.env.step(action)
                imgData = self.env.render()
                print("action: %s, reward:%f"%(Actions(info["action"]), reward))

    def log(self, reward,info):
        file_name = "gameOutput.txt"

        with open(file_name, "a+") as f:
            f.write("------------------------------------\r\n")
            f.write("Step: {0}\r\n".format(info["step"]))
            f.write("Action: {0}\r\n".format(Actions(info["action"])))
            f.write("Reward: {0}\r\n".format(reward))
            f.write("Score: {0}\r\n".format(info["score"]))
            f.write("------------------------------------\r\n")
            f.write("\r\n")
            f.write("\r\n")

       
    def getActionFromUserInput(self, input):
        if input == Key.UP:
            return Actions.ACTION_TRANS_UP.value
        if input == Key.DOWN:
            return Actions.ACTION_TRANS_DOWN.value
        if input == Key.LEFT:
            return Actions.ACTION_TRANS_LEFT.value
        if input == Key.RIGHT:
            return Actions.ACTION_TRANS_RIGHT.value
        if input == Key.SPACE:
            return Actions.ACTION_ROT90_1.value
        if input == Key.NEXT:
            return Actions.ACTION_CYCLE.value
        # if str(input) == "'r'": 
        #     return Actions.ACTION_ROT90_1.value
        return -1
    
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

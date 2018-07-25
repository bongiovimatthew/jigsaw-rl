import matplotlib.pyplot as plt
import numpy as np

class MetricsManager: 
    
    def __init__(self):
        self.metric_qarrayHit = 0.0
        self.metric_qarrayMiss = 0.0

        self.metric_totalQarray = 0.0
        self.metric_totalQarrayLengthOnHit = 0.0

        self.metric_totalReward = 0.0
        self.metric_rewardOps = 0.0

        self.metric_totalExplores = 0.0


    def displayMetricsGraph(self): 
        plt.ion()
        fig = plt.figure()
        fig.text(1,1, "TestText")
        fig.canvas.draw()
        plt.show()
        #fig.canvas.flush_events()

        return 

    def displayGraphs(self):
        x = np.linspace(0, 6*np.pi, 100)
        y = np.sin(x)

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

        for phase in np.linspace(0, 10*np.pi, 500):
            line1.set_ydata(np.sin(x + phase))
            fig.canvas.draw()
            fig.canvas.flush_events()

    def displayMetrics(self):
        hitToMissRatio = 0.0
        if self.metric_qarrayMiss != 0: 
            hitToMissRatio = self.metric_qarrayHit / self.metric_qarrayMiss

        print("-----------------------------------------------------------------------")
        print("Num QArray Hits: {0}".format(self.metric_qarrayHit))
        print("Num QArray Miss: {0}".format(self.metric_qarrayMiss))
        print("Hit to Miss Ratio: {0}".format(hitToMissRatio))

        print("Num Explores: {0}".format(self.metric_totalExplores))
        print("Num Total QArray Ops (num exploits): {0}".format(self.metric_totalQarray))

        exploitExploreRatio = 0.0
        if self.metric_totalExplores != 0: 
            exploitExploreRatio = self.metric_totalQarray / self.metric_totalExplores

        print("Exploit / Explore Ratio: {0}".format(exploitExploreRatio))

        avgQArrayLen = 0
        avgReward = 0
        if self.metric_qarrayHit != 0: 
            avgQArrayLen = self.metric_totalQarrayLengthOnHit / self.metric_qarrayHit

        if self.metric_rewardOps != 0: 
            avgReward = self.metric_totalReward / self.metric_rewardOps

        print("Avg Num Qs in QArray on Hits: {0}".format(avgQArrayLen))
        print("Avg Reward: {0}".format(avgReward))
        print("-----------------------------------------------------------------------")
        print()
        print()
        return 
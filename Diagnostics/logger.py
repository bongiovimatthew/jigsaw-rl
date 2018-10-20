import os
import shutil
import json
import time
import numpy as np
from scipy.misc import imsave
from Environment.env import Actions
import celery
class Logger: 

    root = 'files'
    modelsRoot = 'models'
    path_rewards = 'files/rewards/'
    path_losses = 'files/losses/'
    path_meta = 'files/metadata.json'
    path_model_pi = 'models/model_pi.model'
    path_model_v = 'models/model_v.model'
    path_scores = 'files/scores/'
    path_state_images = 'files/state_images/'
    path_metrics = 'files/metrics/'

    def create_folders(lock, atari_name, cores, tmax, game_length, Tmax, C, gamma, lr):
        lock.acquire()
        try:
            if not os.path.exists(Logger.root):
                # Delete if exists.
                # print ('The folder named files will be deleted!')
                # input ('Press Enter to continue.')
                os.makedirs(Logger.root)
            else:
                shutil.rmtree(Logger.root)
                os.makedirs(Logger.root)     

        except:
            time.sleep(1)
            os.makedirs(Logger.root)        
        print("calling create folder")
            
        # Create the new folders.
        os.makedirs(Logger.path_rewards)
        os.makedirs(Logger.path_losses)
        os.makedirs(Logger.path_scores)
        os.makedirs(Logger.path_state_images)
        os.makedirs(Logger.path_metrics)

        metadata = [time.strftime("%d/%m/%y"), atari_name, 'cores '+str(cores), 'tmax '+str(tmax), 'gl '+str(game_length), 'Tmax '+str(Tmax), 'C '+str(C), 'gamma '+str(gamma), 'lr '+str(lr)]
        with open(Logger.path_meta, "w") as f:
            f.write(json.dumps(metadata))
        lock.release()
    def log_state_image(boardData, steps, learner_id,action):
        #pngfile = "testImage.png"
        #pngWriter.write(pngfile, numpy.reshape(boardData, (-1, column_count * plane_count)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_path = Logger.path_state_images + "stateImage_" + str(learner_id) + "_"+ str(steps) + "_" + timestr + "_action_" + str(action) + ".png"

        input_img = np.array(boardData)
        
        # Reshape input to meet with CNTK expectations.
        grayScaleImg = np.reshape(input_img, (84, 84))

        print(grayScaleImg.shape)
        imsave(file_path, grayScaleImg)

    def log_metrics(info, iteration, learner_id):
        Logger.log_scores(iteration, learner_id, info['score'], info['oldScore'], info['averageScore'], info['slidingWindowAverageScore'])

        file_name = Logger.path_metrics + "metrics_" + str(learner_id) + ".txt"
        with open(file_name, "a+") as f:
            f.write("Step {0}: negativeRewardCount: {1}, zeroRewardCount: {2} positiveRewardCount: {3} averageRewards: {4}\r\n".format(iteration, info["negativeRewardCount"], info["zeroRewardCount"], info["positiveRewardCount"], info["averageRewards"]))

        file_name = Logger.path_metrics + "moves_" + str(learner_id) + ".txt"
        with open(file_name, "a+") as f:
            stringToPrint = ""
            for i in range(len(info["numberOfTimesExecutedEachAction"])):
                actionName = Actions(i).name
                stringToPrint += actionName + ": " + str(info["numberOfTimesExecutedEachAction"][i]) + " "

            stringToPrint += "\r\n"
            f.write(stringToPrint)

    def log_scores(iteration, learner_id, currentScore, oldScore, averageScore, slidingWindowAverageScore):
        file_name = Logger.path_scores + "score_" + str(learner_id) + ".txt"
        with open(file_name, "a+") as f:
            f.write("Step {0}: PreviousScore: {1}, CurrentScore: {2} AverageScore: {3} SlidingWindowAverageScore: {4}\r\n".format(iteration, oldScore, currentScore, averageScore, slidingWindowAverageScore))

    def log_rewards(rewards, iteration, learner_id, rnd):
        file_name = Logger.path_rewards + "rwd_" + str(iteration) + "_" + str(learner_id) + "_" + str(rnd) + ".json"
        with open(file_name, "w") as f:
            f.write(json.dumps(rewards))
        
    def log_losses(loss, iteration, learner_id):
        file_name = Logger.path_losses + "loss_" + str(iteration) + "_" + str(learner_id) + ".json"
        with open(file_name, "w") as f:
            f.write(json.dumps(loss))
        
    def read_metadata():
        with open(Logger.path_meta, "r") as f:
            data = json.load(f)
        return data

    def save_model(agent, shared_params):
        agent.save_model(shared_params, Logger.path_model_pi, Logger.path_model_v)
        
    def load_model(net):
        if os.path.exists(Logger.path_model_pi) and os.path.exists(Logger.path_model_v):
            net.load_model(Logger.path_model_pi, Logger.path_model_v)

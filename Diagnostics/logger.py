import os
import shutil
import json
import time
import numpy as np
from scipy.misc import imsave
#from Environment.env import Actions
from pathlib import Path

class Logger:

    root = Path('files')
    modelsRoot = Path('models')
    path_rewards = Path('files/rewards/')
    path_losses = Path('files/losses/')
    path_meta = Path('files/metadata.json')
    path_model_pi = Path('models/model_pi.model')
    path_model_v = Path('models/model_v.model')
    path_scores = Path('files/scores/')
    path_state_images = Path('files/state_images/')
    path_dnn_intermediate_images = Path('files/dnn_intermediate_images/')
    path_metrics = Path('files/metrics/')

    logger_initiated = False

    def init():
        if Logger.logger_initiated:
            return

        Logger.create_folders_internal()
        Logger.logger_initiated = True

        return

    def create_folders(lock, atari_name, cores, tmax, game_length, Tmax, C, gamma, lr):
        lock.acquire()      

        Logger.create_folders_internal()
        metadata = [time.strftime("%d/%m/%y"), atari_name, 'cores '+str(cores), 'tmax '+str(tmax), 'gl '+str(game_length), 'Tmax '+str(Tmax), 'C '+str(C), 'gamma '+str(gamma), 'lr '+str(lr)]
        with open(Logger.path_meta, "w") as f:
            f.write(json.dumps(metadata))
        lock.release()

    def create_folders_internal():

        try:
            if not os.path.exists(Logger.root):
                # Delete if exists.
                # print ('The folder named files will be deleted!')
                # input ('Press Enter to continue.')
                Logger.root.mkdir(exist_ok=True, parents=True)
            else:
                shutil.rmtree(Logger.root)
                Logger.root.mkdir(exist_ok=True, parents=True)

        except:
            time.sleep(1)
            Logger.root.mkdir(exist_ok=True, parents=True)

        # Create the new folders.
        Logger.path_rewards.mkdir(exist_ok=True, parents=True)
        Logger.path_losses.mkdir(exist_ok=True, parents=True)
        Logger.path_scores.mkdir(exist_ok=True, parents=True)
        Logger.path_state_images.mkdir(exist_ok=True, parents=True)
        Logger.path_dnn_intermediate_images.mkdir(exist_ok=True, parents=True)
        Logger.path_metrics.mkdir(exist_ok=True, parents=True)

    def log_state_image(boardData, steps, learner_id, action, stateShape):
        Logger.init()
        #pngfile = "testImage.png"
        #pngWriter.write(pngfile, numpy.reshape(boardData, (-1, column_count * plane_count)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = "stateImage_" + str(learner_id) + "_" + str(steps) + "_" + timestr + "_action_" + str(action) + ".png"
        file_path = Logger.path_state_images / file_name

        input_img = np.array(boardData)

        # Reshape input to meet with CNTK expectations.
        grayScaleImg = np.reshape(input_img, (stateShape[0], stateShape[1]))

        imsave(file_path, grayScaleImg)

    def log_dnn_intermediate_image(imageToSave, imgInfoStr):
        Logger.init()
        #pngfile = "testImage.png"
        #pngWriter.write(pngfile, numpy.reshape(boardData, (-1, column_count * plane_count)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = imgInfoStr + "_" + timestr + ".png"
        file_path = Logger.path_dnn_intermediate_images / file_name 
        imsave(file_path, imageToSave)

    def log_metrics(info, iteration, learner_id):
        Logger.init()
        Logger.log_scores(iteration, learner_id, info['score'], info['oldScore'])
        file_name = "metrics_" + str(learner_id) + ".txt"
        file_path = Logger.path_metrics / file_name
        with open(file_path, "a+") as f:
            f.write("Step {0}: negativeRewardCount: {1}, zeroRewardCount: {2} positiveRewardCount: {3}\r\n".format(
                iteration, info["negativeRewardCount"], info["zeroRewardCount"], info["positiveRewardCount"]))

        file_name = "moves_" + str(learner_id) + ".txt"
        file_path = Logger.path_metrics / file_name
        with open(file_path, "a+") as f:
            stringToPrint = ""
            for i in range(len(info["numberOfTimesExecutedEachAction"])):
                #actionName = Actions(i).name
                stringToPrint += str(info["numberOfTimesExecutedEachAction"][i]) + ", " # actionName + ": " + \

            stringToPrint += "\r\n"
            f.write(stringToPrint)

    def log_scores(iteration, learner_id, currentScore, oldScore):
        Logger.init()
        file_name = "score_" + str(learner_id) + ".txt"
        file_path = Logger.path_scores / file_name
        with open(file_path, "a+") as f:
            f.write("Step {0}: PreviousScore: {1}, CurrentScore: {2}\r\n".format(
                iteration, oldScore, currentScore))

    def log_rewards(rewards, iteration, learner_id, rnd):
        Logger.init()
        file_name = "rwd_" + str(iteration) + "_" + str(learner_id) + "_" + str(rnd) + ".json"

        file_path = Logger.path_rewards / file_name

        with open(file_path, "w") as f:
            f.write(json.dumps(rewards))

    def log_losses(loss, iteration, learner_id):
        Logger.init()
        file_name = "loss_" + str(iteration) + "_" + str(learner_id) + ".json"
        file_path = Logger.path_losses / file_name
        with open(file_path, "w") as f:
            f.write(json.dumps(loss))

    def read_metadata():
        Logger.init()
        with open(Logger.path_meta, "r") as f:
            data = json.load(f)
        return data

    def save_model(agent, shared_params):
        Logger.init()
        agent.save_model(shared_params, Logger.path_model_pi, Logger.path_model_v)

    def load_model(net):
        Logger.init()
        if os.path.exists(Logger.path_model_pi) and os.path.exists(Logger.path_model_v):
            net.load_model(Logger.path_model_pi, Logger.path_model_v)

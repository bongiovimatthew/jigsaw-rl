import os
import shutil
import json
import time
import numpy as np
from scipy.misc import imsave
from Environment.env import Actions
from celery.contrib import rdb
import imageio
import matplotlib.pyplot as plt
import logging
logging_level = getattr(logging, 'INFO')
logging.basicConfig(level=logging_level,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S%p')
logger = logging.getLogger(__name__)
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
    path_graphs = 'files/graphs/'
    def create_folders(lock, atari_name, cores, tmax, game_length, Tmax, C, gamma, lr):
        lock.acquire()
        # try:
        if not os.path.exists(Logger.root):
            try:
                os.makedirs(Logger.root)
                logger.info("calling create folder")
            except:
                time.sleep(1)
                os.makedirs(Logger.root)  
            # Create the new folders.
            os.makedirs(Logger.path_rewards)
            os.makedirs(Logger.path_losses)
            os.makedirs(Logger.path_scores)
            os.makedirs(Logger.path_state_images)
            os.makedirs(Logger.path_metrics)
            os.makedirs(Logger.path_graphs)

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

        logger.info(grayScaleImg.shape)
        imsave(file_path, grayScaleImg)

    def save_state_image(info,state,action,step):
        img = np.reshape(state,(84,84))
        img = (img).astype(np.uint8)
        imageio.imwrite('files/state_images/puzzle_time_%d_action_%s_reward_%f.jpg'%(step,Actions(action).name,info["rewards"]),img) 


    def log_metrics(info, file_name,metric_names):

        if not os.path.isfile(file_name):
            with open(file_name,"w") as f:
                f.write('\t'.join(metric_names)+'\n')
        else:
            with open(file_name,'a+') as f:
                log_str = '\t'.join([str(round(info[metric],4)) if isinstance(info[metric], (float, int)) else str(info[metric]) for metric in metric_names]) + '\n'
                f.write(log_str)
                logger.info(log_str)
 
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

    def plot_history(metric_history, file_name, metric_names = ['loss_on_v','average_rewards']):
        plt.figure()
        for i, name in enumerate(metric_names):
            vals = []
            iterations = []
            for metric in metric_history:
                vals.append(metric[name])
                iterations.append(metric['iteration'])
            plt_id = '{0}1{1}'.format(len(metric_names),i+1)
            plt.subplot(int(plt_id))
            plt.plot(iterations, vals.copy(), label = name); plt.xlabel("itrs")
            plt.grid()
            plt.legend(loc='best',numpoints = 1)
        plt.savefig(file_name)




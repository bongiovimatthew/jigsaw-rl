import os
import shutil
import json
import time
import numpy as np

from scipy.misc import imsave

root = 'files'
path_rewards = 'files/rewards/'
path_losses = 'files/losses/'
path_meta = 'files/metadata.json'
path_model_pi = 'files/model_pi.model'
path_model_v = 'files/model_v.model'
path_scores = 'files/scores/'
path_state_images = 'files/state_images/'

def create_folders(atari_name, cores, tmax, game_length, Tmax, C, gamma, lr):
    if os.path.exists(root):
        # Delete if exists.
        # print ('The folder named files will be deleted!')
        # input ('Press Enter to continue.')
        shutil.rmtree(root)
        
    # Create the new folders.
    os.makedirs(path_rewards)
    os.makedirs(path_losses)
    os.makedirs(path_scores)
    os.makedirs(path_state_images)

    metadata = [time.strftime("%d/%m/%y"), atari_name, 'cores '+str(cores), 'tmax '+str(tmax), 'gl '+str(game_length), 'Tmax '+str(Tmax), 'C '+str(C), 'gamma '+str(gamma), 'lr '+str(lr)]
    with open(path_meta, "w") as f:
        f.write(json.dumps(metadata))

def log_state_image(boardData):
    #pngfile = "testImage.png"
    #pngWriter.write(pngfile, numpy.reshape(boardData, (-1, column_count * plane_count)))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_path = path_state_images + "stateImage_" + timestr + ".png"

    input_img = np.array(boardData)
    
    # Reshape input to meet with CNTK expectations.
    grayScaleImg = np.reshape(input_img, (84, 84))

    print(grayScaleImg.shape)
    imsave(file_path, grayScaleImg)

def log_scores(iteration, currentScore, oldScore):
    file_name = path_scores + "score_" + ".txt"
    with open(file_name, "a+") as f:
        f.write("Step {0}: PreviousScore: {1}, CurrentScore: {2}\r\n".format(iteration, oldScore, currentScore))

def log_rewards(rewards, iteration, learner_id, rnd):
    file_name = path_rewards + "rwd_" + str(iteration) + "_" + str(learner_id) + "_" + str(rnd) + ".json"
    with open(file_name, "w") as f:
        f.write(json.dumps(rewards))
    
def log_losses(loss, iteration, learner_id):
    file_name = path_losses + "loss_" + str(iteration) + "_" + str(learner_id) + ".json"
    with open(file_name, "w") as f:
        f.write(json.dumps(loss))
    
def read_metadata():
    with open(path_meta, "r") as f:
        data = json.load(f)
    return data

def save_model(agent, shared_params):
    agent.save_model(shared_params, path_model_pi, path_model_v)
    
def load_model(net):
    net.load_model(path_model_pi, path_model_v)

import os
import shutil
import json
import time

root = 'files'
path_rewards = 'files/rewards/'
path_losses = 'files/losses/'
path_meta = 'files/metadata.json'
path_model_pi = 'files/model_pi.model'
path_model_v = 'files/model_v.model'

def create_folders(atari_name, cores, tmax, game_length, Tmax, C, gamma, lr):
    if os.path.exists(root):
        # Delete if exists.
        # print ('The folder named files will be deleted!')
        # input ('Press Enter to continue.')
        shutil.rmtree(root)
        
    # Create the new folders.
    os.makedirs(path_rewards)
    os.makedirs(path_losses)
        
    metadata = [time.strftime("%d/%m/%y"), atari_name, 'cores '+str(cores), 'tmax '+str(tmax), 'gl '+str(game_length), 'Tmax '+str(Tmax), 'C '+str(C), 'gamma '+str(gamma), 'lr '+str(lr)]
    with open(path_meta, "w") as f:
        f.write(json.dumps(metadata))

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

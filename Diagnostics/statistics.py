import json
import os
import matplotlib.pyplot as plt
import argparse
import math as m

parser = argparse.ArgumentParser(description='A3C statistics')

parser.add_argument('--show-all', action='store_true',
        help='call show_all function')


path_rewards = 'files/rewards/'
path_losses = 'files/losses/'
path_meta = 'files/metadata.json'

# CREATE PLOTS and draw some STATISTICAL DATA

def step_progress_bar():
    print ('* ', end="")

def collect_reward_file_names():
    total_rewards = {} 
    list_of_names = os.listdir(path_rewards)
    prev_progress = 0
    progress = 0
    
    print ('Processing reward files.')
    for idx in range(len(list_of_names)):
        progress = m.floor(float(idx)/float(len(list_of_names)) * 10)
        if progress > prev_progress:
            step_progress_bar()
            prev_progress = progress
        
        attributes = list_of_names[idx].split(".")[0] # Cut the json part from the string.
        attributes = attributes.split("/") # Cut the string part with the numbers.
        attributes = attributes[0].split("_")[1:4] # Contains the iteration, learner_id and rnd values.
        
        iteration = int(attributes[0])
        learner_id = int(attributes[1])
        
        file_name = path_rewards + list_of_names[idx]
        with open(file_name, "r") as f:
            rewards = json.load(f)
        
        summa = 0.0
        for x in rewards:
            summa += x
            
        if learner_id not in list(total_rewards):
            total_rewards[learner_id] = {}
        if iteration in list(total_rewards[learner_id]):
             total_rewards[learner_id][iteration].append(summa)
        else:
            total_rewards[learner_id][iteration] = [summa]
    
    print (' -> DONE')
    
    return total_rewards
    
def calculate_reward_statistics(total_rewards, learner_id):
    iterations = list(total_rewards[learner_id])
    
    # Calculating the averages.
    avg_rewards = [0] * len(iterations)
    
    # Calculating the averages.
    max_rewards = [0] * len(iterations)
    
    # Calculating the averages.
    min_rewards = [0] * len(iterations)
    
    for it in range(len(iterations)):
        min_rewards[it] = total_rewards[learner_id][iterations[it]][0]
        max_rewards[it] = total_rewards[learner_id][iterations[it]][0]
        for rw in total_rewards[learner_id][iterations[it]]:
            avg_rewards[it] += rw           
            if min_rewards[it] > rw:
                min_rewards[it] = rw
            if max_rewards[it] < rw:
                max_rewards[it] = rw
        avg_rewards[it] = avg_rewards[it]/len(total_rewards[learner_id][iterations[it]])
        
    return [avg_rewards, min_rewards, max_rewards, iterations]

def collect_loss_file_names():
    total_losses = {} 
    list_of_names = os.listdir(path_losses)
    prev_progress = 0
    progress = 0
    
    print ('Processing loss files.') 
    for idx in range(len(list_of_names)):
        progress = m.floor(float(idx)/float(len(list_of_names)) * 10)
        if progress > prev_progress:
            step_progress_bar()
            prev_progress = progress
        
        attributes = list_of_names[idx].split(".")[0] # Cut the json part from the string.
        attributes = attributes.split("/") # Cut the string part with the numbers.
        attributes = attributes[0].split("_")[1:3] # Contains the iteration, learner_id.
        
        iteration = int(attributes[0])
        learner_id = int(attributes[1])
        
        file_name = path_losses + list_of_names[idx]
        with open(file_name, "r") as f:
            loss = json.load(f)
            
        if learner_id not in list(total_losses):
            total_losses[learner_id] = {}
        if iteration not in list(total_losses[learner_id]):
             total_losses[learner_id][iteration] = loss
    
    print (' -> DONE')
    
    return total_losses

def loss_of_learner(total_losses, learner_id):
    
    iterations = list(total_losses[learner_id])
    losses = list(total_losses[learner_id].values())
    
    return [losses, iterations]
    
def draw_simple_graphs(num, x, y, y_label):
    
    plt.figure(num)
    plt.plot(x, y, 'ro')
    plt.xlabel('Number of actions')
    plt.ylabel(y_label)
    plt.show()

def show_all(learner_id):
    
    with open(path_meta, 'r') as f:
        metadat = json.load(f)
    print (metadat)
    
    tot_rwds = collect_reward_file_names()
    tot_losses = collect_loss_file_names()
    
    print('Calculate statistics.')
    reward_stats = calculate_reward_statistics(tot_rwds, learner_id)
    loss_data = loss_of_learner(tot_losses, learner_id)
    
    print('Show graphs.')
    draw_simple_graphs(1, reward_stats[3], reward_stats[0], 'average reward')
    draw_simple_graphs(2, reward_stats[3], reward_stats[1], 'minimum reward')
    draw_simple_graphs(3, reward_stats[3], reward_stats[2], 'maximum reward')
    draw_simple_graphs(4, loss_data[1], loss_data[0], 'loss')

if show_all:
    show_all(0)


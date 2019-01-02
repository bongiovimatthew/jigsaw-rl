import numpy as np
from A3C import dnn
from Environment.env import PuzzleEnvironment
from Environment.env import Actions
from Diagnostics.logger import Logger as logger
from celery.contrib import rdb
from PIL import Image
import json
import cv2,time
from HumanGame.humanAgent import Key
# In case of Pool the Lock object can not be passed in the initialization argument.
# This is the solution
lock = 0
shared = 0
def init_lock_shared(l, sh):
    global lock
    global shared
    lock = l
    shared = sh

# Easier to call these functions from other modules.

def create_shared(cut_pieces_num):
    temp_env = PuzzleEnvironment(cut_pieces_num)
    # temp_env = gym.make(env_name)
    num_actions = temp_env.action_space.n
    net = dnn.DeepNet(num_actions, 0)
    # temp_env.close()
    
    prms_pi = net.get_parameters_pi()
    prms_v = net.get_parameters_v()
    
    return [prms_pi, prms_v]

def execute_agent(learner_id, puzzle_env, t_max, game_length, T_max, epoch_size, eval_num, gamma, lr,num_cores,cut_pieces_num):
    agent = create_agent(puzzle_env, t_max, game_length, T_max, epoch_size, eval_num, gamma, lr,num_cores,cut_pieces_num)
    agent.run(learner_id)
        
def create_agent(puzzle_env, t_max, game_length, T_max, epoch_size, eval_num, gamma, lr,num_cores,cut_pieces_num):
    agent = Agent(puzzle_env, t_max, game_length, T_max, epoch_size, eval_num, gamma, lr,num_cores,cut_pieces_num)
    # logger.load_model(agent.get_net())  # pick up where it's left of

    return agent
    
def create_agent_for_evaluation(cut_pieces_num):
    
    # read the json with data (environemnt name and dnn model)
    
    # meta_data = logger.read_metadata()
    # env_name = meta_data[1]
    
    agent = Agent("puzzle", 50000, 50000, 0, 0, 0, 0, 0,1,cut_pieces_num) 
    logger.load_model(agent.get_net())
    
    return agent

# During a game attempt, a sequence of observation are generated.
# The last four always forms the state. Rewards and actions also saved.
class OutFiles:
    def __init__(self,configs):
        self.perf_train = logger.path_graphs + "train_learner_id_{learner_id}_T_max{T_max}_lr{lr}_gamma{gamma}_t_max{t_max}_cores{cores}_missing_pieces{cut_pieces_num}.jpg".format(**configs) 
        self.perf_eval = logger.path_metrics + "eval_learner_id_{learner_id}_T_max{T_max}_lr{lr}_gamma{gamma}_t_max{t_max}_cores{cores}_missing_pieces{cut_pieces_num}.jpg".format(**configs)
        self.metrics_log = logger.path_metrics + "train_learner_id_{learner_id}_T_max{T_max}_lr{lr}_gamma{gamma}_t_max{t_max}_cores{cores}_missing_pieces{cut_pieces_num}.tsv".format(**configs)

class Queue:
    
    def __init__(self, max_size, img_dimensions):
        self.size = max_size
        
        self.observations = np.ndarray((self.size, img_dimensions, img_dimensions), dtype=np.uint8) # save memory with uint8
        self.rewards = np.ndarray((self.size))
        self.actions = np.ndarray((self.size))
        
        self.last_idx = -1
        self.is_last_terminal = False
    
    def get_last_idx(self):
        return self.last_idx
        
    def get_is_last_terminal(self):
        return self.is_last_terminal
    
    def queue_reset(self):
        self.observations.fill(0)
        self.rewards.fill(0)
        self.actions.fill(0)
        self.last_idx = -1
        self.is_last_terminal = False
        
    def add(self, observation, reward, action, done):
        self.last_idx += 1
        self.observations[self.last_idx, :, :] = observation[0,:,:]
        # if reward > 1.0:
        #     reward = 1.0 # reward clipping
        self.rewards[self.last_idx] = reward
        self.actions[self.last_idx] = action
        
        self.is_last_terminal = done
        
    def get_recent_state(self):
        if self.last_idx >= 0:
            return np.float32(self.observations[self.last_idx:self.last_idx+1,:,:])
        return None
        
    def get_state_at(self, idx):
        # if idx > 2:
        #     return np.float32(self.observations[idx-3:idx+1,:,:])
        
        if idx >= 0:
            return np.float32(self.observations[idx:idx+1,:,:])

        return None
    
    def get_reward_at(self, idx):
        return self.rewards[idx]
        
    def get_recent_reward(self):
        return self.rewards[self.last_idx]
    
    def get_action_at(self, idx):
        return self.actions[idx]

# Preprocessing of the raw frames from the game.
def process_img(observation):
    img_final = np.array(Image.fromarray(observation, 'RGB').convert("L").resize((84,84), Image.ANTIALIAS))
    img_final = np.reshape(img_final, (1, 84, 84))

    return img_final



# Functions to avoid temporary coupling.
def env_reset(env, queue):
    queue.queue_reset()
    obs = env.reset()
    queue.add(process_img(obs), 0, 0, False)
    return queue.get_recent_state() # should return None
    
def env_step(env, queue, action):
    obs, rw, done, info = env.step(action)
    # pdb.set_trace()
    # if (rw > 0):
    #     print("Action:{0}, rewards:{1}".format(action, rw))
    # Add rewards to info
    info["reward"] = rw
    queue.add(process_img(obs), rw, action, done)
    return queue.get_recent_state(), info

def exponential_decay(step, total, initial, final, rate=1e-4, stairs=None):
    if stairs is not None:
        step = stairs * tf.floor(step / stairs)
    scale, offset = 1. / (1. - rate), 1. - (1. / (1. - rate))
    progress = step / total
    value = (initial - final) * scale * rate ** progress + offset + final
    lower, upper = min(initial, final), max(initial, final)
    return max(lower, min(value, upper))

class Agent:
    
    def __init__(self, env_name, t_max, game_length, T_max, epoch_size, eval_num, gamma, lr,num_cores,cut_pieces_num):
        
        self.t_start = 0
        self.t = 0
        self.counter = 0 
        self.t_max = t_max
        self.lr = lr 
        self.game_length = game_length
        self.T = 0
        self.T_max = T_max
        self.num_cores = num_cores
        self.epoch_size = epoch_size
        self.eval_num = eval_num
        self.gamma = gamma
        self.cut_pieces_num = cut_pieces_num
        self.is_terminal = False
        self.env = PuzzleEnvironment(cut_pieces_num)
        self.queue = Queue(game_length, 84) 
        self.net = dnn.DeepNet(self.env.action_space.n, lr)
        self.s_t = env_reset(self.env, self.queue)
        
        self.R = 0
        self.training_report = False
        self.loss_on_v = []
        self.loss_on_p = []
        self.diff = []
        self.epsilon = 1.0
        self.debugMode = True 

        self.reward_count = {}
        self.average_rewards = 0
        self.averageScore = 0
        self.slidingWindowAverageScore = 0
        self.numStepsForRunningMetrics = 0
        self.stepsForaverage_rewards = 0
        self.slidingWindowScoresArray = []
        self.numberOfTimesExecutedEachAction = [0 for i in range(self.env.action_space.n)]
        self.metrics = {}
        self.metricsHistory = []
        self.start = time.time()
    
    def get_net(self):
        return self.net
    
    # For details: https://arxiv.org/abs/1602.01783
    def run(self, learner_id):
        
        self.learner_id = learner_id
        configs = {'learner_id':self.learner_id,'T_max':self.T_max, 'lr':self.lr, 'gamma':self.gamma,'t_max':self.t_max,'cores':self.num_cores,'cut_pieces_num':self.cut_pieces_num}
        outfiles = OutFiles(configs)
        if self.learner_id == 0: 
            cv2.namedWindow('puzzle',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('puzzle', 600,600)
        while self.T < self.T_max:


            self.synchronize_dnn()
            
            self.play_game_for_a_while()
            
            self.set_R()

            self.calculate_gradients()
            
            self.sync_update() # Syncron update instead of asyncron!


            if self.training_report:
                self.training_report = False
                metric_names = ["iteration","epsilon","negativeRewardCount","zeroRewardCount","terminal_reward_count","average_rewards"]
                metric_names = metric_names + ["averageScore","slidingWindowAverageScore","loss_on_v","loss_on_p","reward_count","time(s)"]
                logger.log_metrics(self.metrics, outfiles.metrics_log,metric_names)
                self.reset_running_metrics()


            if (self.T%1000 == 0): 
                self.save_model_snapshot()

        logger.plot_history(self.metricsHistory,outfiles.perf_train,metric_names = ["terminal_reward_count","loss_on_v","average_rewards"])        
        print("Completed run")
    # IMPLEMENTATIONS FOR the FUNCTIONS above
        
    def synchronize_dnn(self):
        lock.acquire()
        try:
            self.net.synchronize_net(shared) # the shared parameters are copied into 'net'
        finally:
            lock.release()

    def update_metrics(self, info, action):
        


        reward = info["reward"]
        if reward in self.reward_count:
            self.reward_count[reward] += 1 
        else:
            self.reward_count[reward] = 1 
        score = info["score"]
        self.numStepsForRunningMetrics += 1
        self.stepsForaverage_rewards += 1
        self.average_rewards = ((self.average_rewards * (self.stepsForaverage_rewards - 1)) + reward) / self.stepsForaverage_rewards
        self.averageScore = ((self.averageScore * (self.t - 1)) + score) / self.t

        self.slidingWindowAverageScore = 0
        self.slidingWindowScoresArray.append(score)
        self.numberOfTimesExecutedEachAction[action] += 1

        if len(self.slidingWindowScoresArray) > 50:
            self.slidingWindowScoresArray.pop(0)

        self.slidingWindowAverageScore = sum(self.slidingWindowScoresArray) / len(self.slidingWindowScoresArray)

        self.metrics["negativeRewardCount"] = np.sum([v for k,v in self.reward_count.items() if k < 0]) #self.negativeRewardCount
        self.metrics["zeroRewardCount"] = self.reward_count.get(0)
        self.metrics["terminal_reward_count"] = self.reward_count.get(1)
        self.metrics["average_rewards"] = self.average_rewards
        self.metrics["averageScore"] = self.averageScore
        self.metrics["slidingWindowAverageScore"] = self.slidingWindowAverageScore
        self.metrics["numberOfTimesExecutedEachAction"] = self.numberOfTimesExecutedEachAction
        self.metrics["reward_count"] = json.dumps(self.reward_count)

        return info

    def update_loss_metric(self):
        loss = self.net.get_avg_minibatch_loss()
        self.loss_on_v.append(loss[0])
        self.loss_on_p.append(loss[1])
        self.metrics['loss_on_v'] = np.mean(np.array(self.loss_on_v))
        self.metrics['loss_on_p'] = np.mean(np.array(self.loss_on_p))
        self.metrics['iteration'] = self.T
        self.metrics['epsilon'] = self.epsilon
        self.metrics['time(s)'] = time.time() - self.start
    def reset_running_metrics(self):
        self.metricsHistory.append(self.metrics.copy())
        self.reward_count = {}
        self.average_rewards = 0
        self.stepsForaverage_rewards = 0
        self.numStepsForRunningMetrics = 0
        self.numberOfTimesExecutedEachAction = [0 for i in range(self.env.action_space.n)]
        self.loss_on_v = []
        self.loss_on_p = []
    def play_game_for_a_while(self):
    
        if self.is_terminal:
            self.s_t = env_reset(self.env, self.queue)
            self.t = 0
            self.is_terminal = False
            
        self.t_start = self.t
        
        self.epsilon = max(0.1, 1.0 - (((1.0 - 0.1)*1)/self.T_max) * self.T) # first decreasing, then it is constant
        # self.epsilon = exponential_decay(self.T, self.T_max, 1.0, 0.05, rate=.5)
        while not (self.is_terminal or self.t - self.t_start == self.t_max):

            self.t += 1
            self.counter += 1
            self.T += 1
            # if (self.T < 20000 and self.T%2000 in range(60)):
            #     action = self.action_via_operator()
            #     print("iteration:%d"%self.T)
            # else:
            action = dnn.action_with_exploration(self.net, self.s_t, self.epsilon)
            self.s_t, info = env_step(self.env, self.queue, action)
            if self.learner_id == 0:
                cv2.imshow('puzzle',self.env.render())
                cv2.waitKey(5) 
            self.update_metrics(info, action)

            if (self.T % self.epoch_size == 0 and self.T!=0): # log loss when evaluation happens
                self.training_report = True
            
            if self.T % 5000 == 0:
                print('Actual iter. num.: ' + str(self.T))
        
            self.is_terminal = self.queue.get_is_last_terminal()
       
    def action_via_operator(self):
        cv2.imshow('puzzle',self.env.render())
        key = cv2.waitKey(0)
        if key == 27: # exit on ESC
            return False
        action = self.getActionFromUserInput(key)  
        return action

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

    def set_R(self):
        if self.is_terminal:
            self.R = np.array([[0.0]])
        else:
            self.R = self.net.state_value(self.s_t)
        
    def calculate_gradients(self):
        states = []
        actions = []
        targets = []
        idx = self.queue.get_last_idx()
        final_index = idx - self.t_max
        while (idx > self.t_start):
            states.append(self.queue.get_state_at(idx-1))
            reward = self.queue.get_reward_at(idx)
            action = self.queue.get_action_at(idx)
            action_as_array = np.zeros(self.net.num_actions, dtype=np.float32)
            action_as_array[int(action)] = 1
            actions.append(action_as_array.copy())
            target = reward + self.gamma * self.R
            targets.append(np.float32(target))
            idx -= 1
        
        self.diff = self.net.train_net(np.array(states),np.array(actions),np.array(targets))
        self.update_loss_metric()

    def sync_update(self):
        lock.acquire()
        try:
            self.net.sync_update(shared, self.diff)
        finally:
            lock.release()
       
    def evaluate_during_training(self):
        
        # print ('Evaluation at: ' + str(self.T))
        
        for rnd in range(self.eval_num): # Run more game epsiode to get more robust result for performance
            state = env_reset(self.env, self.queue)
            finished = False
            cntr = 0
            rewards = []
            while not (finished or cntr == self.game_length):
                action = dnn.action(self.net, state)
                state, info = env_step(self.env, self.queue, action)
                rewards.append(self.queue.get_recent_reward())
                
                finished = self.queue.get_is_last_terminal()
                cntr += 1
            
            logger.log_rewards(rewards, self.T, self.learner_id, rnd)
            
    def evaluate(self):
        
        print ('Start evaluating.')
        while True:
            env = PuzzleEnvironment()
            state = env_reset(env, self.queue)
            finished = False
            cntr = 0
            rewards = []
            cv2.namedWindow('puzzle',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('puzzle', 600,600)
            while not (finished or cntr == self.game_length):
                # img = Image.fromarray(env.render(), 'RGB')


                action = dnn.action(self.net, state)
                state, info = env_step(env, self.queue, action)
                rewards.append(self.queue.get_recent_reward())
                cv2.imshow('puzzle',env.render())
                cv2.waitKey(500)                    
                finished = self.queue.get_is_last_terminal()
                cntr += 1  
             
            # Representing the results. 
            print ('The collected rewards over duration:')
            total_rw = 0.0
            for x in rewards:
                total_rw += x
            print (total_rw)

    def save_model_snapshot(self):
        lock.acquire()
        try:
            self.net.save_model(logger.path_model_pi,logger.path_model_v)
        finally:
            lock.release()
    def save_model(self, shared_params, path_model_pi, path_model_v):
        self.net.synchronize_net(shared_params) # copy the parameters into the recently created agent's netork
        self.net.save_model(path_model_pi, path_model_v)
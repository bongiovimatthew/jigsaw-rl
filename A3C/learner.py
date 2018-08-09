import numpy as np

from A3C.dnn import dnn
from Environment.env import PuzzleEnvironment
from Diagnostics.logger import Logger as logger

from PIL import Image

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
def create_shared(env_name):
    temp_env = PuzzleEnvironment()
    # temp_env = gym.make(env_name)
    num_actions = temp_env.action_space.n
    net = dnn.DeepNet(num_actions, 0)
    # temp_env.close()
    
    prms_pi = net.get_parameters_pi()
    prms_v = net.get_parameters_v()
    
    return [prms_pi, prms_v]

def execute_agent(learner_id, puzzle_env, t_max, game_length, T_max, C, eval_num, gamma, lr):
    agent = create_agent(puzzle_env, t_max, game_length, T_max, C, eval_num, gamma, lr)
    agent.run(learner_id)
        
def create_agent(puzzle_env, t_max, game_length, T_max, C, eval_num, gamma, lr):
    agent = Agent(puzzle_env, t_max, game_length, T_max, C, eval_num, gamma, lr)
    # logger.load_model(agent.get_net())

    return agent
    
def create_agent_for_evaluation():
    
    # read the json with data (environemnt name and dnn model)
    
    # meta_data = logger.read_metadata()
    # env_name = meta_data[1]
    
    agent = Agent("puzzle", 50000, 50000, 0, 0, 0, 0, 0) 
    logger.load_model(agent.get_net())
    
    return agent

# During a game attempt, a sequence of observation are generated.
# The last four always forms the state. Rewards and actions also saved.
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
    info["rewards"] = rw
    queue.add(process_img(obs), rw, action, done)
    return queue.get_recent_state(), info


class Agent:
    
    def __init__(self, env_name, t_max, game_length, T_max, C, eval_num, gamma, lr):
        
        self.t_start = 0
        self.t = 0
        self.t_max = t_max
        
        self.game_length = game_length
        self.T = 0
        self.T_max = T_max
        
        self.C = C
        self.eval_num = eval_num
        self.gamma = gamma
        
        self.is_terminal = False
        
        self.env = PuzzleEnvironment()
        self.queue = Queue(game_length, 84) 
        self.net = dnn.DeepNet(self.env.action_space.n, lr)
        self.s_t = env_reset(self.env, self.queue)
        
        self.R = 0
        self.signal = False
        
        self.diff = []
        self.epsilon = 1.0
        self.debugMode = True 

        self.negativeRewardCount = 0
        self.zeroRewardCount = 0
        self.positiveRewardCount = 0
        self.averageRewards = 0
        self.averageScore = 0
        self.slidingWindowAverageScore = 0
        self.numStepsForRunningMetrics = 0
        self.stepsForAverageRewards = 0
        self.slidingWindowScoresArray = []
        self.numberOfTimesExecutedEachAction = [0 for i in range(self.env.action_space.n)]
    
    def get_net(self):
        return self.net
    
    # For details: https://arxiv.org/abs/1602.01783
    def run(self, learner_id):
        
        self.learner_id = learner_id
        
        print(self.T_max)
        while self.T < self.T_max:


            self.synchronize_dnn()
            
            self.play_game_for_a_while()
            
            self.set_R()
            
            # According to the article the gradients should be calculated.
            # Here: The parameters are updated and the differences are added to the shared NN's.
            self.calculate_gradients()
            
            self.sync_update() # Syncron update instead of asyncron!
            print(self.T)
            
            if (self.T%100 == 0): 
                self.save_model_snapshot()
            if self.signal:
                self.evaluate_during_training()
                self.signal = False


        print("Completed run")
    # IMPLEMENTATIONS FOR the FUNCTIONS above
        
    def synchronize_dnn(self):
        lock.acquire()
        try:
            self.net.synchronize_net(shared) # the shared parameters are copied into 'net'
        finally:
            lock.release()

    def update_and_get_metrics(self, info, action):
        rewards = info["rewards"]
        score = info["score"]
        self.numStepsForRunningMetrics += 1
        self.stepsForAverageRewards += 1

        if rewards < 0:
            self.negativeRewardCount += 1

        elif rewards == 0:
            self.zeroRewardCount += 1

        elif rewards > 0:
            self.positiveRewardCount += 1


        self.averageRewards = ((self.averageRewards * (self.stepsForAverageRewards - 1)) + rewards) / self.stepsForAverageRewards
        self.averageScore = ((self.averageScore * (self.t - 1)) + score) / self.t

        self.slidingWindowAverageScore = 0
        self.slidingWindowScoresArray.append(score)
        self.numberOfTimesExecutedEachAction[action] += 1

        if len(self.slidingWindowScoresArray) > 50:
            self.slidingWindowScoresArray.pop(0)

        self.slidingWindowAverageScore = sum(self.slidingWindowScoresArray) / len(self.slidingWindowScoresArray)

        info["negativeRewardCount"] = self.negativeRewardCount
        info["zeroRewardCount"] = self.zeroRewardCount
        info["positiveRewardCount"] = self.positiveRewardCount
        info["averageRewards"] = self.averageRewards
        info["averageScore"] = self.averageScore
        info["slidingWindowAverageScore"] = self.slidingWindowAverageScore
        info["numberOfTimesExecutedEachAction"] = self.numberOfTimesExecutedEachAction

        return info

    def reset_running_metrics(self):
        self.negativeRewardCount = 0
        self.positiveRewardCount = 0
        self.zeroRewardCount = 0
        self.averageRewards = 0
        self.stepsForAverageRewards = 0
        self.numStepsForRunningMetrics = 0
        self.numberOfTimesExecutedEachAction = [0 for i in range(self.env.action_space.n)]

    def play_game_for_a_while(self):
    
        if self.is_terminal:
            self.s_t = env_reset(self.env, self.queue)
            self.t = 0
            self.is_terminal = False
            
        self.t_start = self.t
        
        self.epsilon = max(0.1, 1.0 - (((1.0 - 0.1)*1)/self.T_max) * self.T) # first decreasing, then it is constant
        
        while not (self.is_terminal or self.t - self.t_start == self.t_max):
            self.t += 1
            self.T += 1
            # img = Image.fromarray(self.s_t, 'RGB')
            # img.save('files/image_agent_%d.jpg'%(self.learner_id)) 
            action = dnn.action_with_exploration(self.net, self.s_t, self.epsilon)

            self.s_t, info = env_step(self.env, self.queue, action)


            info = self.update_and_get_metrics(info, action)

            if self.debugMode and self.t < 40 : 
                logger.log_metrics(info, self.t, self.learner_id)
                logger.log_state_image(self.s_t, self.t, self.learner_id,action)
                self.reset_running_metrics()

            self.is_terminal = self.queue.get_is_last_terminal()
            if self.T % self.C == 0: # log loss when evaluation happens
                self.signal = True
            if self.T % 5000 == 0:
                print('Actual iter. num.: ' + str(self.T))
        
    def set_R(self):
        if self.is_terminal:
            self.R = self.net.state_value(self.s_t)
            self.R[0][0] = 0.0 # Without this, error dropped. special format is given back.
        else:
            self.R = self.net.state_value(self.s_t)
        
    def calculate_gradients(self):

        idx = self.queue.get_last_idx()
        final_index = idx - self.t_max

        states = []
        rewards = []
        actions = []
        Rs = []
        while idx >= final_index: # the state is 4 pieces of frames stacked together -> at least 4 frames are necessary
            states.append(self.queue.get_state_at(idx))
            reward = (self.queue.get_reward_at(idx))
            actions.append(self.queue.get_action_at(idx))

            self.R = (reward + self.gamma * self.R)
            Rs.append(self.R)

            print("action:{2}, reward:{0}, self.R:{1}".format(reward, self.R, action))
            idx = idx-1        

            # print("action:{0}, Reward:{1}".format(action, self.R))
        self.diff = self.net.train_net(states, actions, Rs, False)
            
        if self.signal:
            logger.log_losses(self.net.get_last_avg_loss(), self.T, self.learner_id)
        
    def sync_update(self):
        lock.acquire()
        try:
            self.net.sync_update(shared, self.diff)
        finally:
            lock.release()
        
    def evaluate_during_training(self):
        
        print ('Evaluation at: ' + str(self.T))
        
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
        env = wrappers.Monitor(self.env, 'videos', force=True)
        state = env_reset(env, self.queue)
        finished = False
        cntr = 0
        rewards = []
        while not (finished or cntr == self.game_length):
            img = Image.fromarray(env.render(), 'RGB')
            img.show()
            env.render()
            action = dnn.action(self.net, state)
            state, info = env_step(env, self.queue, action)
            rewards.append(self.queue.get_recent_reward())
                
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
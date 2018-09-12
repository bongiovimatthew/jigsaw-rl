import numpy as np
from PIL import Image

from Brain.dnn import DeepNetBrain
#from Brain.kerasNetBrain import KerasNetBrain
from Brain.tfBrain import TFBrain 

from Diagnostics.logger import Logger as logger
from ActionChoosers.EpsilonGreedyActionChooser import EpsilonGreedyActionChooser
from Learners.BatchQueue import BatchQueue

class ActorCriticLearner:

    STATE_WIDTH = 224
    STATE_HEIGHT = 224

    def execute_agent(env, batch_length, game_length, total_max_moves, gamma, lr):
        agent = ActorCriticLearner.create_agent(
            env, batch_length, game_length, total_max_moves, gamma, lr)
        agent.run()

    def create_agent(env, batch_length, game_length, total_max_moves, gamma, lr):
        agent = ActorCriticAgent(env, batch_length, game_length, total_max_moves, gamma, lr)
        return agent

    # Preprocessing of the raw frames from the game
    def process_img(observation):
        img_final = np.array(Image.fromarray(observation, 'RGB').convert("L").resize(
            (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT), Image.ANTIALIAS))
        img_final = np.reshape(img_final, (1, ActorCriticLearner.STATE_WIDTH,
                                           ActorCriticLearner.STATE_HEIGHT))

        return img_final

    def env_reset(env, queue):
        queue.queue_reset()
        obs = env.reset()
        queue.add(ActorCriticLearner.process_img(obs), 0, 0, False)
        return queue.get_recent_state()  # should return None

    def env_step(env, queue, action):
        obs, rw, done, info = env.step(action)
        queue.add(ActorCriticLearner.process_img(obs), rw, action, done)
        return queue.get_recent_state(), info

class ActorCriticAgent:

    def __init__(self, env, batch_length, game_length, total_max_moves, gamma, lr):

        self.env = env

        brain = DeepNetBrain(self.env.action_space.n, lr,
                             (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))
        self.ActionChooser = EpsilonGreedyActionChooser(brain)

        self.gamma = gamma
        self.is_terminal = False
        
        self.queue = BatchQueue(total_max_moves, ActorCriticLearner.STATE_HEIGHT) 

        self.current_state = ActorCriticLearner.env_reset(self.env, self.queue)        

        self.R = 0

        self.episode_step_count = 0
        self.total_step_count = 0
        self.total_max_moves = total_max_moves

        self.batch_length = batch_length     
        self.game_length = game_length
        self.learner_id = "ActorCriticLearner"   
        self.debug_mode = False

    def run(self):

        while self.total_step_count < self.total_max_moves:

            self.play_game_for_a_while()
            self.set_R()
            self.calculate_gradients()

    def set_R(self):

        self.R = self.ActionChooser.get_brain().state_value(self.current_state)

        if self.is_terminal:
            self.R[0][0] = 0.0  # Without this, error dropped. special format is given back.

    def save_model(self, path_model_pi, path_model_v):
        self.ActionChooser.get_brain().save_model(path_model_pi, path_model_v)

    def play_game_for_a_while(self):
        if self.is_terminal:
            if self.debug_mode:
                logger.log_state_image(self.current_state, self.total_step_count, -1,
                                       (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))
            self.current_state = ActorCriticLearner.env_reset(self.env, self.queue)
            self.episode_step_count = 0
            self.is_terminal = False

        self.epsilon = max(0.1, 1.0 - (((1.0 - 0.1)*1.5) / self.total_max_moves)
                           * self.total_step_count)  # first decreasing, then it is constant
        
        batch_count = 0
        batch_count_start = self.episode_step_count
        
        while not (self.is_terminal or batch_count - batch_count_start == self.batch_length):
            self.episode_step_count += 1
            self.total_step_count += 1
            batch_count += 1

            action = self.ActionChooser.action(self.current_state, self.epsilon)
            self.current_state, info = ActorCriticLearner.env_step(self.env, self.queue, action)

            if self.debug_mode:
                logger.log_metrics(info, self.episode_step_count, self.learner_id)
                logger.log_state_image(self.current_state, self.episode_step_count, self.learner_id,
                                       action, (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))

                self.reset_running_metrics()

            self.is_terminal = self.queue.get_is_last_terminal()

    def calculate_gradients(self):

        idx = self.queue.get_last_idx()
        final_index = max(idx - self.batch_length, 0)


        states = []
        rewards = []
        actions = []
        Rs = []

        while idx > final_index:

            states.append(self.queue.get_state_at(idx))
            reward = (self.queue.get_reward_at(idx))
            actions.append(self.queue.get_action_at(idx))

            self.R = (reward + self.gamma * self.R)
            Rs.append(self.R)
            idx = idx - 1

        self.ActionChooser.get_brain().train(states, actions, Rs, False)

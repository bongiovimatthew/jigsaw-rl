import numpy as np
import random
from PIL import Image

from Brain.dnn import DeepNetBrain
# from Brain.kerasNetBrain import KerasNetBrain 
from Brain.tfBrain import TFBrain 

from Diagnostics.logger import Logger as logger
from ActionChoosers.EpsilonGreedyActionChooser import EpsilonGreedyActionChooser
from Learners.BatchQueue import BatchQueue
from Learners.CircularBatchQueue import CircularBatchQueue

class ActorCriticLearner:

    STATE_WIDTH = 160
    STATE_HEIGHT = 160

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
        return obs

    def env_step(env, queue, action):
        obs, rw, done, info = env.step(action)
        return obs, rw, done, info

class ActorCriticAgent:

    def __init__(self, env, batch_length, game_length, total_max_moves, gamma, lr):

        self.env = env

        brain = TFBrain(self.env.action_space.n, lr, (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))
        self.ActionChooser = EpsilonGreedyActionChooser(brain)

        self.gamma = gamma
        self.is_terminal = False

        self.use_experience_replay = True
        
        if self.use_experience_replay:
            self.queue = CircularBatchQueue(20, ActorCriticLearner.STATE_HEIGHT) 
        else:
            self.queue = BatchQueue(total_max_moves, ActorCriticLearner.STATE_HEIGHT) 

        self.current_state = ActorCriticLearner.process_img(ActorCriticLearner.env_reset(self.env, self.queue))        

        self.queue.add(self.current_state, 0, 0, False)
        self.queue.set_discounted_reward_at(self.queue.get_last_idx(), 0)

        self.R = 0

        self.episode_step_count = 0
        self.total_step_count = 0
        self.total_max_moves = total_max_moves
        self.current_step_count = 0

        self.batch_length = batch_length
        self.training_batch_length = batch_length     
        self.game_length = game_length
        self.learner_id = "ActorCriticLearner"   
        self.debug_mode = False
        self.pause_when_training = False

    def run(self):

        while self.total_step_count < self.total_max_moves:

            self.play_game_for_a_while()
            self.set_R()
            self.calculate_gradients()
            self.calculate_gradients()

        self.debug_mode = True

        # Just test essentially i.e. no train
        while self.total_step_count < (self.total_max_moves + 100):
            self.play_game_for_a_while()


    def set_R(self):
        self.R = self.ActionChooser.get_brain().state_value(self.current_state)

        if self.is_terminal:
            self.R[0][0] = 0.0  # Without this, error dropped. special format is given back.

    def save_model(self, path_model_pi, path_model_v):
        self.ActionChooser.get_brain().save_model(path_model_pi, path_model_v)

    def play_game_for_a_while(self):
        self.current_step_count = 0

        if self.is_terminal:
            if self.debug_mode:
                logger.log_state_image(self.current_state, self.total_step_count, self.learner_id, -1,
                                       (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))
            self.current_state = ActorCriticLearner.process_img(ActorCriticLearner.env_reset(self.env, self.queue))

            self.queue.add(self.current_state, 0, 0, False)
            self.queue.set_discounted_reward_at(self.queue.get_last_idx(), 0)

            self.episode_step_count = 0
            self.is_terminal = False

        self.epsilon = max(0.1, 1.0 - (((1.0 - 0.1)*1.5) / self.total_max_moves)
                           * self.total_step_count)  # first decreasing, then it is constant

        batch_count_start = self.episode_step_count
        batch_count = batch_count_start 

        while not (self.is_terminal or batch_count - batch_count_start == self.batch_length):
            self.current_step_count += 1
            self.episode_step_count += 1
            self.total_step_count += 1
            batch_count += 1

            action = self.ActionChooser.action(self.current_state, self.epsilon)
            next_state, rw, done, info = ActorCriticLearner.env_step(self.env, self.queue, action)

            self.queue.add(self.current_state, rw, action, done)
            self.current_state = ActorCriticLearner.process_img(next_state)

            if self.debug_mode:
                logger.log_metrics(info, self.episode_step_count, self.learner_id)
                logger.log_state_image(self.current_state, self.total_step_count, self.learner_id,
                                       action, (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))

                # self.reset_running_metrics()

            self.is_terminal = done

        self.updateDiscountedRewards()

    def updateDiscountedRewards(self):
        idx = self.queue.get_last_idx()
        final_index = (idx - self.current_step_count) + 1 # max(idx - self.current_step_count, (-1 * self.current_step_count))

        print("Init self.R: ", self.R)
        for idx in range(idx, final_index - 1, -1):
            reward = (self.queue.get_reward_at(idx))

            self.R = (reward + self.gamma * self.R)
            print("self.R:{0}, idx:{1}, reward:{2}, action:{3}".format(self.R, idx, reward, self.queue.get_action_at(idx)))
            self.queue.set_discounted_reward_at(idx, self.R)
            idx = idx - 1

    def get_indices_to_train(self):
        indices_array = None
        idx = self.queue.get_last_idx()
        final_index = (idx - self.current_step_count) + 1

        if self.use_experience_replay:
            indices_array = list(range(final_index, idx + 1))
            random.shuffle(indices_array)
            # indices_array = random.sample(range(self.queue.get_max_index() + 1), min(self.training_batch_length, self.queue.get_max_index() + 1))

        else:
            indices_array = list(range(final_index, idx + 1))

        return indices_array

    def calculate_gradients(self):
        states = []
        actions = []
        discounted_rewards = []

        indices_array = self.get_indices_to_train()

        print("indices_array: ", indices_array)

        for idx in indices_array:
            states.append(self.queue.get_state_at(idx))
            actions.append(self.queue.get_action_at(idx))
            discounted_rewards.append(self.queue.get_discounted_reward_at(idx))
            if self.pause_when_training
                logger.log_state_image(states[-1], self.total_step_count, self.learner_id,
                                       actions[-1], (ActorCriticLearner.STATE_WIDTH, ActorCriticLearner.STATE_HEIGHT))
                print("action: ", actions[-1])
                print("discounted_rewards: ", discounted_rewards[-1])
                input()

        self.ActionChooser.get_brain().train(states, actions, discounted_rewards, False)

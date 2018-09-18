import numpy as np


class CircularBatchQueue:

    def __init__(self, max_size, img_dimensions):
        self.size = max_size

        self.observations = np.ndarray(
            (self.size, 3, img_dimensions, img_dimensions), dtype=np.uint8)  # save memory with uint8
        self.rewards = np.ndarray((self.size))
        self.actions = np.ndarray((self.size))
        self.discounted_rewards = np.ndarray((self.size))

        self.next_index = -1
        self.max_index = -1

        # self.last_idx = -1
        self.is_last_terminal = False

    def get_last_idx(self):
        return self.next_index

    def get_is_last_terminal(self):
        return self.is_last_terminal

    def get_max_index(self):
        return self.max_index

    def queue_reset(self):
        return
        # self.observations.fill(0)
        # self.rewards.fill(0)
        # self.actions.fill(0)
        # self.last_idx = -1
        # self.is_last_terminal = False

    def add(self, observation, reward, action, done):
        self.next_index = (self.next_index + 1) % self.size
        self.max_index = max(self.max_index, self.next_index)
        self.observations[self.next_index, :, :, :] = observation[:, :, :]
        self.rewards[self.next_index] = reward
        self.actions[self.next_index] = action

        self.is_last_terminal = done

    def get_recent_state(self):
        if self.next_index >= 0:
            return np.float32(self.observations[self.next_index:self.next_index+1, :, :, :])
        return None

    def get_state_at(self, idx):
        # if idx >= 0:
        return np.float32(self.observations[idx, :, :, :])
        # return None

    def get_reward_at(self, idx):
        return self.rewards[idx]

    def get_recent_reward(self):
        return self.rewards[self.next_index]

    def get_action_at(self, idx):
        return self.actions[idx]

    def set_discounted_reward_at(self, idx, value):
        self.discounted_rewards[idx] = value

    def get_discounted_reward_at(self, idx):
        return self.discounted_rewards[idx]

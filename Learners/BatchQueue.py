import numpy as np

class BatchQueue:
    
    def __init__(self, max_size, img_dimensions):
        self.size = max_size
        
        self.observations = np.ndarray((self.size, img_dimensions, img_dimensions), dtype=np.uint8) # save memory with uint8
        self.rewards = np.ndarray((self.size))
        self.actions = np.ndarray((self.size))
        self.discounted_rewards = np.ndarray((self.size))
        
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
        self.rewards[self.last_idx] = reward
        self.actions[self.last_idx] = action
        
        self.is_last_terminal = done
        
    def get_recent_state(self):
        if self.last_idx >= 0:
            return np.float32(self.observations[self.last_idx:self.last_idx+1,:,:])
        return None
        
    def get_state_at(self, idx):
        if idx >= 0:
            return np.float32([self.observations[idx,:,:]])
        return None
    
    def get_reward_at(self, idx):
        return self.rewards[idx]
        
    def get_recent_reward(self):
        return self.rewards[self.last_idx]
    
    def get_action_at(self, idx):
        return self.actions[idx]

    def set_discounted_reward_at(self, idx, value):
        self.discounted_rewards[idx] = value

    def get_discounted_reward_at(self, idx):
        return self.discounted_rewards[idx]
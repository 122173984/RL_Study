import numpy as np
import pandas as pd



class QLearningTable:
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions)
    
    def choose_action(self, observation):
        pass
    
    def learn(self, s, a, r, s_):
        pass
    
    def check_state_exist(self, state):
        pass
    

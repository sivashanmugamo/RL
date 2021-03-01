'''
Creates a random agent

Author: @sivashanmugamo
'''

import numpy as np

class RandomAgent:
    def __init__(self, environment):
        '''
        Initializes a random agent & its parameters

        Input:
            environment: environment instance - Environment in which the agent will act in
        '''

        self.env= environment
        self.observation_space= environment.observation_space
        self.action_space= environment.action_space

    def step(self):
        '''
        Chooses an action at random from the provided action set

        Returns:
            int - Randomly chosen action
        '''

        return np.random.choice(self.action_space.n)
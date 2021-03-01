'''
Creates a Q-learning agent

Author: @sivashanmugamo
'''

from Random_Agent import RandomAgent

import random
import numpy as np

class QLearningAgent(RandomAgent):

    def __init__(self, environment, learning_rate: float= 0.01, discount_factor: float= 0.99, epsilon: float= 1.0, epsilon_decay: float= 0.99, episodes: int= 100) -> None:
        '''
        Initializes a Q-learning agent & its parameters

        Input:
            environment: environment instance - Instance of the environment in which the agent will act
            learning_rate: float - Rate at which the agent learns/converges (Default - 0.01)
            discount_factor: float - Rate at which the reward decays (Default - 0.99)
            epsilon: float - Balances the action of the agent between exploration & exploitation (Default - 1.0)
            epsilon_decay: floar - Rate at which the epsilon decays when iterating through episodes (Default - 0.99)
        '''

        super().__init__(environment)

        # Closer to 1.0 - exploration | Closer to 0.0 - exploitation
        self.epsilon= epsilon
        self.epsilon_decay= epsilon_decay

        self.episodes= episodes
        
        self.alpha= learning_rate
        self.gamma= discount_factor

        self.build()

    def build(self) -> None:
        '''
        Initializes the Q-table when an object is created
        '''

        self.Q_table= np.zeros((self.observation_space.n, self.action_space.n))

    def save(self, path: str) -> None:
        '''
        '''

        pass

    def load(self, path: str) -> None:
        '''
        '''

        pass

    def step(self, state: int) -> int:
        '''
        '''

        Q_state= self.Q_table[state]
        greedy_action= np.argmax(Q_state)
        random_action= super().step(state)

        return random_action if random.random() < self.epsilon else greedy_action

    def train(self) -> None:
        '''
        '''

        for _ in range(self.episodes):
            state= self.env.reset()
            done= False

            while not done:
                action= self.step(state= state)
                next_state, reward, done, info= self.env.step(action= action)

                Q_next= np.zeros((self.action_space.n)) if done else self.Q_table[next_state]
                self.Q_table[state, action] += self.alpha * (reward + self.gamma * np.max(Q_next) - self.Q_table[state, action])

                if done:
                    self.epsilon= self.epsilon * self.epsilon_decay

                state= next_state

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if isinstance(value, float) == False:
            raise TypeError('Float expected for Epsilon, but received {}.'.format(type(value)))
        if (value < 0.0) or (value > 1.0):
            raise ValueError('Epsilon should be positive and within the range of 0.0 & 1.0')
        self._epsilon= value

    @property
    def epsilon_decay(self):
        return self._epsilon_decay

    @epsilon_decay.setter
    def epsilon_decay(self, value):
        if isinstance(value, float) == False:
            raise TypeError('Float expected for Epsilon Decay, but received {}.'.format(type(value)))
        if value < 0.0:
            raise ValueError('Epsilon Decay should be positive')
        self._epsilon_decay= value

    @property
    def discount_factor(self):
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, value):
        if isinstance(value, float):
            raise TypeError('Float expected for the Discount Factor, but received {}.'.format(type(value)))
        if value < 0.0 or value > 1.0:
            raise ValueError('Discount Factor should be positive and within the range of 0.0 & 1.0')
        self.__discount_factor= value

    @property
    def learning_rate(self):
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        if isinstance(value, float):
            raise TypeError('Float expected for the Learning Rate, but received {}.'.format(type(value)))
        if value < 0.0:
            raise ValueError('Learning rate should be positive')
        self._learning_rate= value
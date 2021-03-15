'''
Author: @sivashanmugamo
GitHub [Home]: https://github.com/sivashanmugamo
'''

import numpy as np

from RL.environment.SimpleGrid import SimpleGrid
from RL.agents.Q_Learning_Agent import QLearningAgent

env= SimpleGrid(
    agent_position= [0, 0], 
    goal_position= [4, 4], 
    agent_value= 2, 
    goal_value= 50, 
    reward_set= {(1, 2): 45}, 
    grid_x= 5, 
    grid_y= 5, 
    stochasticity= False, 
    max_timesteps= 50
)

agent= QLearningAgent(environment= env)
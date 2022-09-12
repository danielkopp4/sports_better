from game import Game
from data_api import DataAPI
import numpy as np
from gym import spaces

class MLBGame(Game):
    def __init__(self, data_api: DataAPI):
        super(MLBGame, self).__init__(self)

        self.observation_shape = data_api.observation_space
        self.observation_space =  spaces.Box( # need to also add a value for the pot size
            low = np.zeros(self.observation_shape),
            high = np.ones(self.observation_shape),
            dtype = np.float16
        )

        self.action_space = spaces.Box(low = np.zeros(2), high = np.ones(2)) # two continuous actions

        
 
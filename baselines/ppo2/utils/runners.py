import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        # self.batch_ob_shape = (nenv*nsteps,) + env.observation_space['grid'].shape
        self.obs_grid = np.zeros((nenv,) + env.observation_space['grid'].shape, dtype=env.observation_space['grid'].dtype.name)
        self.obs_vector = np.zeros((nenv,) + env.observation_space['vector'].shape, dtype=env.observation_space['vector'].dtype.name)

        self.obs = env.reset()
        self.obs_grid[:] = self.obs['grid']
        self.obs_vector[:] = self.obs['vector']
        self.obs = dict(grid=self.obs_grid,
                        vector=self.obs_vector)
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


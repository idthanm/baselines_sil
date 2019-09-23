from baselines.common.vec_env import VecEnvWrapper
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=True):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_grid_rms = TfRunningMeanStd(shape=self.observation_space['grid'].shape, scope='ob_grid_rms') if ob else None
            self.ob_vector_rms = TfRunningMeanStd(shape=self.observation_space['vector'].shape, scope='ob_vector_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None

        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_grid_rms:
            self.ob_grid_rms.update(obs['grid'])
            self.ob_vector_rms.update(obs['vector'])
            grid = np.clip((obs['grid'] - self.ob_grid_rms.mean) / np.sqrt(self.ob_grid_rms.var + self.epsilon), -self.clipob, self.clipob)
            vector = np.clip((obs['vector'] - self.ob_vector_rms.mean) / np.sqrt(self.ob_vector_rms.var + self.epsilon), -self.clipob, self.clipob)
            return {'grid': grid, 'vector': vector}
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

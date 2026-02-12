import numpy as np
import torch    

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
            done = np.zeros((obs.shape[0], 1)).astype(bool)
        else:
            done = torch.zeros((obs.shape[0], 1), dtype=torch.bool)

        return done


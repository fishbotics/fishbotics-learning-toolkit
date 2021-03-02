import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        self._rollout_mode = False

    def set_rollout_mode(self):
        # This doesn't do anything unless its implemented
        pass

    def unset_rollout_mode(self):
        # This doesn't do anything unless its implemented
        pass

    def in_rollout_mode(self):
        return self._rollout_mode

    @staticmethod
    def transform(data):
        return data

    def optimizer(self, params=None):
        if params:
            return torch.optim.Adam(params, lr=0.001)
        return torch.optim.Adam(self.parameters(), lr=0.001)



class FishboticsSampleModel(BaseModel):

    def __init__(self):
        super().__init__()
        pass

    def set_rollout_mode(self):
        self.rollout_length = self.trajectory_length - 1
        self._eval_mode = True

    def unset_rollout_mode(self):
        self.rollout_length = self._rollout_length
        self._eval_mode = False


    def forward(self):
        pass

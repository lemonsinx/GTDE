import numpy as np
import torch


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[3:])


class SharedReplayBuffer(object):
    def __init__(self, max_steps, view_space, feature_space, gamma, n_agent):
        self.max_steps = max_steps
        self.view_buf = np.zeros((self.max_steps, n_agent, *view_space), dtype=np.float32)
        self.feature_buf = np.zeros((self.max_steps, n_agent, feature_space), dtype=np.float32)
        self.value_preds = np.zeros((self.max_steps + 1, n_agent, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        # self.returns = np.zeros((self.max_steps + 1, 64, 1), dtype=np.float32)
        self.actions = np.zeros((self.max_steps, n_agent, 1), dtype=np.int32)
        self.rewards = np.zeros((self.max_steps, n_agent, 1), dtype=np.float32)
        self.masks = np.ones((self.max_steps + 1, n_agent, 1), dtype=np.float32)
        self.gamma = gamma
        self.step = 0

    def push(self, **kwargs):
        view, feature = kwargs['state']
        ids = kwargs["ids"]
        self.view_buf[self.step][ids] = view.copy()
        self.feature_buf[self.step][ids] = feature.copy()
        # self.value_preds = np.zeros((max_steps, 64, 1), dtype=np.float32)
        self.actions[self.step][ids] = kwargs['acts'].reshape(-1, 1).copy()
        self.rewards[self.step][ids] = kwargs['rewards'].reshape(-1, 1).copy()
        self.masks[self.step + 1][ids] = kwargs['alives'].reshape(-1, 1).copy()
        self.step = (self.step + 1) % self.max_steps

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def generator(self):
        batch_size = self.rewards.shape[0]
        sampler = torch.randperm(batch_size).numpy()

        view_buf_batch = []
        feature_buf_batch = []
        returns_batch = []
        actions_batch = []

        for indices in sampler:
            view_buf_batch.append(self.view_buf[indices])
            feature_buf_batch.append(self.feature_buf[indices])
            returns_batch.append(self.returns[indices])
            actions_batch.append(self.actions[indices])

        view_buf_batch = np.stack(view_buf_batch, axis=0)
        feature_buf_batch = np.stack(feature_buf_batch, axis=0)
        returns_batch = np.stack(returns_batch, axis=0)
        actions_batch = np.stack(actions_batch, axis=0)
        return view_buf_batch, feature_buf_batch, returns_batch, actions_batch

    def reset(self):
        self.view_buf = np.zeros_like(self.view_buf)
        self.feature_buf = np.zeros_like(self.feature_buf)
        self.value_preds = np.zeros_like(self.value_preds)
        self.returns = np.zeros_like(self.returns)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.masks = np.ones_like(self.masks)


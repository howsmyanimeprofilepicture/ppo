import torch
import torch.nn as nn
import numpy as np
from typing import List
import torch.nn.functional as F
from collections import namedtuple


def discounted_cumulative_sums(x, discount_rate):
    discounted_x = 0.0
    answer = []
    for _x in x[::-1]:
        discounted_x = _x + discount_rate * discounted_x
        answer.insert(0, discounted_x)
    return np.array(answer)


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class MLP(nn.Module):
    def __init__(self, sizes: List[int], output_activation=False) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        for i in range(len(sizes) - 1):
            in_features, out_features = sizes[i : i + 2]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
        if not output_activation:
            layers.pop(-1)

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


def train_actor_net(
    actor_net: nn.Module,
    optim: torch.optim.Optimizer,
    advantage_buffer: np.ndarray,
    old_log_prob: np.ndarray,
    action_buffer: np.ndarray,
    observation_buffer: np.ndarray,
    iterations: int,
    device,
    clip_ratio: float = 0.2,
    target_kl: float = 0.01,
):
    assert isinstance(advantage_buffer, np.ndarray)
    assert isinstance(old_log_prob, np.ndarray)
    assert isinstance(action_buffer, np.ndarray)
    assert action_buffer.ndim == advantage_buffer.ndim == old_log_prob.ndim == 1
    assert observation_buffer.ndim == 2
    assert advantage_buffer.shape == old_log_prob.shape == action_buffer.shape
    assert (
        advantage_buffer.shape[0]
        == old_log_prob.shape[0]
        == action_buffer.shape[0]
        == observation_buffer.shape[0]
    )

    actor_net.train()
    advantage_buffer = torch.from_numpy(advantage_buffer).to(
        dtype=torch.float32,
        device=device,
    )
    old_log_prob = torch.from_numpy(old_log_prob).to(
        dtype=torch.float32,
        device=device,
    )
    action_buffer = torch.from_numpy(action_buffer).to(
        dtype=torch.int64,
        device=device,
    )
    observation_buffer = torch.from_numpy(observation_buffer).to(
        dtype=torch.float32,
        device=device,
    )

    for _ in range(iterations):
        new_log_prob = (
            F.log_softmax(actor_net(observation_buffer), dim=-1)
            .gather(dim=-1, index=action_buffer.reshape(-1, 1))
            .flatten()
        )

        assert new_log_prob.ndim == 1
        assert new_log_prob.size(0) == old_log_prob.size(0)

        ratio = torch.exp(new_log_prob - old_log_prob)
        policy_loss = (
            torch.min(
                ratio * advantage_buffer,
                torch.where(
                    condition=(advantage_buffer > 0),
                    input=(1 + clip_ratio) * advantage_buffer,
                    other=(1 - clip_ratio) * advantage_buffer,
                ),
            ).mean()
            * -1
        )
        optim.zero_grad()
        policy_loss.backward()
        optim.step()
        kl = torch.mean(old_log_prob - new_log_prob.detach(), axis=0)
        if kl > 1.5 * target_kl:
            print("Early Stopping !")
            return


def train_critic_net(
    critic_net: nn.Module,
    optim: torch.optim.Optimizer,
    observation_buffer: np.ndarray,
    return_buffer: np.ndarray,
    iterations: int,
    device,
):
    critic_net.train()
    observation_buffer = torch.from_numpy(observation_buffer).to(
        device=device, dtype=torch.float32
    )
    return_buffer = torch.from_numpy(return_buffer).to(
        device=device,
        dtype=torch.float32,
    )
    for _ in range(iterations):
        loss = F.mse_loss(critic_net(observation_buffer).flatten(), return_buffer)
        optim.zero_grad()
        loss.backward()
        optim.step()


@torch.no_grad()
def sample_action(observation: torch.tensor, actor_net, device):

    logits = actor_net(observation)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob_t = F.log_softmax(logits, dim=-1)[:, action]

    return namedtuple(
        "action_tuple",
        ["action", "log_prob_t"],
    )(action, log_prob_t)

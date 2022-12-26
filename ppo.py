import gym
from utils import *
import yaml
from collections import namedtuple
import torch

# Hyperparameters of the PPO algorithm
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    if config["device"].upper() == "AUTO":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    Argument = namedtuple("Argument", list(config.keys()))
    args = Argument(**config)

env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n
buffer = Buffer(observation_dimensions, args.steps_per_epoch)

print(
    "Actor Net's hidden sizes:",
    [observation_dimensions] + args.hidden_sizes + [num_actions],
)
actor_net = MLP([observation_dimensions] + args.hidden_sizes + [num_actions]).to(
    device=args.device
)
print(
    "Critic Net's hidden sizes:",
    [observation_dimensions] + args.hidden_sizes + [1],
)
critic_net = MLP([observation_dimensions] + args.hidden_sizes + [1]).to(
    device=args.device
)

actor_optim = torch.optim.Adam(
    params=actor_net.parameters(),
    lr=args.policy_learning_rate,
)

critic_optim = torch.optim.Adam(
    params=critic_net.parameters(),
    lr=args.value_function_learning_rate,
)

(observation, _), episode_return, episode_length = env.reset(), 0, 0
observation
for epoch in range(args.epochs):
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    for t in range(args.steps_per_epoch):
        observation = torch.tensor(
            np.expand_dims(observation, 0),
            dtype=torch.float32,
            device=args.device,
        )
        (action, log_prob_t) = sample_action(observation, actor_net, args.device)
        observation_new, reward, done, _, __ = env.step(action.clone().item())
        episode_return += reward
        episode_length += 1

        value_t = critic_net(observation)
        buffer.store(observation, action, reward, value_t, log_prob_t)
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == args.steps_per_epoch - 1):
            observation = torch.tensor(
                observation.reshape(1, -1),
                dtype=torch.float32,
                device=args.device,
            )
            last_value = 0 if done else critic_net(observation).clone().detach().numpy()
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            (observation, _), episode_return, episode_length = env.reset(), 0, 0
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        old_log_prob,
    ) = buffer.get()

    train_actor_net(
        actor_net,
        actor_optim,
        advantage_buffer,
        old_log_prob,
        action_buffer,
        observation_buffer,
        args.train_policy_iterations,
        device=args.device,
    )

    train_critic_net(
        critic_net,
        critic_optim,
        observation_buffer,
        return_buffer,
        args.train_value_iterations,
        device=args.device,
    )

    print(
        f"""Epoch: {epoch + 1}. 
Mean Return: {sum_return / num_episodes}. 
Mean Length: {sum_length / num_episodes}
The number of Episodes: {num_episodes}"""
    )

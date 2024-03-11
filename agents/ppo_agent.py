
# class PPOAgent:
#     Initialize:
#         - Policy network for action probability distribution
#         - Value network for value function estimation
#         - Hyperparameters: clipping value, optimizer, etc.
#
#     Train:
#         - Collect set of trajectories by running policy in environment
#         - Calculate advantages and returns
#         - Optimize policy by maximizing PPO-Clip objective function
#         - Optionally, update value network to minimize value function error
#
#     Act:
#         - Select action based on the probability distribution output by the policy network
#
#     Update Policy:
#         - Perform multiple epochs of stochastic gradient ascent on collected data to improve policy

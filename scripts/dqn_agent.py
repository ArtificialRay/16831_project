from replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Policy built on top of Vanilla DQN.
    Modified from: https://github.com/lorenmt/minimal-isaac-gym/blob/main/dqn.py
"""

# define network architecture
class Net(nn.Module):
    def __init__(self, num_obs=4, num_act=2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_act),
        )

    def forward(self, x):
        return self.net(x)


# some useful functions
def soft_update(net, net_target, tau):
    # update target network with momentum (for approximating ground-truth Q-values)
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * tau + param.data * (1.0 - tau))


class DQN:
    def __init__(self, args, env):
        self.args = args

        # initialise parameters
        self.env = env 
        self.num_envs = env.unwrapped.num_envs  # Get from environment
        self.replay = ReplayBuffer(num_envs=self.num_envs)
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.shape[0]
        self.discount = 0.99
        self.mini_batch_size = 128
        self.batch_size = self.num_envs * self.mini_batch_size
        self.tau = 0.995
        self.num_eval_freq = args.eval_interval
        self.lr = 3e-4

        self.run_step = 1
        self.score = 0

        # define Q-network
        self.q        = Net(num_obs=self.obs_space, num_act=self.act_space).to(self.args.sim_device)
        self.q_target = Net(num_obs=self.obs_space, num_act=self.act_space).to(self.args.sim_device)
        soft_update(self.q, self.q_target, tau=0.0)
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def update(self):
        # policy update using TD loss
        self.optimizer.zero_grad()

        obs, act, reward, next_obs, done_mask = self.replay.sample(self.mini_batch_size)
        q_table = self.q(obs)

        act = torch.round((0.5 * (act + 1)) * (self.act_space - 1))  # maps back to the prediction space
        q_val = q_table[torch.arange(self.batch_size), act.long()] # get Q-value for target action
        # DQN
        # with torch.no_grad():
        #     q_val_next = self.q_target(next_obs).reshape(self.batch_size, -1).max(1)[0]
        
        # Double DQN
        with torch.no_grad():
            next_act = self.q(next_obs).argmax(dim=-1)
            q_val_next = self.q_target(next_obs).gather(1, next_act.unsqueeze(-1)).squeeze(-1)
        
        target = reward + self.discount * q_val_next * done_mask
        loss = F.smooth_l1_loss(q_val, target)

        loss.backward()
        self.optimizer.step()

        # soft update target networks = smoothly update target network towards current network
        soft_update(self.q, self.q_target, self.tau)
        return loss

    def act(self, obs, epsilon=0.0):
        # epsilon greedy: step grows larger, less likely to explore
        coin = torch.rand(self.num_envs, device=self.args.sim_device) < epsilon

        rand_act = torch.rand(self.num_envs, device=self.args.sim_device)
        with torch.no_grad():
            q_table = self.q(obs)
            true_act = torch.cat([(q_table[b] == q_table[b].max()).nonzero(as_tuple=False)[0] # get index of the first largest q in q table
                                 for b in range(self.num_envs)])
            true_act = true_act / (self.act_space - 1) # normalize to [0,1]

        act = coin.float() * rand_act + (1 - coin.float()) * true_act
        return 2 * (act - 0.5)  # maps to -1 to 1

    def run(self):
        epsilon = max(0.01, 0.8 - 0.01 * (self.run_step / 20))

        # collect data using standard Gymnasium API
        if not hasattr(self, 'obs'):
            # First time: initialize environment
            self.obs, _ = self.env.reset()
        
        obs = self.obs
        action = self.act(obs, epsilon)
        
        # Execute action and get transition tuple
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        
        # Store transition in replay buffer
        self.replay.push(obs, action, reward, next_obs, 1 - done.float()) # storing done mask
        
        # Update current observation
        self.obs = next_obs
        
        # training mode
        loss = None
        if self.replay.size() > self.mini_batch_size:
            loss = self.update()
            self.score += torch.mean(reward.float()).item() / self.num_eval_freq

        self.run_step += 1
        return loss, epsilon
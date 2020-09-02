import numpy as np
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import sample, random
from dataclasses import dataclass
from collections import deque
from typing import Any
from tqdm import tqdm
import wandb
from postgres_executor import postgres_executor
import gym_dgame
import argparse 
import argh

# Parse the commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=0)
# Workload parameters
parser.add_argument('--backend', type=int, default=0, help='database to use: {0=postgres, 1=sqlserver}')
parser.add_argument('--workload_size', type=int, default=5, help='size of workload')
parser.add_argument('--index_limit', type=int, default=3, help='maximum number of index that can be created')
parser.add_argument('--tpch_queries', type=int, nargs='+', default=[4, 5, 6],
                    help='tpc-h queries used to create workload', choices=[1, 3, 4, 5, 6, 13, 14])

# Model Hyperparameter
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--hidden_layers', type=int, default=4, help='number of hidden layers')
parser.add_argument('--hidden_units', type=int, default=8, help='number of hidden units in each hidden layer')
parser.add_argument('--agent', default='cem', choices=['cem', 'dqn', 'naf', 'ddpg', 'sarsa'],
                    help='rl agent used for learning')
parser.add_argument('--activation_function', default='relu', choices=['elu', 'relu', 'selu', 'tanh'],
                    help='activation function used in hidden layers')
parser.add_argument('--memory_limit', type=int, default=500, help='episode memory size')
parser.add_argument('--steps_warmup', type=int, default=100, help='number of warmup steps')
parser.add_argument('--nb_steps_test', type=int, default=5, help='number of steps in test')
parser.add_argument('--elite_frac', type=float, default=0.005, help='elite fraction used in rl model')
parser.add_argument('--nb_steps_train', type=int, default=1000, help='number of steps in training')

# Database parameters
parser.add_argument('--host', default='/tmp', help='hostname of database server')
parser.add_argument('--database', default='indexselection_tpch___1', help='database name')
parser.add_argument('--port', default='5109', help='database port')
parser.add_argument('--user', default='matrix', help='database username')
parser.add_argument('--password', default='', help='database password')
# other parameters
parser.add_argument('--hypo', type=int, default=1)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--sf', type=int, default=1)
parser.add_argument('--wandb_flag', type=int, default=0)
parser.add_argument('--env_steps_before_train', type=int, default=1000)
parser.add_argument('--min_rb_size', type=int, default=1000)
parser.add_argument('--sample_size', type=int, default=50)
parser.add_argument('--tgt_model_update', type=int, default=50)

args = parser.parse_args()

postgres_config = {
'host': args.host,
'database': args.database,
'port': args.port,
'user': args.user,
'password': args.password
}


@dataclass
class Trajactory:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class ReplayBuffer:
    def __init__(self, buffer_size = 100000):
        self.buffer_size = buffer_size
#        self.buffer = deque(maxlen=buffer_size)
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, traj):
        self.buffer[self.idx % self.buffer_size] = traj
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)

def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        # obs shape (N, 4)
        q_vals = self.model(observations)

        # q_vals (N, 2)
        return q_vals.max(-1)[1]

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        self.obs_shape = obs_shape
        assert len(obs_shape) == 1, "this nn only works for flat observation"
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions)
        )
        self.opt= optim.Adam(self.net.parameters(), lr = 0.0001)
    
    def forward(self, x):
        return self.net(x)

def train_step(model, state_transitions, tgt, num_actions, device, gamma = 0.99):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))

    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
    actions = [s.action for s in state_transitions]
    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0] # (N, nub_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)
    
    # import ipdb; ipdb.set_trace()
    loss = ((rewards + mask[:,0]*qvals_next - torch.sum(qvals*one_hot_actions, -1)) ** 2).mean()
    # loss = ((rewards + mask[:,0]*qvals_next - torch.sum(qvals*one_hot_actions, -1))).mean()
    # loss_fn = nn.SmoothL1Loss()
    # loss = loss_fn(
    #     torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:,0]*qvals_next*gamma
    #     )
    loss.backward()
    model.opt.step()
    return loss

def main(name, test=False, chkpt=None, device='cpu'):
    workload_size = args.workload_size
    database = postgres_executor.TPCHExecutor(postgres_config, args.hypo)
    database.connect()
    # Enable hypothetical indexes
    if args.hypo:
        database.execute('create extension if not exists hypopg')

    database._connection.commit()
    ENV_NAME = 'dgame-v0'
    env = gym.make(ENV_NAME)

    env.initialize(database, workload_size, args.index_limit, 1, verbose=args.verbose)

    eps_min                = 0.01
    eps_decay              = 0.999995
    min_rb_size            = args.min_rb_size            
    sample_size            = args.sample_size            
    env_steps_before_train = args.env_steps_before_train 
    tgt_model_update       = args.tgt_model_update       
    
    
    if not test:
        wandb.init(project= "autoindex-torch", name=name)
        wandb.config.update({
            "min_rb_size"            : min_rb_size,
            "sample_size"            : sample_size,
            "env_steps_before_train" : env_steps_before_train,
            "tgt_model_update"       : tgt_model_update,
            })
    

    last_observation = env.reset()
  # import ipdb; ipdb.set_trace()
    m = Model(env.observation_space.shape, env.action_space.n)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = Model(env.observation_space.shape, env.action_space.n)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0
    step_num =  -1 * min_rb_size
    # qvals = m(torch.Tensor(last_observation))
    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while step_num < 1000000:
            tq.update(1)
            eps = eps_decay ** (step_num)
            if test:
                eps = 0
            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_observation)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward 

            rb.insert(Trajactory(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
#                wandb.log({'episode_reward':rolling_reward}, step=step_num)
                rolling_reward = 0
                last_observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if (not test) and rb.idx  > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, 'cpu')
                wandb.log({'loss':loss.detach().item(), 'eps':eps, 'avg_reward':
                    np.mean(episode_rewards)}, step= step_num)

                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print("update tgt model")
                    update_tgt_model(m, tgt)
                    episode_rewards = []
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass 

if __name__ == "__main__":
    main('dqn_db_test')

import gym
import argparse
import gym_dgame

from postgres_executor import postgres_executor
from keras.optimizers import Adam
from rl.agents import CEMAgent, DQNAgent, DDPGAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.policy import BoltzmannQPolicy
from rl.callbacks import WandbLogger
from rl.memory import EpisodeParameterMemory, SequentialMemory
# from sqlserver_executor import sqlserver_executor
import wandb
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG as dqn_default_conf
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG as ppo_default_conf
import shutil
import pandas as pd
import bokeh
import json
from util.line_plots import plot_line, plot_line_with_min_max, plot_line_with_stddev

# Parse the commandline arguments
parser = argparse.ArgumentParser()

parser.add_argument('--verbose', type=int, default=2)
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
parser.add_argument('--host', default='localhost', help='hostname of database server')
parser.add_argument('--database', default='indexselection_tpch___1', help='database name')
parser.add_argument('--port', default='5432', help='database port')
parser.add_argument('--user', default='postgres', help='database username')
parser.add_argument('--password', default='', help='database password')
# other parameters
parser.add_argument('--hypo', type=int, default=1)
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--sf', type=int, default=10)
parser.add_argument('--wandb_flag', type=int, default=0)
parser.add_argument('--env_version', type = int, default = 0)
parser.add_argument('--ray_flag', type = bool , default = True)

args = parser.parse_args()

# keras-rl agents
AGENT_DIC = {
        'cem': CEMAgent,
        'dqn': DQNAgent,
        'ddpg': DDPGAgent,
        }

# from random import randrange as rand
if args.user == 'postgres':
    postgres_config = {
            'database': args.database,
            }
else:
    postgres_config = {
            'host': args.host,
            'database': args.database,
            'port': args.port,
            'user': args.user,
            'password': args.password
            }


def gym_env_creator(env_config):
    ENV_NAME = env_config['env_name']
    from gym_dgame.envs.dgame_env import DatabaseGameEnv as env
    database = postgres_executor.TPCHExecutor(postgres_config, args.hypo)

    database.connect()
    if args.hypo:
        database.execute('create extension if not exists hypopg')
    database._connection.commit()

    env_inst = env(env_config)
    env_inst.initialize(database, args.workload_size, args.index_limit, 1, verbose=args.verbose)
    
    return env_inst

    
def create_keras_rl_model(v = 0):
    """create dqn model"""
    if v == 0:
        model = Sequential()
        model.add(Flatten(input_shape=(args.batch_size, observation_n)))
        # Complex Deep NN Model
        for i in range(args.hidden_layers):
            model.add(Dense(args.hidden_units))
            model.add(Activation(args.activation_function))
        model.add(Dense(nb_actions))
        model.add(Activation('softmax'))
        print(model.summary())
    return model


def test_model():
    pass


def test_rllib():
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    # config = ppo_default_conf.copy()
    config = dqn_default_conf.copy()
    config['num_workers'] = 1
    # config['num_sgd_iter'] = 30
    # config['sgd_minibatch_size'] = 128
    config['model']['fcnet_hiddens'] = [100, 100]
    # config['num_cpus_per_worker'] = 0
    # agent = PPOTrainer(config, 'CartPole-v1')
    agent = DQNTrainer(config, 'CartPole-v1')
    # import ipdb; ipdb.set_trace()
    print("rllib testing end")


if __name__ == '__main__':
   # test_rllib()
    env_name = 'dgame-v0'
    ray_tune_flag = True
    # ray_tune_flag = False
    if args.ray_flag:
        ray.init(ignore_reinit_error=True)
        register_env('ray_env', gym_env_creator)
        config = dqn_default_conf.copy()
        config['env'] = 'ray_env'
        config['env_config'] = {"env_name":"dgame-v0", "workload_size": args.workload_size}
        config['num_workers'] = 1
        config['model']['fcnet_activation'] = 'relu'
        if not ray_tune_flag:
            # trainer0 = DQNTrainer(config, 'CartPole-v1')
            config['train_batch_size'] = 50
            config['gamma'] = 0.9
            config['lr'] = 0.001
            config['model']['fcnet_hiddens'] = [4, 8]
            trainer = DQNTrainer(config = config)

            ckp_root = 'tmp/dqn/it'
            shutil.rmtree(ckp_root, ignore_errors=True, onerror=None)
            ray_results='ray_results/'
            shutil.rmtree(ray_results, ignore_errors=True, onerror = None)

            n_iter = 10
            s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

            results = []
            episode_data = []
            episode_json = []
            result_strs = []

            for n in range(n_iter):
                result = trainer.train()
                file_name = trainer.save(ckp_root)
                results.append(result)
                episode = {'n': n,
                       'episode_reward_min':  result['episode_reward_min'],
                       'episode_reward_mean': result['episode_reward_mean'],
                       'episode_reward_max':  result['episode_reward_max'],
                       'episode_len_mean':    result['episode_len_mean']
                       }
                episode_data.append(episode)
                episode_json.append(json.dumps(episode))
                result_strs.append(
                    f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
                # results.append(f'{n+1} reward {result["episode_reward_mean"]:.3f}/'
                #                f'{result["episode_reward_min"]:.3f}/{result["episode_reward_max"]:.3f}'
                #                # f'len {result["episode_len_mean"]} saved {file_name}'
                #                )
            df = pd.DataFrame(data=episode_data)
            for i in range(len(result_strs)):
                print(result_strs[i])
            bokeh.io.reset_output()
            bokeh.io.output_notebook()
            plot_line_with_min_max(df, x_col='n', y_col='episode_reward_mean', min_col='episode_reward_min', max_col='episode_reward_max',
                              title='Episode Rewards', x_axis_label='n', y_axis_label='reward')
        else:

            config['train_batch_size'] = 50
            config['gamma'] = tune.grid_search([0.8, 0.9, 0.95, 0.99])
            config['lr'] = tune.grid_search([0.01, 0.001, 0.0001])
            config['model']['fcnet_hiddens'] = [4, 8]
            analysis = tune.run("DQN",
                        config=config,
                        stop={"training_iteration": 10})
            best_conf = analysis.get_best_config(metric="episode_reward_mean")
            # gamma.0 9, lr 0.01
            print("testing end")

    else:
        ENV_NAME = 'dgame-v0'
        env = gym_env_creator({})
        workload_size = args.workload_size
        nb_actions = env.action_space.n
        observation_n = env.observation_space.n
        # create a model
        Agent = AGENT_DIC[args.agent]
        model = create_keras_rl_model()
        # set up rl agent
        if args.agent == 'dqn':
            memory = SequentialMemory(limit=args.memory_limit, window_length=args.batch_size)
            policy = BoltzmannQPolicy()
            agent = DQNAgent(
                model = model, nb_actions=nb_actions, memory=memory, batch_size=args.batch_size,
                nb_steps_warmup = args.steps_warmup, target_model_update=1e-2,
                policy=policy)
            agent.compile(Adam(lr=1e-3), metrics=['mae'])
        elif args.agent == 'cem':
            model = create_keras_rl_model()        
            memory = EpisodeParameterMemory(limit=args.memory_limit, window_length=args.batch_size)
            agent = Agent(
                    model=model, nb_actions=nb_actions,
                    memory=memory, batch_size=args.batch_size,
                    nb_steps_warmup=args.steps_warmup, train_interval=1,
                    elite_frac=args.elite_frac)
            agent.compile()
        # train or test
        if args.train == 1:
            if not args.wandb_flag:
                # import ipdb; ipdb.set_trace()
                pass
            else:
                pass
            agent.save_weights(f'cem_{ENV_NAME}_sf{args.sf}_{args.hypo}_params.h5', overwrite=True)
        elif args.train == 0:
            agent.load_weights(f'cem_{ENV_NAME}_sf{args.sf}_{args.hypo}_params.h5')
        elif args.train == -1:
            agent.load_weights('cem_index_selection_evaluation.h5')
        env.train = False
        # env.database.hypo=False
        test_model()

import gym
import neptune
import tensorflow as tf

from spinup import ppo_tf1 as ppo


def env_fn(): return gym.make('LunarLander-v2')


ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

neptune.init(project_qualified_name='<namespace/project_name>')
experiment = neptune.create_experiment(name='Spinning Up example')

logger_kwargs = dict(output_dir='./out',
                     exp_name='neptune_logging',
                     neptune_experiment=experiment)

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000,
    epochs=250, logger_kwargs=logger_kwargs)

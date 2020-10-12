"""Script for running RLlib with the Neptune logger."""

import ray
from ray import tune

from logger import NeptuneLogger


ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "logger_config": {"neptune_project_name": '<namespace/project_name>'},
    },
    loggers=tune.logger.DEFAULT_LOGGERS + (NeptuneLogger,),
)

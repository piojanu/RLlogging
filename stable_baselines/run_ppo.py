"""Running Stable Baselines PPO training with Neptune logging."""
import neptune
import stable_baselines  # This is required!
from stable_baselines import logger

from logger import NeptuneLogger

neptune.init('<namespace/project_name>')
experiment = neptune.create_experiment(name='Stable Baselines example')

logger_ = logger.Logger.CURRENT
logger_.output_formats.append(NeptuneLogger(experiment))

model = stable_baselines.PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
model.learn(total_timesteps=10000)

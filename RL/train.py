import torch
from algo import Trainer
import pybullet_envs
import pybullet
import gym_donkeycar
import gym
from SAC import SAC
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
port = 9091
conf = {"exe_path": exe_path, "port": port}

env = gym.make("donkey-generated-track-v0", conf=conf)
# env_test = gym.make("donkey-generated-track-v0", conf=conf)

print("obs shape {}".format(env.observation_space.shape))
print("obs shape {}".format(*env.observation_space.shape))
print("acts shape {}".format(env.action_space.shape))
print("acts shape {}".format(*env.action_space.shape))


# ENV_ID = 'Pendulum-v0'
SEED = 0
REWARD_SCALE = 1.0
NUM_STEPS = 5 * 10 ** 4
EVAL_INTERVAL = 10 ** 3

# env = gym.make(ENV_ID)
# env_test = gym.make(ENV_ID)

algo = SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    seed=SEED,
    reward_scale=REWARD_SCALE,
)

trainer = Trainer(
    env=env,
    # env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()
# trainer.plot()
trainer.visualize()
# trainer.visualize()

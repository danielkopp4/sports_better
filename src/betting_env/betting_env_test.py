from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np

from environment import BettingEnvironment
bias_factor = 0
g_0 = 1
e_length = 100

if __name__ == "__main__":
    env = BettingEnvironment(bias_factor, g_0, e_length)
    env = make_vec_env(lambda: env, n_envs=1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=30, log_interval=4)
    env = model.get_env()
    print('finished')
    obs = env.reset()
    input()
    while True:
        action, _states = model.predict(obs)
        # obs, rewards, dones, info = env.step(action)
        obs, rewards, dones, info = env.step(np.array([0,0,0]))
        if dones:
            input()
        env.render('console')

    # obs = env.reset()
    # env.render()

    # print(obs)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    # # Hardcoded best agent: always go left!
    # n_steps = 1000
    # for step in range(n_steps):
    #     print("Step {}".format(step + 1))
    #     obs, reward, done, info = env.step(env.action_space.sample())
    #     # obs, reward, done, info = env.step(np.array([1,1,-1]))
    #     print('obs=', obs, 'reward=', reward, 'done=', done)
    #     env.render()
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         break

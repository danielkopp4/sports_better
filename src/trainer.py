import logging

logging.basicConfig(
    filename="logs/training.log",
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO
)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG, SAC, TD3, A2C, PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.data_api import HistoricalBettingDataAPI
from src.sports_betting_odds_env import SportsBettingOddsEnv, redistribute



train_test_split = 0.8
initial_amount = 1
test_trials = 1000
episode_length = 1
eval_epsisode_length = 300
fps = 120
training_steps = (10 - 4) * 60 * 60 * fps  # should equate to about a half day, and 23,040 episodes
log_interval = 80 * fps

def evaluate_betting_odds_model(model, test_env: SportsBettingOddsEnv, trials: int):
    env = test_env
    rewards = []
    for _ in range(trials):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        rewards.append(env.pot)

    obs = env.reset()

    return np.mean(rewards), np.std(rewards)

def pause():
    try:
        return input()
    except:
        print()
        exit()

def parse_scalar(inpt):
    try:
        return float(inpt)
    except:
        return None

def train_sports_betting_odds_env(betting_odds_api: HistoricalBettingDataAPI):
    logging.info("starting training on raw sports betting odds...")

    betting_odds_api.shuffle()

    split_index = int(train_test_split * len(betting_odds_api))
    betting_odds_training = betting_odds_api[:split_index]
    betting_odds_testing = betting_odds_api[split_index:]

    env = SportsBettingOddsEnv(initial_amount, betting_odds_training, episode_length=episode_length)

    # be able to split into training and testing
    env = make_vec_env(lambda: env, n_envs=1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model = TD3(
        "MlpPolicy", 
        env,
        action_noise=action_noise,
        # gamma=0,
        # batch_size=512,
        verbose=1
    )

    # model = SAC("MlpPolicy", env, verbose=1)
    # model = A2C("MlpPolicy", env, verbose=1)
    # model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=training_steps, log_interval=log_interval)
    model.save("betting_model_{}".format(np.random.randint(0, 100000)))

    # env = model.get_env()
    env = SportsBettingOddsEnv(1, betting_odds_testing, episode_length=eval_epsisode_length)
    logging.info('finished')
    obs = env.reset()

    logging.info('starting eval...')
    logging.info('eval results mu=%.5e std=%.5e', *evaluate_betting_odds_model(model, env, test_trials))
    logging.info('press enter...')

    scalar = parse_scalar(pause())

    while True:
        env.render('console')

        action, _states = model.predict(obs)
        action = redistribute(action, scalar)
        obs, _rewards, dones, _info = env.step(action)

        if dones:
            env.render('console')
            scalar = parse_scalar(pause())
            env.reset()



if __name__ == "__main__":
    from src.NBA.data_api import NBAHistoricalBettingAPI
    api = NBAHistoricalBettingAPI()
    train_sports_betting_odds_env(api)

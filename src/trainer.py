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
from src.sports_betting_odds_env import SportsBettingOddsEnv, MultiGameSimulation



train_test_split = 0.8
initial_amount = 1
test_trials_short = 1000
test_trials_long = 100
eval_episode_length = 300
fps = 150
training_steps = (4 + 12 + 2) * 60 * 60 * fps  # should equate to about a half day, and 23,040 episodes
log_interval = 80 * fps

def evaluate_betting_odds_model(model, test_env: SportsBettingOddsEnv, scalar=None):
    sim = MultiGameSimulation(model, test_env)
    logging.info("starting short episode...")
    short_trials = sim.run_sims(1, test_trials_short, scalar=scalar)
    logging.info("mu=%.5e std=%.5e sem=%.5e", np.mean(short_trials), np.std(short_trials), np.std(short_trials)/np.sqrt(test_trials_short))
    logging.info("starting long episode...")
    long_trials = sim.run_sims(eval_episode_length, test_trials_long, scalar=scalar)
    logging.info("mu=%.5e std=%.5e sem=%.5e", np.mean(long_trials), np.std(long_trials),  np.std(long_trials)/np.sqrt(test_trials_long))




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

    env = SportsBettingOddsEnv(initial_amount, betting_odds_training)

    env = make_vec_env(lambda: env, n_envs=1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy", 
        env,
        action_noise=action_noise,
        gamma=0,
        batch_size=512,
        verbose=1
    )

    model.learn(total_timesteps=training_steps, log_interval=log_interval)
    model.save("betting_model_{}".format(np.random.randint(0, 100000)))

    env = SportsBettingOddsEnv(1, betting_odds_testing)
    logging.info('finished')

    logging.info('starting eval...')

    evaluate_betting_odds_model(model, env)

    logging.info('press enter...')
    logging.info('enter scalar or 0 to skip')


    while True:
        scalar = parse_scalar(pause())
        if scalar == 0:
            break

        logging.info("*"*30)
        logging.info("using scalar of %s", str(scalar))

        evaluate_betting_odds_model(model, env, scalar)

    sim = MultiGameSimulation(model, env)
    logging.info("enter scalar or enter 0 to exit...")

    while True:
        scalar = parse_scalar(pause())
        if scalar == 0:
            break

        logging.info("*"*30)
        logging.info("using scalar of %s", str(scalar))

        sim.run_sims(eval_episode_length, 1, scalar, verbose=1)
    

if __name__ == "__main__":
    from src.NBA.data_api import NBAHistoricalBettingAPI
    api = NBAHistoricalBettingAPI()
    train_sports_betting_odds_env(api)

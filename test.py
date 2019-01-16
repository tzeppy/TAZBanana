
import argparse
import logging
from unityagents import UnityEnvironment
# import numpy as np

from q_agent import BananaAgent

log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='banana_agent1')
    parser.add_argument('-e', '--episodes', default=5, type=int)
    parser.add_argument('--no_graphics', action='store_true', default=False)
    args = parser.parse_args()
    #
    env = UnityEnvironment(file_name="Banana.app", no_graphics=args.no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    action = 0
    state_size = len(state)
    #

    b_agent = BananaAgent(args.model_name, state_size, action_size)
    b_agent.load()

    for epx in range(1, args.episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        b_agent.reset_episode()
        while True:
            action = b_agent.act(state, use_egreedy=False)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            b_agent.sense(state, action, reward, next_state, learn=False)
            if done:
                break
            state = next_state
        print("{},{}".format(epx, b_agent.cum_rewards()))

    log.info("finished.")

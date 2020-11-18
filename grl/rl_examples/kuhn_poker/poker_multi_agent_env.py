import random
import copy

import numpy as np
from gym.spaces import Discrete, Box, Dict
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from open_spiel.python.rl_environment import TimeStep, Environment

def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config

# game versions
KUHN_POKER = 'kuhn_poker'
LEDUC_POKER = 'leduc_poker'
PHANTOM_TICTACTOE = 'phantom_ttt'
MATRIX_RPS = 'matrix_rps'

PARTIALLY_OBSERVABLE = "partially_observable"
DEFAULT_CONFIG = {
    'version': KUHN_POKER,
    'fixed_players': False
}

OBS_SHAPES = {
    KUHN_POKER: (11,),
    LEDUC_POKER: (30,),
    PHANTOM_TICTACTOE: (214,),
}

VALID_ACTIONS_SHAPES = {
    KUHN_POKER: (2,),
    LEDUC_POKER: (3,),
    PHANTOM_TICTACTOE: (9,),
}

POKER_ENV = 'poker_env'

PARTIAL_OBSERVATION = 'partial_observation'
VALID_ACTIONS_MASK = 'valid_actions_mask'


class PokerMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})
        self._fixed_players = env_config['fixed_players']
        self.game_version = env_config['version']

        if self.game_version in [KUHN_POKER, LEDUC_POKER]:
            open_spiel_env_config = {
                "players": 2
            }
        else:
            open_spiel_env_config = {}

        self.openspiel_env = Environment(game_name=self.game_version, discount=1.0,
                                         **open_spiel_env_config)

        self.action_space = Discrete(self.openspiel_env.action_spec()["num_actions"])
        observation_length = OBS_SHAPES[self.game_version][0] + VALID_ACTIONS_SHAPES[self.game_version][0]
        self.observation_space = Box(low=0.0, high=1.0, shape=(observation_length,))

        self.curr_time_step: TimeStep = None
        self.player_map = None

    def _get_current_obs(self):

        done = self.curr_time_step.last()
        obs = {}
        if done:
            player_ids = [0, 1]
        else:
            curr_player_id = self.curr_time_step.observations["current_player"]
            player_ids = [curr_player_id]

        for player_id in player_ids:
            legal_actions = self.curr_time_step.observations["legal_actions"][player_id]
            legal_actions_mask = np.zeros(self.openspiel_env.action_spec()["num_actions"])
            legal_actions_mask[legal_actions] = 1.0

            info_state = self.curr_time_step.observations["info_state"][player_id]

            # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
            # Taking an illegal action will result in a random legal action being played.
            # Allows easy compatibility with standard RL implementations for small action-space games like this one.
            obs[self.player_map(player_id)] = np.concatenate(
                (np.asarray(info_state, dtype=np.float32), np.asarray(legal_actions_mask, dtype=np.float32)),
                axis=0)

        return obs

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.curr_time_step = self.openspiel_env.reset()

        # swap player mapping in half of the games
        if self._fixed_players:
            self.player_map = lambda p: p
        else:
            self.player_map = random.choice((lambda p: p,
                                             lambda p: (1 - p)))

        return self._get_current_obs()

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        curr_player_id = self.curr_time_step.observations["current_player"]
        legal_actions = self.curr_time_step.observations["legal_actions"][curr_player_id]

        player_action = action_dict[self.player_map(curr_player_id)]

        # If action is illegal, do a random legal action instead
        if player_action not in legal_actions:
            player_action = random.choice(legal_actions)

        self.curr_time_step = self.openspiel_env.step([player_action])

        new_curr_player_id = self.curr_time_step.observations["current_player"]
        obs = self._get_current_obs()
        done = self.curr_time_step.last()
        dones = {self.player_map(new_curr_player_id): done, "__all__": done}

        if done:
            rewards = {self.player_map(0): self.curr_time_step.rewards[0],
                       self.player_map(1): self.curr_time_step.rewards[1]}

            infos = {0: {}, 1: {}}

            infos[self.player_map(0)]['game_result_was_invalid'] = False
            infos[self.player_map(1)]['game_result_was_invalid'] = False

            assert sum(self.curr_time_step.rewards) == 0.0, "curr_time_step rewards in are terminal state are {} (they should sum to zero)".format(self.curr_time_step.rewards)

            infos[self.player_map(0)]['rewards'] = self.curr_time_step.rewards[0]
            infos[self.player_map(1)]['rewards'] = self.curr_time_step.rewards[1]

            if self.curr_time_step.rewards[0] > 0:
                infos[self.player_map(0)]['game_result'] = 'won'
                infos[self.player_map(1)]['game_result'] = 'lost'
            elif self.curr_time_step.rewards[1] > 0:
                infos[self.player_map(1)]['game_result'] = 'won'
                infos[self.player_map(0)]['game_result'] = 'lost'
            else:
                infos[self.player_map(1)]['game_result'] = 'tied'
                infos[self.player_map(0)]['game_result'] = 'tied'
        else:
            assert self.curr_time_step.rewards[new_curr_player_id] == 0, "curr_time_step rewards in non terminal state are {}".format(self.curr_time_step.rewards)
            assert self.curr_time_step.rewards[-(new_curr_player_id-1)] == 0

            rewards = {self.player_map(new_curr_player_id): self.curr_time_step.rewards[new_curr_player_id]}

            infos = {}

        return obs, rewards, dones, infos


register_env(POKER_ENV, lambda env_config: PokerMultiAgentEnv(env_config))
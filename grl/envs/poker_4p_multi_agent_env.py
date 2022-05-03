import copy
import random

import numpy as np
from gym.spaces import Discrete, Box
from open_spiel.python.rl_environment import TimeStep, Environment, StepType
from pyspiel import SpielError
from ray.tune.registry import register_env

from grl.envs.valid_actions_multi_agent_env import ValidActionsMultiAgentEnv


def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


# game versions
KUHN_POKER = 'kuhn_poker'
LEDUC_POKER = 'leduc_poker'

DEFAULT_CONFIG = {
    'version': KUHN_POKER,
    'fixed_players': True,
    'dummy_action_multiplier': 1,
    'continuous_action_space': False,
    'penalty_for_invalid_actions': False,
    'append_valid_actions_mask_to_obs': True,
}

OBS_SHAPES = {
    KUHN_POKER: (23 + 2,),
    LEDUC_POKER: (64 + 3,),
}

VALID_ACTIONS_SHAPES = {
    KUHN_POKER: (2,),
    LEDUC_POKER: (3,),
}

POKER_4p_ENV = 'poker_4p_env'

PARTIAL_OBSERVATION = 'partial_observation'
VALID_ACTIONS_MASK = 'valid_actions_mask'


def parse_discrete_poker_action_from_continuous_space(continuous_action, legal_actions_list,
                                                      total_num_discrete_actions_including_dummy):
    assert isinstance(continuous_action, (float, np.floating)) or \
           isinstance(continuous_action, np.ndarray) and isinstance(continuous_action[0], np.floating), \
        f"action {continuous_action} is a {type(continuous_action)}. " \
        f"If the action is an int, something is likely wrong with the continuous policy output."

    # player action is between -1 and 1, normalize to 0 and 1 and then quantize to a discrete action
    player_action = (np.clip(continuous_action, a_min=-1.0, a_max=1.0) + 1.0) / 2.0
    assert 0.0 - 1e-9 <= player_action <= 1.0 + 1e-9, f"action was: {player_action} before normalization: {continuous_action}"
    # place discrete actions in [0, 1] and find closest corresponding discrete action to player action
    nearest_legal_discrete_action = min(legal_actions_list,
                                        key=lambda x: abs(
                                            x / (total_num_discrete_actions_including_dummy - 1) - player_action))
    # player action is now a discrete action
    return nearest_legal_discrete_action


class Poker4pMultiAgentEnv(ValidActionsMultiAgentEnv): # Note: this is a team game env! Team0: player0 + player 2; Team1: player1 + player3

    def __init__(self, env_config=None):
        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})
        self._fixed_players = env_config['fixed_players']
        self.game_version = env_config['version']

        if not isinstance(env_config['dummy_action_multiplier'], int) and env_config['dummy_action_multiplier'] > 0:
            raise ValueError("dummy_action_multiplier must be a positive non-zero int")
        self.dummy_action_multiplier = env_config['dummy_action_multiplier']
        self._continuous_action_space = env_config['continuous_action_space']

        self._individual_players_with_continuous_action_space = env_config.get("individual_players_with_continuous_action_space")
        self._individual_players_with_orig_obs_space = env_config.get("individual_players_with_orig_obs_space")

        self._apply_penalty_for_invalid_actions = env_config["penalty_for_invalid_actions"]
        self._invalid_action_penalties = [False, False]

        self._append_valid_actions_mask_to_obs = env_config["append_valid_actions_mask_to_obs"]
        self._stack_size = None

        self.open_spiel_env_config = {
                "players": 4
            }

        self.openspiel_env = Environment(game=self.game_version, discount=1.0,
                                         **self.open_spiel_env_config)

        self.base_num_discrete_actions = self.openspiel_env.action_spec()["num_actions"]
        self.num_discrete_actions = int(self.base_num_discrete_actions * self.dummy_action_multiplier)
        self._base_action_space = Discrete(self.base_num_discrete_actions)

        if self._continuous_action_space:
            self.action_space = Box(low=-1, high=1, shape=(1,))
        else:
            self.action_space = Discrete(self.num_discrete_actions)

        self.orig_observation_length = self.openspiel_env.observation_spec()["info_state"][0]

        if self._append_valid_actions_mask_to_obs:
            self.observation_length = self.orig_observation_length + self.base_num_discrete_actions
        else:
            self.observation_length = self.orig_observation_length

        self.observation_space = Box(low=0.0, high=1.0, shape=(self.observation_length,))

        self.curr_time_step: TimeStep = None
        self.player_map = None

    def _get_current_obs(self):

        done = self.curr_time_step.last()
        obs = {}
        player_ids = [0, 1, 2, 3]

        # if done:
        #     player_ids = [0, 1, 2, 3]
        # else:
        #     curr_player_id = self.curr_time_step.observations["current_player"]
        #     player_ids = [curr_player_id]

        for player_id in player_ids:
            legal_actions = self.curr_time_step.observations["legal_actions"][player_id]
            legal_actions_mask = np.zeros(self.openspiel_env.action_spec()["num_actions"])
            legal_actions_mask[legal_actions] = 1.0

            info_state = self.curr_time_step.observations["info_state"][player_id]

            force_orig_obs = self._individual_players_with_orig_obs_space is not None and player_id in self._individual_players_with_orig_obs_space

            if self._append_valid_actions_mask_to_obs and not force_orig_obs:
                obs[self.player_map(player_id)] = np.concatenate(
                    (np.asarray(info_state, dtype=np.float32), np.asarray(legal_actions_mask, dtype=np.float32)),
                    axis=0)
            else:
                obs[self.player_map(player_id)] = np.asarray(info_state, dtype=np.float32)

        return obs

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.curr_time_step = self.openspiel_env.reset()

        if self._fixed_players:
            self.player_map = lambda p: p
        else:
            # swap player mapping in half of the games
            self.player_map = random.choice((lambda p: p,
                                             lambda p: (p + 1) % 4,
                                             lambda p: (p + 2) % 4,
                                             lambda p: (p + 3) % 4)) # TODO: double check

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
        orig_player_action = player_action

        if self._continuous_action_space or \
                (self._individual_players_with_continuous_action_space and curr_player_id in self._individual_players_with_continuous_action_space):
            player_action = parse_discrete_poker_action_from_continuous_space(
                continuous_action=player_action, legal_actions_list=legal_actions,
                total_num_discrete_actions_including_dummy=self.num_discrete_actions)

        if self.dummy_action_multiplier != 1:
            # extended dummy action space is just the base discrete actions repeated multiple times
            # convert to the base discrete action space.
            player_action = player_action % self.base_num_discrete_actions

        if player_action not in self._base_action_space:
            raise ValueError("Processed player action isn't in the base action space.\n"
                             f"orig action: {orig_player_action}\n"
                             f"processed action: {player_action}\n"
                             f"action space: {self.action_space}\n"
                             f"base action space: {self._base_action_space}")

        if player_action not in legal_actions:
            legal_actions_mask = np.zeros(self.openspiel_env.action_spec()["num_actions"])
            legal_actions_mask[legal_actions] = 1.0
            raise ValueError(f"illegal actions are not allowed.\n"
                             f"Action was {player_action}.\n"
                             f"Legal actions are {legal_actions}\n"
                             f"Legal actions vector is {legal_actions_mask}")
        try:
            self.curr_time_step = self.openspiel_env.step([player_action])
        except SpielError:
            raise

        new_curr_player_id = self.curr_time_step.observations["current_player"]
        obs = self._get_current_obs()
        done = self.curr_time_step.last()

        dones = {self.player_map(new_curr_player_id): done, "__all__": done}

        if True:
            #### TODO: need to sum the reward and assert zero-sum between teams
            team0_reward = self.curr_time_step.rewards[0] + self.curr_time_step.rewards[2]
            team1_reward = self.curr_time_step.rewards[1] + self.curr_time_step.rewards[3]
            team_rewards = [team0_reward, team1_reward, team0_reward, team1_reward]

            rewards = {self.player_map(i): team_rewards[i] for i in range(4)}
            # assert self.curr_time_step.rewards[0] == -self.curr_time_step.rewards[1]
            infos = {i: {} for i in range(4)}

            for i in range(4):
                infos[self.player_map(i)]['game_result_was_invalid'] = False
                infos[self.player_map(i)]['rewards'] = team_rewards[i]

            assert sum(
                team_rewards) == 0.0, "curr_time_step rewards in are terminal state are {} (they should sum to zero)".format(
                self.curr_time_step.rewards)

            if team_rewards[0] > 0:
                infos[self.player_map(0)]['game_result'] = 'won'
                infos[self.player_map(1)]['game_result'] = 'lost'
                infos[self.player_map(2)]['game_result'] = 'won'
                infos[self.player_map(3)]['game_result'] = 'lost'
            elif team_rewards[1] > 0:
                infos[self.player_map(1)]['game_result'] = 'won'
                infos[self.player_map(0)]['game_result'] = 'lost'
                infos[self.player_map(3)]['game_result'] = 'won'
                infos[self.player_map(2)]['game_result'] = 'lost'
            else:
                infos[self.player_map(0)]['game_result'] = 'tied'
                infos[self.player_map(1)]['game_result'] = 'tied'
                infos[self.player_map(2)]['game_result'] = 'tied'
                infos[self.player_map(3)]['game_result'] = 'tied'

        else:
            assert self.curr_time_step.rewards[
                       new_curr_player_id] == 0, "curr_time_step rewards in non terminal state are {}".format(
                self.curr_time_step.rewards)
            assert self.curr_time_step.rewards[-(new_curr_player_id - 1)] == 0

            rewards = {self.player_map(new_curr_player_id): self.curr_time_step.rewards[new_curr_player_id]}
            assert self.curr_time_step.rewards[1 - new_curr_player_id] == 0.0
            infos = {}

        if self._apply_penalty_for_invalid_actions:
            for player_id, penalty in enumerate(self._invalid_action_penalties):
                if penalty and self.player_map(player_id) in rewards:
                    rewards[self.player_map(player_id)] -= 4.0
                    self._invalid_action_penalties[player_id] = False

        return obs, rewards, dones, infos


register_env(POKER_4p_ENV, lambda env_config: Poker4pMultiAgentEnv(env_config))

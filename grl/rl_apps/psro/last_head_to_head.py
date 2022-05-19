import logging
import os
import time
from typing import Dict, Type, List

import deepdish
import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.models import MODEL_DEFAULTS

from ray.rllib.utils import merge_dicts
from tables.exceptions import HDF5ExtError
import argparse
import re
import traceback 
import sys
import pyspiel
from itertools import product

from grl.rl_apps.psro.general_psro_eval import run_episode_team
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.envs.poker_4p_multi_agent_env  import Poker4pMultiAgentEnv

from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.psro.poker_utils import openspiel_policy_from_nonlstm_rllib_policy
from grl.rl_apps.scenarios.trainer_configs.poker_4p_psro_configs import *
from grl.rl_apps.tiny_bridge_4p_mappo import CCTrainer_4P, CCPPOTorchPolicy_4P

from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir
from grl.utils.strategy_spec import StrategySpec
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.tiny_bridge_4p_mappo import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rl_apps.scenarios.trainer_configs.bridge_psro_configs import *
from grl.algos.p2sro.payoff_table import PayoffTable
from grl.algos.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critic_model import TorchCentralizedCriticModel

import pickle
import matplotlib.pyplot as plt


def load_metanash_pure_strat(policy: Policy, pure_strat_spec: StrategySpec):
    pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.p2sro_policy_spec = pure_strat_spec


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Sample new pure strategy policy weights from the metanash of the subgame population for the best response to
        # train against. For better runtime performance, consider loading new weights only every few episodes instead.
        metanash_policy: Policy = policies[f"metanash"]
        metanash_policy_specs = worker.metanash_policy_specs
        metanash_weights = worker.metanash_weights

        new_pure_strat_spec: StrategySpec = np.random.choice(a=metanash_policy_specs, p=metanash_weights)
        # noinspection PyTypeChecker
        load_metanash_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)

def select_policy(agent_id):
    return ValueError
    if agent_id % 2 == team:
        return "best_response"
    elif agent_id % 2 == other_team:
        return "metanash"
    else:
        raise ValueError(f"Unknown agent id: {agent_id}")

def get_policy(scenario, env):
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": scenario.env_class,
        "env_config": scenario.env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (
                policy_classes["metanash"], env.observation_space, env.action_space, {"explore": False}),
                f"best_response": (
                policy_classes["best_response"], env.observation_space, env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    trainer_config = merge_dicts(trainer_config, scenario.get_trainer_config(env))
    policy = scenario.policy_classes["metanash"](
            obs_space=env.observation_space,
            action_space=env.action_space,
            config=trainer_config)
    return policy


def get_policy_from_scenario(scenario_name):
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)
    if not isinstance(scenario, PSROScenario):
        raise TypeError(f"Only instances of {PSROScenario} can be used here. {scenario.name} is a {type(scenario)}.")
    get_trainer_config = scenario.get_trainer_config

    env_class = scenario.env_class
    env_config = scenario.env_config
    trainer_class = scenario.trainer_class
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    tmp_env = env_class(env_config=env_config)

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (
                policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space, {"explore": False}),
                f"best_response": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))
    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(
                                base_dir="./junk", scenario_name=scenario_name,
                                should_log_result_fn=True))

    policy = trainer.workers.local_worker().policy_map["metanash"]
    return policy


def get_population_distribution_for_player(player, payoff_table):
    population_meta_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=payoff_table,
        as_player=1-player,  # get strategies for opponent to 1, so player 0's metanash mixed strat
        as_policy_num=payoff_table.shape()[0],  # get whole payoff table
        fictitious_play_iters=int(os.getenv("FSP_ITERS", 2000)),
        mix_with_uniform_dist_coeff=0.0,
        print_matrix=False
    )

    # this is a mixed strategy below, we sample StrategySpecs from it
    policy_distribution: PolicySpecDistribution = population_meta_strategies[player]  # player zero strategy
    return policy_distribution.probabilities_for_each_strategy(), policy_distribution

def find_timestep_mapping(psro_path, sp_path):
    psro_mapping = psro_iter2timestep[psro_path]
    sp_mapping = sp_iter2timestep[sp_path]
    mapping = {k: None for k in psro_mapping.keys()}
    for psro_itr, psro_ts in psro_mapping.items():
        for sp_itr, sp_ts in sp_mapping.items():
            if sp_ts >= psro_ts:
                mapping[psro_itr] = sp_itr
                break
    mapping = {k: v for k, v in mapping.items() if v is not None} # only return mapped ckpts
    return mapping

def get_all_psro_specs_with_prob(ckpt, psro_seed_path):
    payoff_table_json_path = os.path.join(psro_seed_path, "payoff_table_checkpoints", f"payoff_table_checkpoint_{ckpt}.json")
    payoff_table = PayoffTable.from_json_file(payoff_table_json_path)
    pop_distribution0, pop_metanash_strat0 = get_population_distribution_for_player(0, payoff_table)
    pop_distribution1, pop_metanash_strat1 = get_population_distribution_for_player(1, payoff_table)
    list_of_strategy_specs_team_0: List[StrategySpec] = payoff_table.get_ordered_spec_list_for_player(player=0)
    list_of_strategy_specs_team_1: List[StrategySpec] = payoff_table.get_ordered_spec_list_for_player(player=1)
    all_specs_with_prob = []
    for i in range(len(pop_distribution0)):
        all_specs_with_prob.append({'specs': [list_of_strategy_specs_team_0[i], list_of_strategy_specs_team_1[i]],\
                                     'weights': [pop_distribution0[i], pop_distribution1[i]]})
    return all_specs_with_prob, [pop_metanash_strat0, pop_metanash_strat1]

def get_all_sp_specs(sp_iter, sp_seed_path):
    all_specs = [] 
    player_0_br_iter_json_path = os.path.join(sp_seed_path, "br_policy_checkpoint_specs", f"best_response_player_0_iter_{sp_iter}.json") # use 0 since it's the same for player 0 and player 1
    player_0_sp_spec = StrategySpec.from_json_file(player_0_br_iter_json_path)
    player_1_br_iter_json_path = os.path.join(sp_seed_path, "br_policy_checkpoint_specs", f"best_response_player_1_iter_{sp_iter}.json") # use 0 since it's the same for player 0 and player 1
    player_1_sp_spec = StrategySpec.from_json_file(player_1_br_iter_json_path)
    all_specs.append({'specs': [player_0_sp_spec, player_1_sp_spec]})
    return all_specs # should only have 1 element!


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario_name", type=str, required=True)
    parser.add_argument("-r", "--random_policy", action="store_true")
    parser.add_argument("-n", "--num_games", type=int, default=5)
    args = parser.parse_args()


    scenario_name = args.scenario_name
    num_games = args.num_games
    if scenario_name == 'tiny_bridge_4p_psro':
        psro_path = "/home/gaoyue/nxdo/grl/data/3_seeds_psro_tiny_bridge"
        sp_path = "/home/gaoyue/nxdo/grl/data/3_seeds_self_play_tiny_bridge"
    elif scenario_name == 'bridge_psro':
        psro_path = "/home/gaoyue/nxdo/grl/data/2_seeds_psro_bridge"
        sp_path = "/home/gaoyue/nxdo/grl/data/2_seeds_self_play_bridge"
    else:
        raise NotImplementedError

    output_dir = f'./last_head_to_head_{scenario_name}_n{num_games}'
    if args.random_policy:
        output_dir += '_rand' 
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    odd_ckpts = list(np.array(range(0, 300, 2))+1) # Note: should > number of ckpts we have!
    sp_iters = [500 * i for i in range(1000)] # Note: should > number of iters we have!
    sp_iters[0] += 1 # the first checkpoint is 1 instead of 0
    psro_seeds_paths = []
    sp_seeds_paths = []
    for seed in os.listdir(psro_path):
        psro_seeds_paths.append(os.path.join(psro_path, seed))
    for seed in os.listdir(sp_path):
        sp_seeds_paths.append(os.path.join(sp_path, seed))

    psro_iter2timestep = {}
    sp_iter2timestep = {}
    for psro_seed_path in psro_seeds_paths:
        psro_iter2timestep[psro_seed_path] = {}
        for ckpt in odd_ckpts:
            payoff_table_json_path = os.path.join(psro_seed_path, "payoff_table_checkpoints", f"payoff_table_checkpoint_{ckpt}.json")
            if os.path.isfile(payoff_table_json_path):
                payoff_table = PayoffTable.from_json_file(payoff_table_json_path)
                cur_timesteps = sum(spec.metadata["timesteps_training_br"] for spec in payoff_table.get_ordered_spec_list_for_player(0))
                cur_timesteps += sum(spec.metadata["timesteps_training_br"] for spec in payoff_table.get_ordered_spec_list_for_player(1))
                psro_iter2timestep[psro_seed_path][ckpt] = cur_timesteps

    for sp_seed_path in sp_seeds_paths:
        sp_iter2timestep[sp_seed_path] = {}
        for sp_iter in sp_iters:
            br_iter_json_path = os.path.join(sp_seed_path, "br_policy_checkpoint_specs", f"best_response_player_0_iter_{sp_iter}.json") # use 0 since it's the same for player 0 and player 1
            if os.path.isfile(br_iter_json_path):
                sp_spec = StrategySpec.from_json_file(br_iter_json_path)
                sp_iter2timestep[sp_seed_path][sp_iter] = sp_spec.metadata['timesteps_training'] 

    results = [] # should be len(psro_seeds_paths) * len(sp_seeds_paths)
    r = 0
    for psro_seed_path, sp_seed_path in product(psro_seeds_paths, sp_seeds_paths):
        ckpt_mapping = find_timestep_mapping(psro_seed_path, sp_seed_path)
        print(f"@@@@@@@@@@@ max ckpt: {sorted(ckpt_mapping.keys())[-1]}")
        ts_info = {}
        ckpt = sorted(ckpt_mapping.keys())[-1]
        sp_iter = ckpt_mapping[ckpt]
        psro_ts = psro_iter2timestep[psro_seed_path][ckpt]
        sp_ts = sp_iter2timestep[sp_seed_path][sp_iter]
        ts_info[ckpt] = [psro_ts, sp_ts, sp_iter]
        print(f"psro_timestep: {psro_ts}   sp_timestep: {sp_ts}")
        psro_specs_with_prob, psro_metanash_distibutions = get_all_psro_specs_with_prob(ckpt, psro_seed_path)
        sp_specs = get_all_sp_specs(sp_iter, sp_seed_path)
        seed_results = []  # len == 2 * num_games
        for psro_team, sp_team in [[0, 1], [1, 0]]:
            sampled_psro_spec = psro_metanash_distibutions[psro_team].sample_policy_spec()
            sp_spec = sp_specs[0]
            psro_rewards = []
            for i in range(num_games):
                scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)
                env = scenario.env_class(env_config=scenario.env_config)

                policies_for_each_player = [get_policy(scenario, env), get_policy(scenario, env)]
                load_metanash_pure_strat(policies_for_each_player[psro_team], pure_strat_spec=sampled_psro_spec)
                if not args.random_policy:
                    load_metanash_pure_strat(policies_for_each_player[sp_team], pure_strat_spec=sp_spec['specs'][sp_team])

                payoffs_per_team_this_episode = run_episode_team(env,  [policies_for_each_player[0], \
                                                                        policies_for_each_player[1], \
                                                                        policies_for_each_player[0], \
                                                                        policies_for_each_player[1]])
                psro_rewards.append(payoffs_per_team_this_episode[psro_team])
            print(np.mean(psro_rewards))
            seed_results.extend(psro_rewards) # preserve the full result
        assert len(seed_results) == 2 * num_games
        # with open(os.path.join(output_dir, f'combo{r}_data.txt'), 'w') as text_file:
        #     np.savetxt(text_file, seed_results)
        with open(os.path.join(output_dir, f'combo{r}_data.pkl'), 'wb') as fp:
            pickle.dump(seed_results, fp)
        with open(os.path.join(output_dir, f'combo{r}_timestep_info.pkl'), 'wb') as fp:
            pickle.dump(ts_info, fp)
        with open(os.path.join(output_dir, f'combo{r}_seed_info.txt'), 'w') as text_file:
            text_file.write(f"psro: {psro_seed_path}   sp: {sp_seed_path} \n")
        results.append(seed_results)
        r += 1

    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as fp:
        pickle.dump(results, fp)
    
    flat_results = []
    for row in results:
        flat_results.extend(row)
    
    print(f"#### mean: {np.mean(flat_results)}   std: {np.std(flat_results)}")

    # with open(os.path.join(output_dir, f'{scenario_name}_head_to_head.pkl'), 'wb') as fp:
    #     pickle.dump(results, fp)
        
    import pdb; pdb.set_trace()


        


import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

import os
import time
import logging
import random
import tempfile
from gym.spaces import Discrete

from typing import Dict, List, Type, Callable
import deepdish

from ray.rllib.agents import Trainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from grl.utils.strategy_spec import StrategySpec
from grl.utils.common import pretty_dict_str, datetime_str
from grl.algos.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.algos.xfdo.restricted_game import RestrictedGame
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.algos.xfdo.opnsl_restricted_game import OpenSpielRestrictedGame, get_restricted_game_obs_conversions

logger = logging.getLogger(__name__)


def get_trainer_logger_creator(base_dir: str, scenario_name: str):
    logdir_prefix = f"{scenario_name}_sparse_{datetime_str()}"

    def trainer_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=base_dir)

        def _should_log(result: dict) -> bool:
            return result["training_iteration"] % 100 == 0

        return SpaceSavingLogger(config=config, logdir=logdir, should_log_result_fn=_should_log, print_log_dir=False)

    return trainer_logger_creator


def load_pure_strat(policy: Policy, pure_strat_spec, checkpoint_path: str = None):
    assert pure_strat_spec is None or checkpoint_path is None, "can only pass one or the other"
    if checkpoint_path is None:
        if hasattr(policy, "policy_spec") and pure_strat_spec == policy.policy_spec:
            return
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    else:
        pure_strat_checkpoint_path = checkpoint_path

    pure_strat_checkpoint_path = pure_strat_checkpoint_path.replace(
        "/home/jb/git/grl/grl/data/12_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive/manager_01.04.42AM_Feb-03-2021/",
        "/home/jblanier/gokuleduc/xfdo/manager_01.04.42AM_Feb-03-2021/")

    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec


def create_get_pure_strat_cached(cache: dict):
    def load_pure_strat_cached(policy: Policy, pure_strat_spec):

        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
        pure_strat_checkpoint_path = pure_strat_checkpoint_path.replace(
            "/home/jb/git/grl/grl/data/12_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive/manager_01.04.42AM_Feb-03-2021/",
            "/home/jblanier/gokuleduc/xfdo/manager_01.04.42AM_Feb-03-2021/")

        if pure_strat_checkpoint_path in cache:
            weights = cache[pure_strat_checkpoint_path]
        else:
            checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
            weights = checkpoint_data["weights"]
            weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
            cache[pure_strat_checkpoint_path] = weights

        policy.set_weights(weights=weights)
        policy.policy_spec = pure_strat_spec

    return load_pure_strat_cached


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                 env_index=env_index, **kwargs)
        avg_policy: Policy = worker.policy_map["metanash"]
        if not hasattr(avg_policy, "cfp_br_specs") or not avg_policy.cfp_br_specs:
            return
        new_br_spec = random.choice(avg_policy.cfp_br_specs)
        if hasattr(avg_policy, "policy_spec") and new_br_spec == avg_policy.policy_spec:
            return
        load_pure_strat(policy=avg_policy, pure_strat_spec=new_br_spec)


def set_restricted_game_conversations_for_all_workers(
        trainer: Trainer,
        delegate_policy_id: PolicyID,
        agent_id_to_restricted_game_specs: Dict[AgentID, List[StrategySpec]],
        load_policy_spec_fn):
    def _set_conversions(worker: RolloutWorker):

        def _set_restricted_env_convertions(restricted_env):
            assert isinstance(restricted_env, RestrictedGame)
            for agent_id, action_policy_specs in agent_id_to_restricted_game_specs.items():
                if len(action_policy_specs) > 0:
                    convertor = RestrictedToBaseGameActionSpaceConverter(
                        delegate_policy=worker.policy_map[delegate_policy_id],
                        policy_specs=action_policy_specs,
                        load_policy_spec_fn=load_policy_spec_fn)
                    restricted_env.set_action_conversion(agent_id=agent_id, converter=convertor)

        worker.foreach_env(_set_restricted_env_convertions)

    trainer.workers.foreach_worker(_set_conversions)


def set_restricted_game_conversions_for_all_workers_openspiel(
        trainer: Trainer,
        tmp_base_env: MultiAgentEnv,
        delegate_policy_id: PolicyID,
        agent_id_to_restricted_game_specs: Dict[AgentID, List[StrategySpec]],
        load_policy_spec_fn):
    local_delegate_policy = trainer.workers.local_worker().policy_map[delegate_policy_id]
    player_converters = {}
    for p, restricted_game_specs in agent_id_to_restricted_game_specs.items():
        if len(restricted_game_specs) == 0:
            continue
        player_converters[p] = (get_restricted_game_obs_conversions(player=p, delegate_policy=local_delegate_policy,
                                                                    policy_specs=restricted_game_specs,
                                                                    load_policy_spec_fn=load_policy_spec_fn,
                                                                    tmp_base_env=tmp_base_env))
    assert len(player_converters) == 0 or len(player_converters) == 1

    def _set_worker_converters(worker: RolloutWorker):
        worker_delegate_policy = worker.policy_map[delegate_policy_id]
        for p, player_converter in player_converters.items():
            worker.foreach_env(lambda env: env.set_obs_conversion_dict(p, player_converter))
        worker_delegate_policy.player_converters = player_converters

    trainer.workers.foreach_worker(_set_worker_converters)
    trainer.get_local_converters = lambda: trainer.workers.local_worker().policy_map[
        delegate_policy_id].player_converters


def train_poker_approx_best_response_xdfo(br_player: int,
                                          ray_head_address,
                                          scenario,
                                          general_trainer_config_overrrides,
                                          br_policy_config_overrides: dict,
                                          get_stopping_condition: Callable[[], StoppingCondition],
                                          metanash_specs_for_players: Dict[int, StrategySpec],
                                          delegate_specs_for_players: Dict[int, List[StrategySpec]],
                                          results_dir: str,
                                          print_train_results: bool = True):
    use_openspiel_restricted_game: bool = scenario.use_openspiel_restricted_game
    get_restricted_game_custom_model = scenario.get_restricted_game_custom_model

    env_class = scenario.env_class
    base_env_config = scenario.env_config
    trainer_class = scenario.trainer_class_br
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes_br
    get_trainer_config = scenario.get_trainer_config_br
    xfdo_metanash_method: str = scenario.xdo_metanash_method
    use_cfp_metanash = (xfdo_metanash_method == "cfp")

    if metanash_specs_for_players is not None and use_cfp_metanash:
        cfp_metanash_specs_for_players = {}
        for p, player_metanash_spec in metanash_specs_for_players.items():
            player_cfp_json_specs = player_metanash_spec.metadata["cfp_pure_strat_specs"]
            player_cfp_specs = [StrategySpec.from_json(json_spec) for json_spec in player_cfp_json_specs]
            cfp_metanash_specs_for_players[p] = player_cfp_specs
    else:
        cfp_metanash_specs_for_players = None

    other_player = 1 - br_player
    br_learner_name = f"approx br player {br_player}"

    def log(message, level=logging.INFO):
        logger.log(level, f"({br_learner_name}): {message}")

    def select_policy(agent_id):
        if agent_id == br_player:
            return f"best_response"
        elif agent_id == other_player:
            return f"metanash"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    restricted_env_config = {
        "create_env_fn": lambda: env_class(env_config=base_env_config),
        "raise_if_no_restricted_players": metanash_specs_for_players is not None
    }
    tmp_base_env = env_class(env_config=base_env_config)

    if use_openspiel_restricted_game:
        restricted_game_class = OpenSpielRestrictedGame
    else:
        restricted_game_class = RestrictedGame

    tmp_env = restricted_game_class(env_config=restricted_env_config)

    if metanash_specs_for_players is None or use_openspiel_restricted_game:
        other_player_restricted_action_space = tmp_env.base_action_space
    else:
        other_player_restricted_action_space = Discrete(n=len(delegate_specs_for_players[other_player]))

    if metanash_specs_for_players is None and use_openspiel_restricted_game:
        other_player_restricted_obs_space = tmp_env.base_observation_space
    else:
        other_player_restricted_obs_space = tmp_env.observation_space

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": restricted_game_class,
        "env_config": restricted_env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (
                policy_classes["metanash"], other_player_restricted_obs_space, other_player_restricted_action_space,
                {"explore": False}),
                f"metanash_delegate": (
                policy_classes["best_response"], tmp_env.base_observation_space, tmp_env.base_action_space,
                {"explore": False}),
                f"best_response": (
                policy_classes["best_response"], tmp_env.base_observation_space, tmp_env.base_action_space,
                br_policy_config_overrides),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    if metanash_specs_for_players is not None:
        if get_restricted_game_custom_model is not None:
            restricted_game_custom_model = get_restricted_game_custom_model(tmp_base_env)
        else:
            restricted_game_custom_model = None

        trainer_config["multiagent"]["policies"]["metanash"][3]["model"] = {
            "custom_model": restricted_game_custom_model}

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_base_env))

    trainer_config = merge_dicts(trainer_config, general_trainer_config_overrrides)

    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(base_dir=results_dir, scenario_name="approx_br"))

    if use_cfp_metanash and cfp_metanash_specs_for_players:
        # metanash is uniform distribution of pure strat specs
        def _set_worker_metanash_cfp_specs(worker: RolloutWorker):
            worker.policy_map["metanash"].cfp_br_specs = cfp_metanash_specs_for_players[other_player]

        trainer.workers.foreach_worker(_set_worker_metanash_cfp_specs)
    elif not use_cfp_metanash:
        # metanash is single pure strat spec
        def _set_worker_metanash(worker: RolloutWorker):
            if metanash_specs_for_players is not None:
                metanash_policy = worker.policy_map["metanash"]
                load_pure_strat(policy=metanash_policy, pure_strat_spec=metanash_specs_for_players[other_player])

        trainer.workers.foreach_worker(_set_worker_metanash)

    trainer.weights_cache = {}
    if delegate_specs_for_players:
        if use_openspiel_restricted_game:
            set_restricted_game_conversions_for_all_workers_openspiel(trainer=trainer,
                                                                      tmp_base_env=tmp_base_env,
                                                                      delegate_policy_id="metanash_delegate",
                                                                      agent_id_to_restricted_game_specs={
                                                                          other_player: delegate_specs_for_players[
                                                                              other_player]},
                                                                      load_policy_spec_fn=load_pure_strat)
        else:
            set_restricted_game_conversations_for_all_workers(trainer=trainer, delegate_policy_id="metanash_delegate",
                                                              agent_id_to_restricted_game_specs={
                                                                  other_player: delegate_specs_for_players[
                                                                      other_player]},
                                                              load_policy_spec_fn=create_get_pure_strat_cached(
                                                                  cache=trainer.weights_cache))

    # Perform main RL training loop. Stop if we reach max iters or saturate.
    # Saturation is determined by checking if we improve by a minimum amount every n iters.
    train_iter_count = 0

    stopping_condition: StoppingCondition = get_stopping_condition()

    max_reward = None
    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["best_response_player"] = br_player
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"][f"best_response"]:
                del train_iter_results["info"]["learner"][f"best_response"]["td_error"]
            log(f"Trainer log dir is {trainer.logdir}")
            print(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]
        if max_reward is None or br_reward_this_iter > max_reward:
            max_reward = br_reward_this_iter
        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            log("Stopping condition met.")
            break

    log(f"Training stopped.")

    # trainer.cleanup()
    # del trainer
    ray.shutdown()
    time.sleep(10)

    return max_reward

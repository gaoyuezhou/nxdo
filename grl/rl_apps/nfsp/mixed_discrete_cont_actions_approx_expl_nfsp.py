import logging
import os
from typing import Type, Dict

import deepdish
from ray.rllib.agents import Trainer
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils import merge_dicts, try_import_torch

from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import pretty_dict_str
from grl.utils.strategy_spec import StrategySpec

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def load_pure_strat(policy: Policy, pure_strat_spec, checkpoint_path: str = None):
    assert pure_strat_spec is None or checkpoint_path is None, "can only pass one or the other"
    if checkpoint_path is None:
        if hasattr(policy, "policy_spec") and pure_strat_spec == policy.policy_spec:
            return
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    else:
        pure_strat_checkpoint_path = checkpoint_path

    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "avg_policy_checkpoints")


def spec_checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "avg_policy_checkpoint_specs")


def train_poker_approx_best_response_nfsp(br_player,
                                          ray_head_address,
                                          scenario,
                                          general_trainer_config_overrrides,
                                          br_policy_config_overrides,
                                          get_stopping_condition,
                                          avg_policy_specs_for_players: Dict[int, StrategySpec],
                                          results_dir: str,
                                          trainer_class_override=None,
                                          br_policy_class_override=None,
                                          print_train_results: bool = True):
    env_class = scenario.env_class
    env_config = scenario.env_config

    other_player = 1 - br_player
    env_config["discrete_actions_for_players"] = [other_player]

    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes

    if br_policy_class_override is not None:
        policy_classes["best_response"] = br_policy_class_override

    get_trainer_config = scenario.get_trainer_config
    should_log_result_fn = scenario.ray_should_log_result_filter

    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def select_policy(agent_id):
        if agent_id == br_player:
            return "best_response"
        else:
            return f"average_policy"

    tmp_env = env_class(env_config=env_config)

    all_discrete_action_env_config = env_config.copy()
    all_discrete_action_env_config["discrete_actions_for_players"] = [0, 1]
    all_discrete_action_tmp_env = env_class(env_config)

    avg_policy_model_config = get_trainer_config(all_discrete_action_tmp_env)["model"]

    from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
    from grl.rl_apps.scenarios.trainer_configs.loss_game_configs import loss_game_psro_ppo_params

    br_trainer_config = {
        "log_level": "INFO",
        # "callbacks": None,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0.0,
        "num_workers": 0,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "average_policy": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.discrete_action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response": (PPOTorchPolicy, tmp_env.observation_space, tmp_env.continuous_action_space,
                                  {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    # br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(tmp_env))
    br_trainer_config = merge_dicts(br_trainer_config, loss_game_psro_ppo_params(tmp_env))

    br_trainer = PPOTrainer(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                         scenario_name="approx_br",
                                                                         should_log_result_fn=should_log_result_fn))

    def _set_avg_policy(worker: RolloutWorker):
        avg_policy = worker.policy_map["average_policy"]
        load_pure_strat(policy=avg_policy, pure_strat_spec=avg_policy_specs_for_players[1 - br_player])

    br_trainer.workers.foreach_worker(_set_avg_policy)

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    stopping_condition: StoppingCondition = get_stopping_condition()

    max_reward = None
    while True:
        train_iter_results = br_trainer.train()  # do a step (or several) in the main RL loop
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]

        if max_reward is None or br_reward_this_iter > max_reward:
            max_reward = br_reward_this_iter

        train_iter_count += 1
        if print_train_results:
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
                del train_iter_results["info"]["learner"]["best_response"]["td_error"]
            print(pretty_dict_str(train_iter_results))
            log(f"Trainer logdir is {br_trainer.logdir}")

        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            print("stopping condition met.")
            break

    return max_reward, None

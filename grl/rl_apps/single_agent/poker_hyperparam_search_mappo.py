import logging
import os
from typing import Dict

import ray
from ray.rllib import BaseEnv
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import PolicyID
from ray.rllib.models import MODEL_DEFAULTS
from ray import tune
from ray.tune import choice, loguniform
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.dqn import DQNTrainer
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched
from grl.utils.strategy_spec import StrategySpec
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.utils.common import data_dir
from grl.envs.poker_4p_multi_agent_env import Poker4pMultiAgentEnv
from grl.envs.goofspiel_4p_multi_agent_env import Goofspiel4pMultiAgentEnv
from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from ray.tune.suggest.hyperopt import HyperOptSearch
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env

from grl.rl_apps.tiny_bridge_2p_mappo import CCTrainer
from gym.spaces import Discrete

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # game_version = 'kuhn_poker'
    # game_version = 'leduc_poker'
    game_version = 'goofspiel'

    experiment_name = f"{game_version}_hyperparam_search_mappo"
    num_cpus = 60
    num_gpus = 0
    env_class = Poker4pMultiAgentEnv
    if game_version == 'goofspiel':
        env_class = Goofspiel4pMultiAgentEnv

    br_team = 1
    avg_policy_team = 1 - br_team

    env_config = {
        "version": game_version,
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    }

    avg_pol_scenario: NFSPScenario = scenario_catalog.get(scenario_name="leduc_nfsp_dqn")

    trainer_class = CCTrainer

    tmp_env = env_class(env_config=env_config)

    address_info = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        object_store_memory=int(1073741824 * 1),
        local_mode=False,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=find_free_port(),
        ignore_reinit_error=True,
        logging_level=logging.INFO,
        log_to_driver=os.getenv("RAY_LOG_TO_DRIVER", False))


    def select_policy(agent_id):
        if agent_id % 2 == br_team:
            return "best_response"
        else:
            return f"average_policy"


    avg_policy_model_config = avg_pol_scenario.get_avg_trainer_config(tmp_env)["model"]

    # player_0_avg_pol_spec = StrategySpec.from_json_file(
    #     "/home/jblanier/git/grl/grl/data/leduc_nfsp_dqn_sparse_02.34.06PM_Apr-08-2021bt5ym0l8/avg_policy_checkpoint_specs/average_policy_player_0_iter_263000.json"
    # )


    class HyperParamSearchCallbacks(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)
            if not hasattr(worker, "avg_pol_loaded") or not worker.avg_pol_loaded:
                # avg_policy = worker.policy_map["average_policy"]
                # load_pure_strat(policy=avg_policy, pure_strat_spec=player_0_avg_pol_spec)
                worker.avg_pol_loaded = True

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            result["br_reward_mean"] = result["policy_reward_mean"]["best_response"]

    hyperparams = {
        "framework": "torch",
        "callbacks": HyperParamSearchCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "average_policy": (
                    avg_pol_scenario.policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space,
                    {
                        "model": avg_policy_model_config,
                        "explore": False,
                    }),
                # "best_response": (SimpleQTorchPolicyPatched, tmp_env.observation_space, tmp_env.action_space, {
                #     "model": merge_dicts(MODEL_DEFAULTS, {
                #         "fcnet_activation": "relu",
                #         "fcnet_hiddens": [128],
                #         "custom_model": get_valid_action_fcn_class_for_env(env=tmp_env),
                #     }),
                # }),
                "best_response": (None, tmp_env.observation_space, tmp_env.action_space, {
                    "framework": "torch",
                }),
            },
            "policy_mapping_fn": select_policy,
        },

        "vf_share_layers": False,
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 1,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 5000,

        # Coefficient of the entropy regularizer.
        "entropy_coeff": choice([0.0, 0.001, 0.0001]),
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,

        "model": {
            "custom_model": "cc_model",
            "vf_share_layers": False ### overriding the default parameter
        },

        # "model": choice([
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128, 128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128, 128, 128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        # ]),
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 256,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": choice([2048, 4096]),
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": choice([128, 256, 512, 1024]),
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": choice([1, 5, 10, 30, 60]),
        # Stepsize of SGD.
        "lr": choice([5e-2, 5e-3, 5e-4, 5e-5, 5e-6]),
        # PPO clip parameter.
        "clip_param": choice([0.0, 0.03, 0.3]),
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": choice([0.001, 0.01, 0.1]),
    }


    search = HyperOptSearch(metric="br_reward_mean", mode="max", n_initial_points=20)

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",

             config=hyperparams,
             num_samples=1, #### is 2 for ppo in oshi_zumo
             search_alg=search, #### can be None?
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(3e6)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )


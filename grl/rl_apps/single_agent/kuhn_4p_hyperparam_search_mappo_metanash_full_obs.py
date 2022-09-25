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
# from grl.rl_apps.tiny_bridge_4p_multi_agent_env import TinyBridge4pMultiAgentEnv
from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.envs.poker_4p_multi_agent_env import Poker4pMultiAgentEnv
from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from ray.tune.suggest.hyperopt import HyperOptSearch
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.algos.p2sro.payoff_table import PayoffTable
from grl.algos.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critic_model_full_obs import TorchCentralizedCriticModelFullObs

from grl.rl_apps.tiny_bridge_2p_mappo import CCTrainer
from grl.rl_apps.tiny_bridge_4p_mappo_full_obs import CCTrainer_4P_full_obs
from gym.spaces import Discrete
from ray.rllib.models import ModelCatalog

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ModelCatalog.register_custom_model(
        "cc_model_full_obs", TorchCentralizedCriticModelFullObs)

    experiment_name = f"kuhn_4p_hyperparam_search_mappo_ctrainer_4p_full_obs"
    num_cpus = 24
    num_gpus = 0
    env_class = Poker4pMultiAgentEnv

    br_team = 1
    avg_policy_team = 1 - br_team

    env_config = {
        "version": "kuhn_poker",
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    }

    metanash_pol_scenario = scenario_catalog.get(scenario_name="kuhn_4p_psro")

    # trainer_class = CCTrainer
    trainer_class = CCTrainer_4P_full_obs


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
            return f"metanash"


    metanash_policy_model_config = metanash_pol_scenario.get_trainer_config(tmp_env)["model"]

    payoff_table_path = "/home/gaoyue/nxdo/grl/data/full_obs/manager_11.18.48AM_Jun-07-2022/payoff_table_checkpoints/payoff_table_checkpoint_latest.json"
    payoff_table = PayoffTable.from_json_file(json_file_path=payoff_table_path)
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=payoff_table,
        as_player=br_team,
        as_policy_num=None,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=0.0
    )
    opponent_policy_distribution: PolicySpecDistribution = latest_strategies[1 - br_team]


    class HyperParamSearchCallbacks(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)
            metanash_policy = worker.policy_map["metanash"]
            load_pure_strat(policy=metanash_policy,
                            checkpoint_path=opponent_policy_distribution.sample_policy_spec().metadata[
                                "checkpoint_path"].replace("deploy", "jblanier"))

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
                "metanash": (
                    metanash_pol_scenario.policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space,
                    {
                        "model": metanash_policy_model_config,
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
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 5000,

        # Coefficient of the entropy regularizer.
        "entropy_coeff": choice([0.0, 0.001, 0.0001]),
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,

        "model": {
            "custom_model": "cc_model_full_obs",
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
        "lr": choice([5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 1e-7]),
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
             num_samples=-1,
             search_alg=search,
             mode="max",
             # progress_reporter=CLIReporter(max_report_frequency=60 * 10, max_progress_rows=5),
             local_dir=data_dir(),
             stop={"timesteps_total": int(3e7)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )

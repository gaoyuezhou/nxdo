from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rl_apps.scenarios.trainer_configs.bridge_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

from grl.envs.tiny_bridge_2p_multi_agent_env  import TinyBridge2pMultiAgentEnv
from grl.rl_apps.tiny_bridge_2p_mappo import CCTrainer, CCPPOTorchPolicy

scenario_catalog.add(PSROScenario(
    name="tiny_bridge_4p_psro",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=TinyBridge2pMultiAgentEnv,
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer,
    policy_classes={
        "metanash": CCPPOTorchPolicy,
        "best_response": CCPPOTorchPolicy,
        "eval": CCPPOTorchPolicy,
    },
    num_eval_workers=8,
    games_per_payoff_eval=20000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_tiny_bridge_ccppo_params,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))
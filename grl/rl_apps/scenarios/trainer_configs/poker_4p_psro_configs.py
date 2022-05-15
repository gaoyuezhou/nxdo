import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.utils import merge_dicts
from ray.tune.registry import RLLIB_ACTION_DIST, _global_registry

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, \
    GRL_DEFAULT_POKER_PPO_PARAMS
from grl.rllib_tools.action_dists import TorchGaussianSquashedGaussian
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy

from grl.envs.tiny_bridge_2p_multi_agent_env  import TinyBridge2pMultiAgentEnv
from grl.envs.tiny_bridge_4p_multi_agent_env  import TinyBridge4pMultiAgentEnv
from grl.envs.poker_4p_multi_agent_env  import Poker4pMultiAgentEnv
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critic_model import TorchCentralizedCriticModel

from gym.spaces import Discrete

from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model(
        "cc_model", TorchCentralizedCriticModel)


def psro_kuhn_4p_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "kuhn_poker",
        "fixed_players": True,
    }
    tmp_env = Poker4pMultiAgentEnv(env_config)

    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.001,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "lr": 0.0005,
        "batch_mode": "complete_episodes",
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 1,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 128,
        "train_batch_size": 2048,
        "vf_clip_param": 10.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 1,

    }
    return config

def psro_leduc_4p_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "leduc_poker",
        "fixed_players": True,
    }
    tmp_env = Poker4pMultiAgentEnv(env_config)
    return {
        "batch_mode": "complete_episodes",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 20,
        "model": {
            "custom_model": "cc_model",
            "vf_share_layers": False ### overriding the default parameter
        },
        "framework": "torch",
    }

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
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critique_model import TorchCentralizedCriticModel

from gym.spaces import Discrete

from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model(
        "cc_model", TorchCentralizedCriticModel)

def psro_tiny_bridge_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "tiny_bridge_2p",
        "fixed_players": True,
    }
    tmp_env = TinyBridge2pMultiAgentEnv(env_config)
    return {
        "env": TinyBridge2pMultiAgentEnv,
        "batch_mode": "complete_episodes",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "multiagent": {
            "policies": {
                "pol1": (None, tmp_env.observation_space, Discrete(7), {
                    "framework": "torch",
                }),
            },
            "policy_mapping_fn": lambda x: "pol1" if x == 0 else "pol1",
        },
        "model": {
            "custom_model": "cc_model",
            "vf_share_layers": False ### overriding the default parameter
        },
        "framework": "torch",
    }

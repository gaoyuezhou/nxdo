import os
from typing import Dict

from gym.spaces import Discrete, Space, Box
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from grl.envs.oshi_zumo_multi_agent_env import OSHI_ZUMO_OBS_LENGTH
from grl.envs.poker_multi_agent_env import OBS_SHAPES, LEDUC_POKER
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rllib_tools.valid_actions_fcnet import get_valid_action_fcn_class

_LEDUC_OBS_LEN = OBS_SHAPES[LEDUC_POKER][0]
_12_NO_LIMIT_LEDUC_OBS_LEN = 60
_30_NO_LIMIT_LEDUC_OBS_LEN = 64
_60_NO_LIMIT_LEDUC_OBS_LEN = 68
EXTENDED_LEDUC_OBS_LEN = 68


def psro_kuhn_sac_params(action_space: Discrete):
    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,

        "framework": "torch",
        # RL Algo Specific
        "initial_alpha": 0.0,
        "target_entropy": 0,
        "train_batch_size": 1024,
        "rollout_fragment_length": 10,
        "normalize_actions": False,
        "model": merge_dicts(MODEL_DEFAULTS, {}),

        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [60, 60],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [60, 60],
        },

        "use_state_preprocessor": False,
        "optimization": {
            "actor_learning_rate": 1e-2,
            "critic_learning_rate": 1e-2,
            "entropy_learning_rate": 1e-2,
        },

    }


def psro_kuhn_dqn_params(action_space: Space) -> Dict:
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        # "double_q": False,
        "double_q": True,

        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(3e7),  # Timesteps over which to anneal epsilon.
            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        # "target_network_update_freq": 5000,
        "target_network_update_freq": 10000,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 2000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 64,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 128,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def psro_leduc_dqn_params(action_space: Space) -> Dict:
    assert isinstance(action_space, Discrete)
    action_space: Discrete = action_space
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6) * 10,  # Timesteps over which to anneal epsilon.
            # "epsilon_timesteps": int(200000)
            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 19200 * 10,
        # "target_network_update_freq": 10000,

        # "target_network_update_freq": 1,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 2000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 64,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 128,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n),
        }),
    }


def psro_fast_leduc_dqn_params(action_space: Space) -> Dict:
    assert isinstance(action_space, Discrete)
    action_space: Discrete = action_space
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 10000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        # "double_q": False,
        "double_q": True,

        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(3e7),  # Timesteps over which to anneal epsilon.
            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        # "target_network_update_freq": 5000,
        "target_network_update_freq": 10000,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 2000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 64,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 128,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n),
        }),
    }


def psro_leduc_ppo_params_gpu(action_space: Space) -> Dict:
    assert isinstance(action_space, Box)
    action_space: Box = action_space
    return {
        "framework": "torch",
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 256,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 2048,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,
        # Switch on Trajectory View API for PPO by default.
        # NOTE: Only supported for PyTorch so far.
        "_use_trajectory_view_api": True,

        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": None,
        }),
    }


def psro_leduc_dqn_params_gpu(action_space: Space) -> Dict:
    assert isinstance(action_space, Discrete)
    action_space: Discrete = action_space
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 10000,
        # "target_network_update_freq": 1,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n),
        }),
    }


def psro_20x_dummy_leduc_params_gpu(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=20)
    return params


def psro_20x_dummy_leduc_params_gpu_more_experience(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["rollout_fragment_length"] = params["rollout_fragment_length"] * 2
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=20)
    return params


def psro_40x_dummy_leduc_params_gpu(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=40)
    return params


def psro_extended_leduc_params_gpu(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["metrics_smoothing_episodes"] = 6000
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=EXTENDED_LEDUC_OBS_LEN,
                                                                 action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=1)
    return params


def psro_80x_dummy_leduc_params_gpu(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=80)
    return params


def psro_12_no_limit_leduc_params_gpu(action_space: Space) -> Dict:
    assert action_space.n == 13, action_space.n
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    # params["metrics_smoothing_episodes"] = 3000
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_12_NO_LIMIT_LEDUC_OBS_LEN,
                                                                 action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=1)
    return params


def psro_30_no_limit_leduc_params_gpu(action_space: Space) -> Dict:
    assert action_space.n == 31, action_space.n
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_30_NO_LIMIT_LEDUC_OBS_LEN,
                                                                 action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=1)
    return params


def psro_60_no_limit_leduc_params_gpu(action_space: Space) -> Dict:
    assert action_space.n == 61, action_space.n
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=_60_NO_LIMIT_LEDUC_OBS_LEN,
                                                                 action_space_n=action_space.n,
                                                                 dummy_actions_multiplier=1)
    return params


def psro_kuhn_dqn_params_gpu(action_space: Space) -> Dict:
    assert isinstance(action_space, Discrete)
    action_space: Discrete = action_space
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 10000,
        # "target_network_update_freq": 1,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def psro_20x_dummy_leduc_params_gpu_v2(action_space: Space) -> Dict:
    params = psro_20x_dummy_leduc_params_gpu(action_space=action_space)
    params["lr"] = 0.001
    params["metrics_smoothing_episodes"] = 3000
    return params


def psro_oshi_zumo_dqn_params_like_leduc_gpu(action_space: Space) -> Dict:
    params = psro_leduc_dqn_params_gpu(action_space=action_space)
    params["lr"] = 0.001
    params["model"]["fcnet_hiddens"] = [128, 128]
    params["model"]["custom_model"] = get_valid_action_fcn_class(obs_len=OSHI_ZUMO_OBS_LENGTH,
                                                                 action_space_n=action_space.n)
    return params
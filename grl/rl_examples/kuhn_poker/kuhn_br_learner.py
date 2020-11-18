import os
import time
from typing import Dict, List

import deepdish

import ray
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, \
    convert_to_torch_tensor
from ray.rllib.utils.typing import ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from grl.payoff_table import PayoffTableStrategySpec
from grl.rl_examples.kuhn_poker.poker_multi_agent_env import PokerMultiAgentEnv
from grl.utils import pretty_dict_str, datetime_str, ensure_dir
from grl.p2sro_manager import RemoteP2SROManagerClient
from grl.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rl_examples.kuhn_poker.config import kuhn_sac_params

torch, _ = try_import_torch()

BR_CHECKPOINT_SAVE_DIR = "/tmp/p2sro_policies"

# class SACTorchPolicyCheckpointMixin():
#     def get_weights(self):
#         return {
#             "main": {
#                 k: v.cpu().detach().numpy()
#                 for k, v in self.model.state_dict().items()
#             },
#             "target": {
#                 k: v.cpu().detach().numpy()
#                 for k, v in self.target_model.state_dict().items()
#             }
#         }
#
#     def set_weights(self, main_and_target_model_weights) -> None:
#         model_weights: ModelWeights = main_and_target_model_weights["main"]
#         target_model_weights: ModelWeights = main_and_target_model_weights["target"]
#         model_weights = convert_to_torch_tensor(model_weights, device=self.device)
#         self.model.load_state_dict(model_weights)
#         target_model_weights = convert_to_torch_tensor(target_model_weights, device=self.device)
#         self.target_model_weights.load_state_dict(target_model_weights)
#


def save_best_response_checkpoint(trainer: SACTrainer,
                                  save_dir: str,
                                  timesteps_training_br: int,
                                  episodes_training_br: int,
                                  active_policy_num: int = None):
    policy_name = active_policy_num if active_policy_num is not None else "unclaimed"
    date_time = datetime_str()
    checkpoint_name = f"policy_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights(["best_response"])["best_response"]
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(path=checkpoint_path, data={
        "weights": br_weights,
        "policy_num": active_policy_num,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    })
    return checkpoint_path


def load_metanash_pure_strat(policy: SACTorchPolicy, pure_strat_spec: PayoffTableStrategySpec):
    pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    policy.set_weights(checkpoint_data["weights"])
    policy.p2sro_policy_spec = pure_strat_spec


def set_best_response_active_policy_spec_for_all_workers(trainer: SACTrainer,
                                                         active_policy_spec: PayoffTableStrategySpec):
    def _set_p2sro_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy: SACTorchPolicy = worker.policy_map["best_response"]
        br_policy.p2sro_policy_spec = active_policy_spec
    trainer.workers.foreach_worker(_set_p2sro_policy_spec_on_best_response_policy)


def update_all_workers_to_latest_metanash(trainer: SACTrainer,
                                          p2sro_manager: RemoteP2SROManagerClient,
                                          active_policy_num: int):

    latest_payoff_table, active_policy_nums, fixed_policy_nums = p2sro_manager.get_copy_of_latest_data()
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=latest_payoff_table,
        as_player=0,
        as_policy_num=active_policy_num,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=0.0
    )
    if latest_strategies is None:
        opponent_policy_distribution = None
    else:
        # get the strategy for the opposing player, 1.
        # In a symmetric two-player game, this is the same as what player 0's strategy would be.
        opponent_policy_distribution = latest_strategies[1]

    def _set_opponent_policy_distribution_for_one_worker(worker: RolloutWorker):
        worker.opponent_policy_distribution = opponent_policy_distribution
    trainer.workers.foreach_worker(_set_opponent_policy_distribution_for_one_worker)


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        # Sample new pure strategy policy weights from the metanash of the subgame population for the best response to
        # train against. For better runtime performance, consider loading new weights only every few episodes instead.
        resample_pure_strat_every_n_episodes = 1
        metanash_policy: SACTorchPolicy = policies["metanash"]
        opponent_policy_distribution: PolicySpecDistribution = worker.opponent_policy_distribution
        time_for_resample = (not hasattr(metanash_policy, "episodes_since_resample") or
                             metanash_policy.episodes_since_resample >= resample_pure_strat_every_n_episodes)
        if time_for_resample and opponent_policy_distribution is not None:
            new_pure_strat_spec: PayoffTableStrategySpec = opponent_policy_distribution.sample_policy_spec()
            # noinspection PyTypeChecker
            load_metanash_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)
            metanash_policy.episodes_since_resample = 1
        elif opponent_policy_distribution is not None:
            metanash_policy.episodes_since_resample += 1

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        if not hasattr(worker, "p2sro_manager"):
            worker.p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=4535, remote_server_host="127.0.0.1")

        br_policy_spec: PayoffTableStrategySpec = worker.policy_map["best_response"].p2sro_policy_spec
        if br_policy_spec.pure_strat_index_for_player(player=0) == 0:
            # We're training policy 0 if True.
            # The PSRO subgame should be empty, and instead the metanash is a random neural network.
            # No need to report results for this.
            return

        # Report payoff results for individual episodes to the p2sro manager to keep a real-time approximation of the
        # payoff matrix entries for (learning) active policies.
        policy_specs_for_each_player: List[PayoffTableStrategySpec] = [None, None]
        payoffs_for_each_player: List[float] = [None, None]
        for (player, policy_name), reward in episode.agent_rewards.items():
            assert policy_name in ["best_response", "metanash"]
            policy: SACTorchPolicy = worker.policy_map[policy_name]
            assert policy.p2sro_policy_spec is not None
            policy_specs_for_each_player[player] = policy.p2sro_policy_spec
            payoffs_for_each_player[player] = reward
        assert all(payoff is not None for payoff in payoffs_for_each_player)

        worker.p2sro_manager.submit_empirical_payoff_result(
            policy_specs_for_each_player=tuple(policy_specs_for_each_player),
            payoffs_for_each_player=tuple(payoffs_for_each_player),
            games_played=1,
            override_all_previous_results=False)


def train_poker_sac_best_response():

    def select_policy(agent_id):
        if agent_id == 0:
            return "best_response"
        else:
            return "metanash"

    env_config = {'version': "leduc_poker"}
    tmp_env = PokerMultiAgentEnv(env_config=env_config)

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": PokerMultiAgentEnv,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "metanash": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
                "best_response": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    trainer_config = merge_dicts(trainer_config, kuhn_sac_params)

    ray.init(ignore_reinit_error=True, local_mode=True)
    trainer = SACTrainer(config=trainer_config)
    p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=4535, remote_server_host="127.0.0.1")
    active_policy_spec: PayoffTableStrategySpec = p2sro_manager.claim_new_active_policy_for_player(
        player=0, new_policy_metadata_dict={
            "checkpoint_path": save_best_response_checkpoint(trainer=trainer, save_dir=BR_CHECKPOINT_SAVE_DIR,
                                                             active_policy_num=None,
                                                                     timesteps_training_br=0,
                                                                     episodes_training_br=0),
                    "timesteps_training_br": 0,
                    "episodes_training_br": 0
                })

    print(f"active policy pure strat indexes are: {active_policy_spec._pure_strategy_indexes}")

    active_policy_num = active_policy_spec.pure_strat_index_for_player(player=0)
    print(f"got policy {active_policy_num}")

    set_best_response_active_policy_spec_for_all_workers(trainer=trainer, active_policy_spec=active_policy_spec)

    update_all_workers_to_latest_metanash(p2sro_manager=p2sro_manager, trainer=trainer,
                                          active_policy_num=active_policy_num)

    # Perform main RL training loop. Stop if we reach max iters or saturate.
    # Saturation is determined by checking if we improve by a minimum amount every n iters.
    max_train_iters = 1000
    dont_do_saturation_checks_before_n_train_iters = 100
    check_for_saturation_every_n_train_iters = 100
    minimum_reward_improvement_otherwise_saturated = 0.1
    last_saturation_check_reward = None
    train_iter_count = 0
    total_timesteps_training_br = 0
    total_episodes_training_br = 0
    while True:
        train_iter_results = trainer.train() # do a step (or several) in the main RL loop
        train_iter_count += 1

        # Delete verbose debugging info before printing
        if "hist_stats" in train_iter_results:
            del train_iter_results["hist_stats"]
        if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
            del train_iter_results["info"]["learner"]["best_response"]["td_error"]
        print(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"]["best_response"]

        time_to_stop_training = False

        if train_iter_count % check_for_saturation_every_n_train_iters == 0:
            if last_saturation_check_reward is not None:
                # Do a checkpoint for other learners to use and get the latest metanash subgame stats.

                # Checkpoint first.
                p2sro_manager.submit_new_active_policy_metadata(player=0, policy_num=active_policy_num, metadata_dict={
                    "checkpoint_path": save_best_response_checkpoint(trainer=trainer, save_dir=BR_CHECKPOINT_SAVE_DIR,
                                                                     active_policy_num=active_policy_num,
                                                                     timesteps_training_br=total_timesteps_training_br,
                                                                     episodes_training_br=total_episodes_training_br),
                    "timesteps_training_br": total_timesteps_training_br,
                    "episodes_training_br": total_episodes_training_br
                })

                # Get latest stats
                update_all_workers_to_latest_metanash(p2sro_manager=p2sro_manager, trainer=trainer,
                                                      active_policy_num=active_policy_num)

                improvement_since_last_check = br_reward_this_iter - last_saturation_check_reward
                print(f"Improvement since last saturation check: {improvement_since_last_check}, minimum target is "
                      f"{minimum_reward_improvement_otherwise_saturated}.")
                if (improvement_since_last_check < minimum_reward_improvement_otherwise_saturated and
                    train_iter_count > dont_do_saturation_checks_before_n_train_iters):
                    # We're no longer improving. Assume we have saturated, and stop training.
                    print(f"Improvement target not reached, stopping training if allowed.")
                    time_to_stop_training = True
            last_saturation_check_reward = br_reward_this_iter

        if train_iter_count >= max_train_iters:
            # Regardless of whether we've saturated, we've been training for too long, so we stop.
            print(f"Max training iters reached ({train_iter_count}). stopping training if allowed.")
            time_to_stop_training = True

        if time_to_stop_training:
            if p2sro_manager.can_active_policy_be_set_as_fixed_now(player=0, policy_num=active_policy_num):
                break
            else:
                print(f"Forcing training to continue since lower policies are still active.")

    p2sro_manager.set_active_policy_as_fixed(
        player=0, policy_num=active_policy_num, final_metadata_dict={
        "checkpoint_path": save_best_response_checkpoint(trainer=trainer, save_dir=BR_CHECKPOINT_SAVE_DIR,
                                                         active_policy_num=active_policy_num,
                                                         timesteps_training_br=total_timesteps_training_br,
                                                         episodes_training_br=total_episodes_training_br)
    })
    return active_policy_num


if __name__ == "__main__":
    train_poker_sac_best_response()

import argparse
import logging
import os
import time
from typing import Dict, Type, List

import ray
from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils import merge_dicts, try_import_torch

from grl.algos.p2sro.p2sro_manager import RemoteP2SROManagerClient
from grl.algos.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rl_apps import GRL_SEED
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rllib_tools.policy_checkpoints import save_policy_checkpoint, load_pure_strat
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import pretty_dict_str
from grl.utils.port_listings import get_client_port_for_service
from grl.utils.strategy_spec import StrategySpec

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def create_metadata_with_new_checkpoint_for_current_best_response(trainer: Trainer,
                                                                  player: int,
                                                                  save_dir: str,
                                                                  timesteps_training_br: int,
                                                                  episodes_training_br: int,
                                                                  active_policy_num: int = None,
                                                                  average_br_reward: float = None):
    return {
        "checkpoint_path": save_policy_checkpoint(trainer=trainer,
                                                  player=player,
                                                  save_dir=save_dir,
                                                  policy_id_to_save="best_response",
                                                  checkpoint_name=f"player_{player}_policy_{active_policy_num}",
                                                  additional_data={
                                                      "policy_num": active_policy_num,
                                                      "timesteps_training_br": timesteps_training_br,
                                                      "episodes_training_br": episodes_training_br,
                                                      "average_br_reward": average_br_reward,
                                                  }),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br,
        "average_br_reward": average_br_reward,
    }


def set_best_response_active_policy_spec_and_player_for_all_workers(trainer: Trainer,
                                                                    player: int,
                                                                    active_policy_spec: StrategySpec):
    def _set_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy: Policy = worker.policy_map[f"best_response"]
        br_policy.policy_spec = active_policy_spec
        worker.br_player = player

    trainer.workers.foreach_worker(_set_policy_spec_on_best_response_policy)


def update_all_workers_to_latest_metanash(trainer: Trainer,
                                          br_player: int,
                                          metanash_player: int,
                                          p2sro_manager: RemoteP2SROManagerClient,
                                          active_policy_num: int,
                                          mix_metanash_with_uniform_dist_coeff: float,
                                          one_agent_plays_all_sides: bool = False):
    latest_payoff_table, active_policy_nums, fixed_policy_nums = p2sro_manager.get_copy_of_latest_data()
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=latest_payoff_table,
        as_player=1 if one_agent_plays_all_sides else br_player,
        as_policy_num=active_policy_num,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff
    )

    if latest_strategies is None:
        opponent_policy_distribution = None
    else:
        opponent_player = 0 if one_agent_plays_all_sides else metanash_player
        print(f"latest payoff matrix for player {opponent_player}:\n"
              f"{latest_payoff_table.get_payoff_matrix_for_player(player=opponent_player)}")
        print(f"metanash for player {opponent_player}: "
              f"{latest_strategies[opponent_player].probabilities_for_each_strategy()}")

        # get the strategy distribution for the opposing player.
        opponent_policy_distribution = latest_strategies[opponent_player]

        # double check that these policy specs are for the opponent player
        assert opponent_player in opponent_policy_distribution.sample_policy_spec().get_pure_strat_indexes().keys()

    def _set_opponent_policy_distribution_for_worker(worker: RolloutWorker):
        worker.opponent_policy_distribution = opponent_policy_distribution

    trainer.workers.foreach_worker(_set_opponent_policy_distribution_for_worker)


def sync_active_policy_br_and_metanash_with_p2sro_manager(trainer: DQNTrainer,
                                                          player: int,
                                                          metanash_player: int,
                                                          one_agent_plays_all_sides: bool,
                                                          p2sro_manager: RemoteP2SROManagerClient,
                                                          mix_metanash_with_uniform_dist_coeff: float,
                                                          active_policy_num: int,
                                                          timesteps_training_br: int,
                                                          episodes_training_br: int):
    p2sro_manager.submit_new_active_policy_metadata(
        player=player, policy_num=active_policy_num,
        metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=player, save_dir=checkpoint_dir(trainer),
            timesteps_training_br=timesteps_training_br,
            episodes_training_br=episodes_training_br,
            active_policy_num=active_policy_num
        ))

    update_all_workers_to_latest_metanash(p2sro_manager=p2sro_manager, br_player=player,
                                          metanash_player=metanash_player, trainer=trainer,
                                          active_policy_num=active_policy_num,
                                          mix_metanash_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
                                          one_agent_plays_all_sides=one_agent_plays_all_sides)


def train_psro_best_response(team: int, results_dir: str, scenario_name: str, psro_manager_port: int,
                             psro_manager_host: str, print_train_results=True, previous_br_checkpoint_path=None) -> str:
    # import pdb; pdb.set_trace()

    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)
    if not isinstance(scenario, PSROScenario):
        raise TypeError(f"Only instances of {PSROScenario} can be used here. {scenario.name} is a {type(scenario)}.")

    env_class = scenario.env_class
    env_config = scenario.env_config
    trainer_class = scenario.trainer_class
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    single_agent_symmetric_game = scenario.single_agent_symmetric_game
    if single_agent_symmetric_game and team != 0:
        if team is None:
            team = 0
        else:
            raise ValueError(f"If treating the game as single agent symmetric, only use player 0 "
                             f"(one agent plays all sides).")

    p2sro = scenario.p2sro
    p2sro_sync_with_payoff_table_every_n_episodes = scenario.p2sro_sync_with_payoff_table_every_n_episodes
    get_trainer_config = scenario.get_trainer_config
    psro_get_stopping_condition = scenario.psro_get_stopping_condition
    mix_metanash_with_uniform_dist_coeff = scenario.mix_metanash_with_uniform_dist_coeff
    allow_stochastic_best_response = scenario.allow_stochastic_best_responses
    should_log_result_fn = scenario.ray_should_log_result_filter

    class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

        # def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode,
        #                     env_index: int,
        #                     **kwargs):
        #     super().on_episode_step(worker=worker, base_env=base_env, episode=episode, env_index=env_index, **kwargs)
        #
        #     # Debug render a single environment.
        #     if worker.worker_index == 1 and env_index == 0:
        #         base_env.get_unwrapped()[0].render()

        def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):

            # Sample new pure strategy policy weights from the opponent strategy distribution for the best response to
            # train against. For better runtime performance, this function can be modified to load new weights
            # only every few episodes instead.
            resample_pure_strat_every_n_episodes = 1
            metanash_policy: Policy = policies[f"metanash"]
            opponent_policy_distribution: PolicySpecDistribution = worker.opponent_policy_distribution
            time_for_resample = (not hasattr(metanash_policy, "episodes_since_resample") or
                                 metanash_policy.episodes_since_resample >= resample_pure_strat_every_n_episodes)
            if time_for_resample and opponent_policy_distribution is not None:
                new_pure_strat_spec: StrategySpec = opponent_policy_distribution.sample_policy_spec()
                # noinspection PyTypeChecker
                load_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)
                metanash_policy.episodes_since_resample = 1
            elif opponent_policy_distribution is not None:
                metanash_policy.episodes_since_resample += 1

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            result["scenario_name"] = trainer.scenario_name
            super().on_train_result(trainer=trainer, result=result, **kwargs)

        def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           env_index: int, **kwargs):

            # If using P2SRO, report payoff results of the actively training BR to the payoff table.
            if not p2sro:
                return

            if not hasattr(worker, "p2sro_manager"):
                worker.p2sro_manager = RemoteP2SROManagerClient(n_players=2,
                                                                port=psro_manager_port,
                                                                remote_server_host=psro_manager_host)

            br_policy_spec: StrategySpec = worker.policy_map["best_response"].policy_spec
            if br_policy_spec.pure_strat_index_for_player(player=worker.br_player) == 0:
                # We're training policy 0 if True (first iteration of PSRO).
                # The PSRO subgame should be empty, and instead the metanash is a random neural network.
                # No need to report payoff results for this.
                return

            # Report payoff results for individual episodes to the p2sro manager to keep a real-time approximation
            # of the payoff matrix entries for (learning) active policies.
            policy_specs_for_each_player: List[StrategySpec] = [None, None]
            payoffs_for_each_player: List[float] = [None, None]
            for (player, policy_name), reward in episode.agent_rewards.items():
                assert policy_name in ["best_response", "metanash"]
                policy: Policy = worker.policy_map[policy_name]
                assert policy.policy_spec is not None
                policy_specs_for_each_player[player] = policy.policy_spec
                payoffs_for_each_player[player] = reward
            assert all(payoff is not None for payoff in payoffs_for_each_player)

            # Send payoff result to the manager for inclusion in the payoff table.
            worker.p2sro_manager.submit_empirical_payoff_result(
                policy_specs_for_each_player=tuple(policy_specs_for_each_player),
                payoffs_for_each_player=tuple(payoffs_for_each_player),
                games_played=1,
                override_all_previous_results=False)

    # player is a team now, so team 0 is player 0 and 2, and team 1 is player 1 and 3
    other_team = 1 - team
    br_learner_name = f"new_learner_{team}"

    def log(message):
        print(f"({br_learner_name}): {message}")

    def select_policy(agent_id):
        if agent_id % 2 == team:
            return "best_response"
        elif agent_id % 2 == other_team:
            return "metanash"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=psro_manager_port,
                                             remote_server_host=psro_manager_host)
    manager_metadata = p2sro_manager.get_manager_metadata()
    ray_head_address = manager_metadata["ray_head_address"]
    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

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
                policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space, {"explore": allow_stochastic_best_response}),
                f"best_response": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))

    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(
                                base_dir=results_dir, scenario_name=scenario_name,
                                should_log_result_fn=should_log_result_fn))

    # scenario_name logged in on_train_result_callback
    trainer.scenario_name = scenario_name

    if previous_br_checkpoint_path is not None:
        def _set_br_initial_weights(worker: RolloutWorker):
            br_policy = worker.policy_map["best_response"]
            load_pure_strat(policy=br_policy, checkpoint_path=previous_br_checkpoint_path)
        trainer.workers.foreach_worker(_set_br_initial_weights)

    active_policy_spec: StrategySpec = p2sro_manager.claim_new_active_policy_for_player(
        player=team, new_policy_metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=team, save_dir=checkpoint_dir(trainer), timesteps_training_br=0,
            episodes_training_br=0,
            active_policy_num=None
        ))

    active_policy_num = active_policy_spec.pure_strat_index_for_player(player=team)
    br_learner_name = f"policy {active_policy_num} player {team}"

    log(f"got policy {active_policy_num}")

    set_best_response_active_policy_spec_and_player_for_all_workers(trainer=trainer, player=team,
                                                                    active_policy_spec=active_policy_spec)

    sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=trainer,
                                                          player=team,
                                                          metanash_player=other_team,
                                                          one_agent_plays_all_sides=single_agent_symmetric_game,
                                                          p2sro_manager=p2sro_manager,
                                                          mix_metanash_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
                                                          active_policy_num=active_policy_num,
                                                          timesteps_training_br=0,
                                                          episodes_training_br=0)

    # Perform main RL training loop. Stop training according to our StoppingCondition.
    train_iter_count = 0
    episodes_since_last_sync_with_manager = 0
    stopping_condition: StoppingCondition = psro_get_stopping_condition()

    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["p2sro_active_policy_num"] = active_policy_num
            train_iter_results["best_response_player"] = team
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"][f"best_response"]:
                del train_iter_results["info"]["learner"][f"best_response"]["td_error"]
            log(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]

        episodes_since_last_sync_with_manager += train_iter_results["episodes_this_iter"]
        if p2sro and episodes_since_last_sync_with_manager >= p2sro_sync_with_payoff_table_every_n_episodes:
            if p2sro_sync_with_payoff_table_every_n_episodes > 0:
                episodes_since_last_sync_with_manager = episodes_since_last_sync_with_manager % p2sro_sync_with_payoff_table_every_n_episodes
            else:
                episodes_since_last_sync_with_manager = 0

            sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=trainer,
                                                                  player=team,
                                                                  metanash_player=other_player,
                                                                  one_agent_plays_all_sides=single_agent_symmetric_game,
                                                                  p2sro_manager=p2sro_manager,
                                                                  mix_metanash_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
                                                                  active_policy_num=active_policy_num,
                                                                  timesteps_training_br=total_timesteps_training_br,
                                                                  episodes_training_br=total_episodes_training_br)

        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            if p2sro_manager.can_active_policy_be_set_as_fixed_now(player=team, policy_num=active_policy_num):
                break
            else:
                log(f"Forcing training to continue since lower policies are still active.")

    log(f"Training stopped. Setting active policy {active_policy_num} as fixed.")

    final_policy_metadata = create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=team, save_dir=checkpoint_dir(trainer=trainer),
            timesteps_training_br=total_timesteps_training_br,
            episodes_training_br=total_episodes_training_br,
            active_policy_num=active_policy_num,
            average_br_reward=train_iter_results["policy_reward_mean"]["best_response"])

    p2sro_manager.set_active_policy_as_fixed(
        player=team, policy_num=active_policy_num,
        final_metadata_dict=final_policy_metadata)

    trainer.cleanup()
    ray.shutdown()
    time.sleep(10)

    if not p2sro:
        # wait for both player policies to be fixed.
        for player_to_wait_on in range(2):
            wait_count = 0
            while True:
                if p2sro_manager.is_policy_fixed(player=player_to_wait_on, policy_num=active_policy_num):
                    break
                if wait_count % 10 == 0:
                    log(f"Waiting for policy {active_policy_num} player {player_to_wait_on} to become fixed")
                time.sleep(2.0)
                wait_count += 1

    return final_policy_metadata["checkpoint_path"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=int)
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--use_prev_brs', default=False, action='store_true')
    parser.add_argument('--psro_port', type=int, required=False, default=None)
    parser.add_argument('--psro_host', type=str, required=False, default='localhost')
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    use_prev_brs = commandline_args.use_prev_brs

    psro_host = commandline_args.psro_host
    psro_port = commandline_args.psro_port
    if psro_port is None:
        psro_port = get_client_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario_name}")

    manager_log_dir = RemoteP2SROManagerClient(n_players=2, port=os.getenv("P2SRO_PORT", psro_port),
                                               remote_server_host=psro_host).get_log_dir()
    results_dir = os.path.join(manager_log_dir, f"learners_team_{commandline_args.team}/")
    print(f"results dir is {results_dir}")

    previous_br_checkpoint_path = None
    while True:
        # Train a br for the specified player, then repeat.
        result = train_psro_best_response(
            team=commandline_args.team,
            results_dir=results_dir,
            scenario_name=scenario_name,
            psro_manager_port=psro_port,
            psro_manager_host=psro_host,
            print_train_results=True,
            previous_br_checkpoint_path=previous_br_checkpoint_path
        )
        if use_prev_brs:
            previous_br_checkpoint_path = result
import os
import json
from typing import Tuple
import logging
from abc import ABC, abstractmethod

from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from grl.utils import ensure_dir

logger = logging.getLogger(__name__)


class P2SROManagerLogger(ABC):
    """
    Logging for a P2SROManager.
    Extend this class to add additional functionality like tracking exploitability over time for small games.
    """

    @abstractmethod
    def __init__(self, p2sro_manger, log_dir: str):
        pass

    def on_new_active_policy(self, player: int, new_policy_num: int, new_policy_spec: PayoffTableStrategySpec):
        pass

    def on_new_active_policy_metadata(self, player: int, policy_num: int, new_policy_spec: PayoffTableStrategySpec):
        pass

    def on_active_policy_moved_to_fixed(self, player: int, policy_num: int, fixed_policy_spec: PayoffTableStrategySpec):
        pass

    def on_payoff_result(self,
                                policy_specs_for_each_player: Tuple[PayoffTableStrategySpec],
                                payoffs_for_each_player: Tuple[float],
                                games_played: int,
                                overrode_all_previous_results: bool):
        pass


class SimpleP2SROManagerLogger(P2SROManagerLogger):
    """
    Saves payoff table checkpoints every time an active policy is set to fixed.
    """

    def __init__(self, p2sro_manger, log_dir: str):
        super().__init__(p2sro_manger, log_dir)
        self._log_dir = log_dir
        self._manager = p2sro_manger

        self._payoff_table_checkpoint_dir = os.path.join(self._log_dir, "payoff_table_checkpoints")
        self._payoff_table_checkpoint_count = 0

    def on_new_active_policy(self, player: int, new_policy_num: int, new_policy_spec: PayoffTableStrategySpec):
        logger.info(f"Player {player} active policy {new_policy_num} claimed")

    def on_new_active_policy_metadata(self, player: int, policy_num: int, new_policy_spec: PayoffTableStrategySpec):
        logger.info(f"Player {player} active policy {policy_num} new metadata: {new_policy_spec.metadata}")

    def on_active_policy_moved_to_fixed(self, player: int, policy_num: int, fixed_policy_spec: PayoffTableStrategySpec):
        logger.info(f"Player {player} policy {policy_num} moved to fixed.")

        # save a checkpoint of the payoff table
        data = self._manager.get_copy_of_latest_data()
        latest_payoff_table, active_policy_nums_per_player, fixed_policy_nums_per_player = data
        pt_checkpoint_path = os.path.join(self._payoff_table_checkpoint_dir,
                                          f"payoff_table_checkpoint_{self._payoff_table_checkpoint_count}.json")
        policy_nums_path = os.path.join(self._payoff_table_checkpoint_dir,
                                        f"policy_nums_checkpoint_{self._payoff_table_checkpoint_count}.json")
        ensure_dir(file_path=pt_checkpoint_path)
        ensure_dir(file_path=policy_nums_path)

        latest_payoff_table.to_json_file(file_path=pt_checkpoint_path)
        print(f"\n\n\nSaved payoff table checkpoint to {pt_checkpoint_path}")

        player_policy_nums = {}
        for player_i, (active_policy_nums, fixed_policy_nums) in enumerate(
                zip(active_policy_nums_per_player, fixed_policy_nums_per_player)):
            player_policy_nums[player_i] = {
                "active_policies": active_policy_nums,
                "fixed_policies": fixed_policy_nums
            }

        with open(policy_nums_path, "w+") as policy_nums_file:
            json.dump(obj=player_policy_nums, fp=policy_nums_file)
        print(f"Saved policy nums checkpoint to {policy_nums_path}\n\n\n")

        self._payoff_table_checkpoint_count += 1

    def on_payoff_result(self, policy_specs_for_each_player: Tuple[PayoffTableStrategySpec],
                                payoffs_for_each_player: Tuple[float], games_played: int,
                                overrode_all_previous_results: bool):
        pass
        json_specs = [spec.to_json() for spec in policy_specs_for_each_player]
        logger.debug(f"Payoff result for {json_specs}, payoffs: {payoffs_for_each_player}, games: {games_played},"
                    f" overrides existing results: {overrode_all_previous_results}")


        data = self._manager.get_copy_of_latest_data()
        latest_payoff_table, active_policy_nums_per_player, fixed_policy_nums_per_player = data
        latest_payoff_table: PayoffTable = latest_payoff_table
        # print("Player 0 matrix ---------------------------------------")
        # print(latest_payoff_table.get_payoff_matrix_for_player(0))
        # print("------------------------------------------------------")
        # print("Player 1 matrix ---------------------------------------")
        # print(latest_payoff_table.get_payoff_matrix_for_player(1))
        # print("------------------------------------------------------")

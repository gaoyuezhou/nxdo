from typing import Union, Type, Dict, Any, Callable

from ray.rllib import MultiAgentEnv, Policy
from ray.rllib.agents import Trainer

from grl.rl_apps.scenarios.scenario import RayScenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition


class PSROScenario(RayScenario):

    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float],
                 env_class: Type[MultiAgentEnv],
                 env_config: Dict[str, Any],
                 mix_metanash_with_uniform_dist_coeff: float,
                 trainer_class: Type[Trainer],
                 policy_classes: Dict[str, Type[Policy]],
                 num_eval_workers: int,
                 games_per_payoff_eval: int,
                 p2sro: bool,
                 get_trainer_config: Callable[[MultiAgentEnv], Dict[str, Any]],
                 psro_get_stopping_condition: Callable[[], StoppingCondition]):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes)
        self.env_class = env_class
        self.env_config = env_config
        self.mix_metanash_with_uniform_dist_coeff = mix_metanash_with_uniform_dist_coeff
        self.trainer_class = trainer_class
        self.policy_classes = policy_classes
        self.num_eval_workers = num_eval_workers
        self.games_per_payoff_eval = games_per_payoff_eval
        self.p2sro = p2sro
        self.get_trainer_config = get_trainer_config
        self.psro_get_stopping_condition = psro_get_stopping_condition
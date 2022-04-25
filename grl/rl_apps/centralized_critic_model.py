### directly adapts from rllib's centralized_critic_models.py

from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class
torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name) ### TODO: checl this model_config
        nn.Module.__init__(self)

        # Base of the model
        self.model = get_valid_action_fcn_class(22, 7)(obs_space, action_space, num_outputs,
                             model_config, name) ### TODO: check why the value_function of this model is called! It should not be since we want to use only the central_value_function
                                                ### can figure this out by renaming this and never instantiate self. model
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 29 + 29 + 7  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        # raise ValueError(model_out, type(self.model), input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        if obs.shape[0] == opponent_obs.shape[0]:
            # print(1/0)
            input_ = torch.cat([
                obs, opponent_obs,
                torch.nn.functional.one_hot(opponent_actions, 7).float()
            ], 1)
            # print("### C success!", obs.shape,  opponent_obs.shape) ## return has the same shape[0] as input
            return torch.reshape(self.central_vf(input_), [-1])
        else:
            import pdb; pdb.set_trace()
            print(1/0)
        
    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
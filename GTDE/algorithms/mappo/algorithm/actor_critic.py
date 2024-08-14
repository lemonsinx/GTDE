import torch
import torch.nn as nn
from GTDE.algorithms.utils.util import init, check
from GTDE.algorithms.utils.mlp import MLPBase
from GTDE.algorithms.utils.rnn import RNNLayer
from GTDE.algorithms.utils.act import ACTLayer
from GTDE.algorithms.utils.popart import PopArt
from GTDE.algorithms.utils.link import Link
from GTDE.algorithms.utils.GAT import GAT
from GTDE.utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N,
                            self._use_orthogonal)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_GTGE = args.use_GTGE
        self._use_use_mappo = args.use_mappo
        self._attention_head = args.attention_head
        self.GAT_dim = args.gat_dim

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.obs_dim = cent_obs_space[0]
        self.ally_features = cent_obs_space[1]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if self._use_use_mappojo:
            self.base = MLPBase(args, [cent_obs_shape[0] * (self.ally_features[0] + 1)])
        else:
            self.base = MLPBase(args, cent_obs_shape)
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N,
                            self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_GTGE:
            self.GAT = GAT(self.hidden_size, self.GAT_dim, self.hidden_size, self._attention_head)
            self.link = Link(args, cent_obs_space, self.hidden_size)
            if self._use_popart:
                self.v_out = init_(PopArt(self.hidden_size + self.ally_features[0] + 1, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(self.hidden_size + self.ally_features[0] + 1, 1))
        else:
            if self._use_popart:
                self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)
        self._use_uniform = args.use_uniform

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute Value from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        if self._use_use_mappojo:
            feature = cent_obs.shape[-1]
            cent_obs = cent_obs.reshape(-1, feature * (self.ally_features[0] + 1)).repeat_interleave(
                self.ally_features[0] + 1, 0)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        critic_features = self.base(cent_obs)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        if self._use_GTGE:
            edge_matrix = self.link(critic_features.detach())

            critic_features = critic_features.reshape(-1, (self.ally_features[0] + 1), critic_features.shape[-1])
            critic_features = self.GAT(critic_features,
                                       edge_matrix.reshape(-1, (self.ally_features[0] + 1),
                                                           (self.ally_features[
                                                                0] + 1)).detach())
            critic_features = torch.cat([critic_features.reshape(-1, critic_features.shape[-1]), edge_matrix], dim=-1)

        values = self.v_out(critic_features)
        return values, rnn_states

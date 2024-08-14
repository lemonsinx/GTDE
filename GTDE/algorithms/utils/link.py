import torch
from torch import nn
from .util import init


class Link(nn.Module):

    def __init__(self, args, obs_shape, input_dim):
        super(Link, self).__init__()
        self.agent_num = obs_shape[1][0] + 1
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.link_dim
        self._gain = args.gain

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][args.use_ReLU])
        active_func = nn.ReLU()

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.encoder = nn.Sequential(
            init_(nn.Linear(input_dim, self.hidden_size)), active_func, nn.LayerNorm(self.hidden_size),
            init_(nn.Linear(self.hidden_size, self.hidden_size)), active_func, nn.LayerNorm(self.hidden_size))

        self.link = nn.Sequential(init_(nn.Linear(self.hidden_size, self.agent_num)))

    def encode(self, x):
        x = self.encoder(x)
        x = self.link(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = gumbel_sigmoid(x)
        return x


def gumbel_sigmoid(logits):
    gumbels1 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels2 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = logits + gumbels1 - gumbels2
    y_soft = gumbels.sigmoid()
    y_hard = torch.gt(y_soft, 0.5).float()
    ret = (y_hard - y_soft).detach() + y_soft
    return ret

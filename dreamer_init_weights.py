from torch import nn
import numpy as np

# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(state_dict, keyname):
    param = state_dict[keyname]
    if param.dim() == 2:
        out_num, in_num = param.shape
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(param, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    elif param.dim() == 2:
        out_ch, in_ch, k1, k2 = param.shape
        space = k1 * k2
        in_num = space * in_ch
        out_num = space * out_ch
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(param, mean=0.0, std=std, a=-2.0, b=2.0)
    if param.dim() == 1:
        if "layer_norm" in keyname:
            param.fill_(1.0)
        else: # bias
            param.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(state_dict, keyname, given_scale):
    param = state_dict[keyname]
    if param.dim() == 2:
        out_num, in_num = param.shape
        denoms = (in_num + out_num) / 2.0
        scale = given_scale / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(param, a=-limit, b=limit)
    if param.dim() == 1:
        if "layer_norm" in keyname:
            param.fill_(1.0)
        else: # bias
            param.fill_(0.0)


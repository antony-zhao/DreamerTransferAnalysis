import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from dreamer_init_weights import *

initial_model_path = "assault_checkpoint.ckpt" # only used to get the general structure that needs to be loaded later by sheeprl
encoder_model_path = "pretrained_encoder.ckpt"
decoder_model_path = "pretrained_decoder.ckpt"

initial_model_state = torch.load(initial_model_path, map_location="cpu", weights_only=False)
encoder_model_state = torch.load(encoder_model_path, map_location="cpu", weights_only=False)
decoder_model_state = torch.load(decoder_model_path, map_location="cpu", weights_only=False)
world_model_enc_keys = [key for key in initial_model_state["world_model"].keys() if "encoder" in key]
world_model_dec_keys = [key for key in initial_model_state["world_model"].keys() if "decoder" in key]
enc_keys = list(encoder_model_state.keys())
dec_keys = list(decoder_model_state.keys())

for key1 in ['world_model', 'actor', 'critic']:
    # print(initial_model_state[key1].keys())
    for key2 in list(initial_model_state[key1].keys()):
        init_weights(initial_model_state[key1], key2)
uniform_init_weights(initial_model_state["actor"], list(initial_model_state["actor"].keys())[-1], 1.0)
uniform_init_weights(initial_model_state["actor"], list(initial_model_state["actor"].keys())[-2], 1.0)
uniform_init_weights(initial_model_state["critic"], list(initial_model_state["critic"].keys())[-1], 0.0) # accesses the last linear layer of the critic (specifically the weight and the bias)
uniform_init_weights(initial_model_state["critic"], list(initial_model_state["critic"].keys())[-2], 0.0)
reward_model_keys = [key for key in initial_model_state["world_model"].keys() if "reward_model" in key]
uniform_init_weights(initial_model_state["world_model"], reward_model_keys[-1], 0.0) # accesses the last linear layer of the reward model
uniform_init_weights(initial_model_state["world_model"], reward_model_keys[-2], 0.0)
continue_model_keys = [key for key in initial_model_state["world_model"].keys() if "continue_model" in key]
uniform_init_weights(initial_model_state["world_model"], continue_model_keys[-1], 1.0) # accesses the last linear layer of the continue model
uniform_init_weights(initial_model_state["world_model"], continue_model_keys[-2], 1.0)
rep_keys = [key for key in initial_model_state["world_model"].keys() if "representation_model" in key]
uniform_init_weights(initial_model_state["world_model"], rep_keys[-1], 1.0)
uniform_init_weights(initial_model_state["world_model"], rep_keys[-2], 1.0)
transition_keys = [key for key in initial_model_state["world_model"].keys() if "transition_model" in key]
uniform_init_weights(initial_model_state["world_model"], transition_keys[-1], 1.0)
uniform_init_weights(initial_model_state["world_model"], transition_keys[-2], 1.0)
initial_model_state['world_model']['rssm.initial_recurrent_state'] = torch.zeros_like(initial_model_state['world_model']['rssm.initial_recurrent_state'])
initial_model_state['target_critic'] = initial_model_state['critic'].copy()


for i in range(len(world_model_enc_keys)):
    initial_model_state["world_model"][world_model_enc_keys[i]] = encoder_model_state[enc_keys[i]]
for i in range(2, len(world_model_dec_keys)):
    initial_model_state["world_model"][world_model_dec_keys[i]] = decoder_model_state[dec_keys[i]]

initial_model_state["iter_num"] = 0
initial_model_state['last_log'] = 0
initial_model_state['last_checkpoint'] = 0

torch.save(initial_model_state, 'imagenet_enc_only.ckpt')
import torch
from dreamer_init_weights import *

def init_heads(state):
    uniform_init_weights(state["actor"], "mlp_heads.0.weight", 1.0)
    uniform_init_weights(state["critic"], list(state["critic"].keys())[-1], 0.0) # accesses the last linear layer of the critic (specifically the weight and the bias)
    uniform_init_weights(state["critic"], list(state["critic"].keys())[-2], 0.0)
    reward_model_keys = [key for key in state["world_model"].keys() if "reward_model" in key]
    uniform_init_weights(state["world_model"], reward_model_keys[-1], 0.0) # accesses the last linear layer of the reward model
    uniform_init_weights(state["world_model"], reward_model_keys[-2], 0.0)
    continue_model_keys = [key for key in state["world_model"].keys() if "continue_model" in key]
    uniform_init_weights(state["world_model"], continue_model_keys[-1], 1.0) # accesses the last linear layer of the continue model
    uniform_init_weights(state["world_model"], continue_model_keys[-2], 1.0)

def init_models(state):
    for key in state["actor"].keys():
        init_weights(state["actor"], key)
    for key in state["critic"].keys():
        init_weights(state["critic"], key)
    reward_model_keys = [key for key in state["world_model"].keys() if "reward_model" in key]
    for key in reward_model_keys:
        init_weights(state["world_model"], key)
    continue_model_keys = [key for key in state["world_model"].keys() if "continue_model" in key]
    for key in continue_model_keys:
        init_weights(state["world_model"], key)
        
def main(config):
    model_path = config.model_path
    new_path = config.new_path

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    for k, v in state["actor"].items():
        print(k, v.shape)
    for k, v in state["critic"].items():
        print(k, v.shape)
    for k, v in state["world_model"].items():
        print(k, v.shape)
    if config.configuration == 0:
        pass
    elif config.configuration == 1: # replace actor, critic, reward, continue heads
        init_heads(state)
    elif config.configuration == 2: # replace entire actor, critic, reward, continue models
        init_models(state)
        init_heads(state)
    elif config.configuration == 3: # replace encoder last and decoder first
        pass

    state["iter_num"] = 0
    state['last_log'] = 0
    state['last_checkpoint'] = 0

    torch.save(state, new_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--new_path", type=str)
    parser.add_argument("--configuration", type=int, choices=[0, 1, 2]) 
    config = parser.parse_args()
    main(config)
    # 0 is basically no change outside of resetting some parameters
    # 1 is ___

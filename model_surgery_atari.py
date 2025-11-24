import torch
from dreamer_init_weights import *

def main(config):
    model_path = config.model_path
    new_path = config.new_path

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    for k, v in state["actor"].items():
        print(k, v.shape)
    for k, v in state["world_model"].items():
        if "rssm" in k:
            print(k, v.shape)
    if config.configuration == 0:
        pass
    elif config.configuration == 1: # replace actor head, TODO STILL
        state["actor"]["mlp_heads.0.weight"].apply(uniform_init_weights(1.0))
    elif config.configuration == 2: # replace entire actor
        state["actor"].apply(init_weights)
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
    parser.add_argument("--configuration", type=int, choices=[0, 1]) 
    config = parser.parse_args()
    main(config)
    # 0 is basically no change outside of resetting some parameters
    # 1 is ___

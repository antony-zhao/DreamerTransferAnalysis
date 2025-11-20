import torch

num_actions_new = None
model_path = "ckpt_5000000_0.ckpt"
new_path = "ckpt_start_assault.ckpt"

state = torch.load(model_path, map_location="cpu", weights_only=False)


for k, v in state["actor"].items():
    print(k, v.shape)
for k, v in state["world_model"].items():
    if "rssm" in k:
        print(k, v.shape)

old_act_dim = state["actor"]["mlp_heads.0.weight"].shape[0]
new_act_dim = 6

hidden = state["actor"]["mlp_heads.0.weight"].shape[1]

new_actor_w = torch.nn.init.xavier_uniform_(torch.randn(new_act_dim, hidden)) #torch.randn(new_act_dim, hidden) * 0.01
new_actor_b = torch.zeros(new_act_dim)

copy_dim = min(old_act_dim, new_act_dim)
new_actor_w[:copy_dim] = state["actor"]["mlp_heads.0.weight"][:copy_dim]
new_actor_b[:copy_dim] = state["actor"]["mlp_heads.0.bias"][:copy_dim]

state["actor"]["mlp_heads.0.weight"] = new_actor_w
state["actor"]["mlp_heads.0.bias"] = new_actor_b

old_in_dim = state["world_model"]["rssm.recurrent_model.mlp._model.0.weight"].shape[1]
new_in_dim = old_in_dim + new_act_dim - old_act_dim

old_w = state["world_model"]["rssm.recurrent_model.mlp._model.0.weight"]
hidden = old_w.shape[0]

new_w = torch.nn.init.xavier_uniform_(torch.randn(hidden, new_in_dim))

copy_dim = old_in_dim - old_act_dim
new_w[:, :copy_dim] = old_w[:, :copy_dim]

state["world_model"]["rssm.recurrent_model.mlp._model.0.weight"] = new_w
state["iter_num"] = 0
state['last_log'] = 0
state['last_checkpoint'] = 0

torch.save(state, new_path)

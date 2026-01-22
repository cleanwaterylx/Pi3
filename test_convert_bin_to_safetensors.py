import torch
from safetensors.torch import save_file, load_file

state_dict = torch.load(
    "outputs/pi3_visymscenes/ckpts/best_model/pytorch_model.bin",
    map_location="cpu"
)

save_file(
    state_dict,
    "ckpts/model_epoch46.safetensors"
)
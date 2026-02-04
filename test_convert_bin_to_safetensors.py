import torch
from safetensors.torch import save_file, load_file

state_dict = torch.load(
    "outputs/pi3_visymscenes_classification/ckpts/best_model/pytorch_model.bin",
    map_location="cpu"
)

save_file(
    state_dict,
    "ckpts/model_pi3_visymscenes_classification.safetensors"
)
import torch
from safetensors.torch import save_file, load_file

state_dict = torch.load(
    "outputs/pi3_visymscenes_feature_align_first_img/ckpts/checkpoint_1/pytorch_model.bin",
    # "pytorch_model.bin",
    map_location="cpu"
)

save_file(
    state_dict,
    "ckpts/pi3_visymscenes_feature_align_first_img_512_epoch3.safetensors"
)
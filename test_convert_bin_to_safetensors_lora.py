import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from pi3.models.pi3_training_lora import Pi3


DEFAULT_INPUT = "outputs/pi3_visymscenes_feature_align_first_img/ckpts/checkpoint_10/pytorch_model.bin"
DEFAULT_OUTPUT = "ckpts/pi3_visymscenes_feature_align_first_img_lora_merged.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge decoder LoRA weights from pytorch_model.bin and export a normal safetensors checkpoint."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pytorch_model.bin path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .safetensors path.")
    parser.add_argument("--decoder-size", default="large", choices=["small", "base", "large"])
    parser.add_argument("--load-vggt", action="store_true", help="Load VGGT weights before loading the input state dict.")
    parser.add_argument("--use-global-points", action="store_true")
    parser.add_argument("--num-dec-blk-not-to-checkpoint", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=["attn.qkv", "attn.proj"])
    parser.add_argument("--adapter-name", default="decoder_lora")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading the state dict.")
    return parser.parse_args()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict or state dict, got {type(checkpoint)!r}.")

    return {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}


def strip_prefix_if_all_keys_match(state_dict, prefix):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def normalize_state_dict_keys(state_dict):
    for prefix in ("module.", "_orig_mod.", "model."):
        state_dict = strip_prefix_if_all_keys_match(state_dict, prefix)
    return state_dict


def build_model(args):
    return Pi3(
        decoder_size=args.decoder_size,
        load_vggt=args.load_vggt,
        freeze_encoder=True,
        freeze_decoder=False,
        use_global_points=args.use_global_points,
        train_conf=False,
        train_feature=True,
        num_dec_blk_not_to_checkpoint=args.num_dec_blk_not_to_checkpoint,
        ckpt=None,
        use_decoder_lora=True,
        decoder_lora_r=args.lora_r,
        decoder_lora_alpha=args.lora_alpha,
        decoder_lora_dropout=args.lora_dropout,
        decoder_lora_targets=tuple(args.lora_targets),
        decoder_lora_adapter_name=args.adapter_name,
    )


def print_load_result(load_result, max_items=20):
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    print(f"Missing keys: {len(missing_keys)}")
    for key in missing_keys[:max_items]:
        print(f"  {key}")

    print(f"Unexpected keys: {len(unexpected_keys)}")
    for key in unexpected_keys[:max_items]:
        print(f"  {key}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    checkpoint = torch.load(input_path, map_location="cpu")
    input_state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))

    if not any("lora_" in key for key in input_state_dict):
        print("Warning: no LoRA keys were found in the input checkpoint.")

    model = build_model(args)
    model.eval()

    load_result = model.load_state_dict(input_state_dict, strict=args.strict)
    print_load_result(load_result)

    merged_blocks = model.merge_decoder_lora(safe_merge=True)
    print(f"Merged decoder LoRA blocks: {merged_blocks}")

    state_dict = model.state_dict()
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
    }

    leftover_lora_keys = [
        key for key in state_dict
        if "lora_" in key or ".base_layer." in key
    ]
    if leftover_lora_keys:
        preview = "\n".join(f"  {key}" for key in leftover_lora_keys[:20])
        raise RuntimeError(f"LoRA keys still exist after merge:\n{preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path)
    print(f"Saved merged safetensors checkpoint to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge a LoRA-trained DiT checkpoint back into a plain, flat-keyed
safetensors file that the existing eval/visualize/inference scripts
(eval_dit_vrmse.py, visualize_dit_predictions.py, inference_dit.py) can
load unchanged.

Why this is needed: training with lora.enabled=true injects peft LoRA
layers into the transformer (see LtxvTrainer._apply_lora), which renames
each targeted nn.Linear's weight to "<module>.base_layer.weight" and adds
"<module>.lora_A.default.weight" / "lora_B.default.weight". Checkpoints
saved during LoRA training carry those keys. The eval/visualize/inference
scripts load checkpoints with `strict=False`, so pointed at a raw LoRA
checkpoint they would silently DROP the lora_A/lora_B deltas (unexpected
keys) and quietly evaluate the untouched base model instead of raising an
error. Always merge before handing a LoRA checkpoint to those scripts.

Usage:
    python scripts/merge_dit_lora.py \\
        --checkpoint outputs/my_lora_run/checkpoints/model_weights_step_05000.safetensors \\
        --output outputs/my_lora_run/checkpoints/model_weights_step_05000.merged.safetensors

--lora_config defaults to lora_config.json next to --checkpoint (written
automatically by LtxvTrainer._save_checkpoint during LoRA training).
--model_source defaults to the same pretrained base the training config
pointed at (windinet.inference.model_loader.LtxvModelVersion.latest());
override it if the run used a non-default model.model_source.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import inject_adapter_in_model
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file, save_file

from windinet.inference.model_loader import load_ltxv_components


def merge_and_unload(model: torch.nn.Module) -> torch.nn.Module:
    """Fold each injected LoRA adapter into its base_layer, then replace the
    peft wrapper module with the plain (now-merged) nn.Linear in its parent,
    restoring the original flat state_dict key names."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLinear):
            module.merge()
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.base_layer)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="LoRA-training checkpoint (.safetensors)")
    ap.add_argument("--output", type=Path, required=True, help="Where to write the merged, flat-keyed checkpoint")
    ap.add_argument("--lora_config", type=Path, default=None,
                     help="Path to lora_config.json (default: alongside --checkpoint)")
    ap.add_argument("--model_source", type=str, default=None,
                     help="Pretrained base model source (default: LtxvModelVersion.latest())")
    args = ap.parse_args()

    lora_config_path = args.lora_config or (args.checkpoint.parent / "lora_config.json")
    if not lora_config_path.exists():
        raise FileNotFoundError(
            f"No lora_config.json found at {lora_config_path}. Pass --lora_config explicitly, "
            "or confirm this checkpoint actually came from a lora.enabled=true training run."
        )
    lora_cfg = json.loads(lora_config_path.read_text())
    print(f"Loaded LoRA config from {lora_config_path}: {lora_cfg}")

    print(f"Loading base transformer (model_source={args.model_source or 'default'})...")
    components = load_ltxv_components(model_source=args.model_source, transformer_dtype=torch.float32)
    transformer = components.transformer

    peft_config = PeftLoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    inject_adapter_in_model(peft_config, transformer)

    print(f"Loading LoRA checkpoint: {args.checkpoint}")
    state_dict = load_file(str(args.checkpoint))
    transformer.load_state_dict(state_dict, strict=True)

    print("Merging LoRA adapters into base weights...")
    merge_and_unload(transformer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(transformer.state_dict(), args.output)
    print(f"Merged checkpoint written to {args.output}")
    print("This file has plain, flat keys -- use it with the existing eval/visualize/inference scripts unchanged.")


if __name__ == "__main__":
    main()

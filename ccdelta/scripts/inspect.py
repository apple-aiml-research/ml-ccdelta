# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

# helper script to inspect pipeline outputs
import click
import torch
import numpy as np


def load_state_dict(path: str):
    """Load state dict from a local path."""
    with open(path, "rb") as f:
        return torch.load(f, weights_only=False, map_location=torch.device("cpu"))  # nosemgrep


def preview_statedict(input_path):
    state_dict = load_state_dict(input_path)
    print(state_dict.keys())

    steering_indices = state_dict.get("steering_indices")
    mul_indices = state_dict.get("mul_indices")
    mul_values = state_dict.get("mul_values")
    add_values = state_dict.get("add_values")

    # Handle both old and new state dict formats
    sae_config_tensor = state_dict.get("sae_config_tensor")
    sae_url_tensor = state_dict.get("sae_url_tensor")  # Legacy support

    if sae_config_tensor is not None:
        import json

        config_str = "".join(chr(int(c)) for c in sae_config_tensor)
        sae_config = json.loads(config_str)
        print(f"sae_config: {sae_config}")
    elif sae_url_tensor is not None:
        sae_url = "".join(chr(int(c)) for c in sae_url_tensor)
        print(f"sae_url (legacy): {sae_url}")
    else:
        print("No SAE configuration found in state dict")

    print(f"steering_indices: {steering_indices}")
    print(f"mul_indices: {mul_indices}")
    print(f"mul_values: {mul_values}")
    print(f"add_values: {add_values}")


def compare_statedicts(path1: str, path2: str):
    """Compare two steering state dicts to check if they differ."""
    print("=" * 80)
    print(f"Comparing state dicts:")
    print(f"  A: {path1}")
    print(f"  B: {path2}")
    print("=" * 80)

    sd1 = load_state_dict(path1)
    sd2 = load_state_dict(path2)

    # Print all keys for debugging
    print(f"\n--- State Dict Keys ---")
    print(f"  A keys: {list(sd1.keys())}")
    print(f"  B keys: {list(sd2.keys())}")

    # Check for steering indices under various possible key names
    idx_keys = ["steering_indices", "mul_indices", "feature_indices", "indices"]
    idx1, idx2 = None, None
    idx_key_used = None

    for key in idx_keys:
        if key in sd1 and key in sd2:
            idx1 = sd1.get(key)
            idx2 = sd2.get(key)
            idx_key_used = key
            break

    if idx1 is not None and idx2 is not None:
        idx1_list = idx1.tolist() if hasattr(idx1, "tolist") else list(idx1)
        idx2_list = idx2.tolist() if hasattr(idx2, "tolist") else list(idx2)
        idx1_set = set(idx1_list)
        idx2_set = set(idx2_list)

        common = idx1_set & idx2_set
        only_in_a = idx1_set - idx2_set
        only_in_b = idx2_set - idx1_set

        print(f"\n--- Steering Indices (key: {idx_key_used}) ---")
        print(f"  A has {len(idx1_set)} indices, B has {len(idx2_set)} indices")
        print(f"  Common: {len(common)} ({100*len(common)/max(len(idx1_set),1):.1f}%)")

        # Print full indices if <= 25 features
        if len(idx1_list) <= 25:
            print(f"  A indices: {idx1_list}")
        if len(idx2_list) <= 25:
            print(f"  B indices: {idx2_list}")

        if only_in_a:
            print(f"  Only in A: {len(only_in_a)} -> {sorted(only_in_a)[:10]}{'...' if len(only_in_a) > 10 else ''}")
        if only_in_b:
            print(f"  Only in B: {len(only_in_b)} -> {sorted(only_in_b)[:10]}{'...' if len(only_in_b) > 10 else ''}")

        if idx1_set == idx2_set:
            print(f"  ✓ IDENTICAL feature selection")
        else:
            print(f"  ✗ DIFFERENT feature selection")
    else:
        print(f"\n  steering_indices not found in one or both state dicts")
        print(f"  (searched keys: {idx_keys})")

    # Compare add_values
    add1 = sd1.get("add_values")
    add2 = sd2.get("add_values")

    if add1 is not None and add2 is not None:
        print(f"\n--- Add Values ---")
        if add1.shape == add2.shape:
            diff = (add1 - add2).abs()
            print(f"  Shape: {add1.shape}")
            print(f"  A: norm={add1.norm().item():.4f}, mean={add1.mean().item():.6f}, std={add1.std().item():.6f}")
            print(f"  B: norm={add2.norm().item():.4f}, mean={add2.mean().item():.6f}, std={add2.std().item():.6f}")
            print(f"  Diff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

            # Print full values if <= 25 features
            if add1.numel() <= 25:
                print(f"  A values: {add1.tolist()}")
                print(f"  B values: {add2.tolist()}")

            if torch.allclose(add1, add2, atol=1e-6):
                print(f"  ✓ IDENTICAL add_values (within 1e-6)")
            else:
                print(f"  ✗ DIFFERENT add_values")
        else:
            print(f"  Different shapes: A={add1.shape}, B={add2.shape}")
    else:
        print("\n  add_values not found in one or both state dicts")

    # Compare mul_values
    mul1 = sd1.get("mul_values")
    mul2 = sd2.get("mul_values")

    if mul1 is not None and mul2 is not None:
        print(f"\n--- Mul Values ---")
        if mul1.shape == mul2.shape:
            diff = (mul1 - mul2).abs()
            print(f"  Shape: {mul1.shape}")
            print(f"  A: norm={mul1.norm().item():.4f}")
            print(f"  B: norm={mul2.norm().item():.4f}")
            print(f"  Diff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

            # Print full values if <= 25 features
            if mul1.numel() <= 25:
                print(f"  A values: {mul1.tolist()}")
                print(f"  B values: {mul2.tolist()}")

            if torch.allclose(mul1, mul2, atol=1e-6):
                print(f"  ✓ IDENTICAL mul_values")
            else:
                print(f"  ✗ DIFFERENT mul_values")
        else:
            print(f"  Different shapes: A={mul1.shape}, B={mul2.shape}")

    print("\n" + "=" * 80)


@click.command()
@click.option("--path", help="Path to single state dict to preview")
@click.option("--compare", nargs=2, help="Two paths to compare")
def main(path: str, compare: tuple):
    if compare:
        compare_statedicts(compare[0], compare[1])
    elif path:
        preview_statedict(path)
    else:
        print("Usage:")
        print("  Preview single:  python inspect.py --path <state_dict.pt>")
        print("  Compare two:     python inspect.py --compare <path1.pt> <path2.pt>")


# if main script
if __name__ == "__main__":
    main()

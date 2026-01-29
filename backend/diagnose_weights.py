"""
Diagnostic script - shows exactly what's in the weights file
Run this and send me the output
"""
import torch
import os

weights_path = os.path.join(os.path.dirname(__file__), "weights", "SHTechA.pth")

print("="*60)
print("P2PNet Weights Diagnostic")
print("="*60)

if not os.path.exists(weights_path):
    print(f"ERROR: Weights not found at {weights_path}")
    exit(1)

print(f"File: {weights_path}")
print(f"Size: {os.path.getsize(weights_path) / 1024 / 1024:.1f} MB")
print()

# Load checkpoint
checkpoint = torch.load(weights_path, map_location='cpu')

print("Top-level keys:", list(checkpoint.keys()))
print()

if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print(f"Number of tensors: {len(state_dict)}")
print()

# Group by prefix
prefixes = {}
for key in sorted(state_dict.keys()):
    prefix = key.split('.')[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

print("Key prefixes found:")
for prefix, keys in prefixes.items():
    print(f"  {prefix}: {len(keys)} keys")
print()

print("All keys with shapes:")
print("-"*60)
for key in sorted(state_dict.keys()):
    shape = list(state_dict[key].shape)
    print(f"  {key}: {shape}")

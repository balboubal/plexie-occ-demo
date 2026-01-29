"""
Debug script to check P2PNet weights structure
"""
import torch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "SHTechA.pth")

if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
    print("Keys in checkpoint:", checkpoint.keys())
    print("\nModel state dict keys:")
    for k in sorted(checkpoint['model'].keys()):
        print(f"  {k}: {checkpoint['model'][k].shape}")
else:
    print(f"Weights not found at: {WEIGHTS_PATH}")

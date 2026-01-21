# Test Dataloader Unification and Config Updates

import sys
import os
import yaml
from omegaconf import OmegaConf
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.getcwd())

from data.dvsgesture.dataset_precomputed import DVSGesturePrecomputed

def verify_configs():
    print("="*60)
    print("Verifying Configurations")
    print("="*60)
    
    datasets = ['dvsgesture', 'hmdb', 'ucf101']
    
    for ds in datasets:
        cfg_path = f'configs/config_{ds}.yaml'
        if not os.path.exists(cfg_path):
            print(f"❌ Missing config: {cfg_path}")
            continue
            
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        print(f"\nChecking {ds}:")
        
        # Check Positional Encoding Removal
        if 'use_position_encoding' in cfg['DATA']['dataset']['dataset_init_args']:
             print(f"  ❌ use_position_encoding STILL PRESENT in DATA.dataset...")
        else:
             print(f"  ✅ use_position_encoding removed from DATA.dataset")
             
        # Check Sampling
        sampling = cfg['PRECOMPUTING'].get('sampling', {})
        method = sampling.get('method', 'unknown')
        print(f"  Sampling method: {method}")
        
        if method == 'adaptive_striding':
            adaptive = sampling.get('adaptive_striding', {})
            print(f"  Adaptive settings: K={adaptive.get('kernel_size')}, Overlap={adaptive.get('overlap_factor')}")
            if adaptive.get('overlap_factor') is not None:
                print(f"  ✅ Adaptive config present")
            else:
                print(f"  ❌ Missing adaptive config block")

def verify_dataloader():
    print("\n" + "="*60)
    print("Verifying Dataloader (Mock)")
    print("="*60)
    
    # We can't load real data without running preprocessing first (which takes time)
    # But we can inspect the DVSGesturePrecomputed class signature and attributes
    
    import inspect
    sig = inspect.signature(DVSGesturePrecomputed.__init__)
    
    if 'use_position_encoding' in sig.parameters:
        print(f"❌ DVSGesturePrecomputed.__init__ still has 'use_position_encoding' parameter")
    else:
        print(f"✅ DVSGesturePrecomputed.__init__ CLEAN (no pos encoding arg)")

if __name__ == "__main__":
    verify_configs()
    verify_dataloader()

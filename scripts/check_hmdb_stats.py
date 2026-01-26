
import h5py
import os
import numpy as np

def check_h5_stats(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Checking {path}...")
    with h5py.File(path, 'r') as f:
        labels = f['labels'][:]
        print(f"  Total samples: {len(labels)}")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Class distribution stats:")
        print(f"    Min samples per class: {counts.min()}")
        print(f"    Max samples per class: {counts.max()}")
        print(f"    Mean samples per class: {counts.mean():.2f}")
        print(f"  Classes: {len(unique)}")

        # Check total vectors
        total_vectors = 0
        num_samples = len(labels)
        # check first 100 samples to estimate
        for i in range(min(100, num_samples)):
            grp = f[f'sample_{i:06d}']
            # Sum vectors across intervals
            # We need to look at intervals
            # But the structure is sample_XXX/interval_YYY
             # iterate keys
            for k in grp.keys():
                if k.startswith('interval_'):
                    total_vectors += grp[k]['real'].shape[0]
        
        print(f"  Avg vectors per sample (est from 100): {total_vectors / min(100, num_samples):.1f}")

dataset_dir = "/fs/nexus-projects/DVS_Actions/HMDB_DVS/HMDB_precomputed_data"
check_h5_stats(os.path.join(dataset_dir, "train.h5"))
check_h5_stats(os.path.join(dataset_dir, "validation.h5"))

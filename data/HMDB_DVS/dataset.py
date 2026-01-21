import torch.utils.data as data
import numpy as np
import os
import random
import sys
from pathlib import Path

# Import custom AEDAT2 reader
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from aedat2_reader import read_aedat2_with_dynamic_parsing

# HMDB-DVS action classes (51 classes)
HMDB_CLASSES = [
    "brush_hair",
    "cartwheel",
    "catch",
    "chew",
    "clap",
    "climb",
    "climb_stairs",
    "dive",
    "draw_sword",
    "dribble",
    "drink",
    "eat",
    "fall_floor",
    "fencing",
    "flic_flac",
    "golf",
    "handstand",
    "hit",
    "hug",
    "jump",
    "kick",
    "kick_ball",
    "kiss",
    "laugh",
    "pick",
    "pour",
    "pullup",
    "punch",
    "push",
    "pushup",
    "ride_bike",
    "ride_horse",
    "run",
    "shake_hands",
    "shoot_ball",
    "shoot_bow",
    "shoot_gun",
    "sit",
    "situp",
    "smile",
    "smoke",
    "somersault",
    "stand",
    "swing_baseball",
    "sword",
    "sword_exercise",
    "talk",
    "throw",
    "turn",
    "walk",
    "wave",
]

CLASS_NAME_TO_INT = {name: i for i, name in enumerate(HMDB_CLASSES)}
INT_TO_CLASS_NAME = {i: name for i, name in enumerate(HMDB_CLASSES)}


class HMDB_DVS(data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        purpose: str = 'train',
        height: int = 180,
        width: int = 240,
        use_flip_augmentation = False,
        accumulation_interval_ms = 100.0,
        train_split: float = 0.8,
    ):
        """
        HMDB-DVS Dataset Loader
        
        Args:
            dataset_dir: Path to HMDB_DVS dataset directory containing action class folders
            purpose: 'train', 'validation', or 'test'
            height: Sensor height (default: 180)
            width: Sensor width (default: 240)
            use_flip_augmentation: Whether to use random flip augmentation
            accumulation_interval_ms: Time interval for event accumulation in milliseconds
            train_split: Training data split ratio (default: 0.8, remaining 0.2 for validation)
        """
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.purpose = purpose
        self.height = height
        self.width = width
        self.use_flip_augmentation = use_flip_augmentation
        self.accumulation_interval_ms = accumulation_interval_ms
        self.train_split = train_split
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Collect all .aedat files organized by class
        self.all_seqs = []
        self.all_labels = []
        
        for class_idx, class_name in enumerate(HMDB_CLASSES):
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all .aedat files in this class directory
            aedat_files = sorted(class_dir.glob('*.aedat'))
            
            # Split into train/val based on file index
            num_files = len(aedat_files)
            num_train = int(num_files * train_split)
            
            if purpose == 'train':
                selected_files = aedat_files[:num_train]
            elif purpose == 'validation':
                selected_files = aedat_files[num_train:]
            elif purpose == 'all':
                selected_files = aedat_files
            else:
                raise ValueError(f"Unknown purpose: {purpose}")
            
            for file_path in selected_files:
                self.all_seqs.append(str(file_path))
                self.all_labels.append(class_idx)
        
        if len(self.all_seqs) == 0:
            raise ValueError(f"No .aedat files found in {self.dataset_dir}")

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        """Load and return raw events with class label"""
        file_path = self.all_seqs[idx]
        label = self.all_labels[idx]
        
        # Load events from .aedat file using custom reader
        # Returns [N, 4] array: [x, y, t, p]
        events = read_aedat2_with_dynamic_parsing(file_path, expected_width=self.width, expected_height=self.height)
        
        
        # All corrupted files have been filtered out during __init__,
        # so this should never be empty. If it is, raise an error.
        if len(events) == 0:
            raise ValueError(f"Unexpected empty events (should have been filtered): {file_path}")
        
        # Extract event components
        events_xy_all = events[:, 0:2]  # [N, 2] - x, y coordinates
        events_t_all = events[:, 2]     # [N] - timestamps (microseconds)
        events_p_all = events[:, 3]     # [N] - polarities (0 or 1)
        
        # Apply augmentation if enabled
        augmentation_method = ''
        if self.use_flip_augmentation:
            # Random horizontal flip
            rand_num_horizontal = random.random()
            if rand_num_horizontal <= 0.5:
                augmentation_method += "H"
            
            # Random vertical flip
            rand_num_vertical = random.random()
            if rand_num_vertical <= 0.5:
                augmentation_method += "V"
        
        if 'H' in augmentation_method:
            X_LIMIT = events_xy_all[:, 0].max()
            assert X_LIMIT <= self.width, f"X range does not match, given range: 0-{self.width}, actual range: 0-{X_LIMIT}"
            events_xy_all[:, 0] = self.width - events_xy_all[:, 0]
        
        if 'V' in augmentation_method:
            Y_LIMIT = events_xy_all[:, 1].max()
            assert Y_LIMIT <= self.height, f"Y range does not match, given range: 0-{self.height}, actual range: 0-{Y_LIMIT}"
            events_xy_all[:, 1] = self.height - events_xy_all[:, 1]
        
        # Slice events into time intervals
        cur_time = events_t_all[0]  # Start from first event timestamp
        t_end = events_t_all[-1]
        # Convert accumulation interval from ms to microseconds
        interval_us = self.accumulation_interval_ms * 1000.0
        
        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []
        
        while cur_time < t_end:
            new_time = cur_time + interval_us
            
            idx0 = np.searchsorted(events_t_all, cur_time, side='left')
            idx1 = np.searchsorted(events_t_all, new_time, side='left')
            
            if idx1 > idx0:
                events_p_sliced.append(events_p_all[idx0:idx1])
                events_t_sliced.append(events_t_all[idx0:idx1])
                events_xy_sliced.append(events_xy_all[idx0:idx1])
            
            cur_time = new_time
        
        return {
            'events_xy_sliced': events_xy_sliced,
            'events_t_sliced': events_t_sliced,
            'events_p_sliced': events_p_sliced,
            'label': label,
            'file_path': file_path,
            'augmentation_method': augmentation_method
        }


if __name__ == "__main__":
    # Test the dataloader
    dataset = HMDB_DVS(
        dataset_dir="~/Downloads/HMDB_DVS",
        purpose="train",
        height=180,
        width=240,
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {len(HMDB_CLASSES)}")
    
    # Get first item
    if len(dataset) > 0:
        sample = dataset[0]
        
        print(f"\nSample keys: {sample.keys()}")
        print(f"Label: {sample['label']} ({HMDB_CLASSES[sample['label']]})")
        print(f"File path: {sample['file_path']}")
        print(f"Number of time intervals: {len(sample['events_xy_sliced'])}")
        
        # Print statistics for first interval
        if len(sample['events_xy_sliced']) > 0:
            events_xy = sample['events_xy_sliced'][0]
            events_t = sample['events_t_sliced'][0]
            events_p = sample['events_p_sliced'][0]
            
            print(f"\nFirst interval statistics:")
            print(f"  Number of events: {len(events_t)}")
            if len(events_t) > 0:
                print(f"  X range: [{events_xy[:, 0].min():.1f}, {events_xy[:, 0].max():.1f}]")
                print(f"  Y range: [{events_xy[:, 1].min():.1f}, {events_xy[:, 1].max():.1f}]")
                print(f"  Time range: [{events_t.min():.3e}, {events_t.max():.3e}] (microseconds)")
                print(f"  Polarity distribution: ON={np.sum(events_p == 1)}, OFF={np.sum(events_p == 0)}")

import torch.utils.data as data
import numpy as np
import os
import random
import sys
from pathlib import Path
from scipy.io import loadmat

# UCF101-DVS action classes (101 classes)
UCF101_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow",
    "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo",
]

CLASS_NAME_TO_INT = {name: i for i, name in enumerate(UCF101_CLASSES)}
INT_TO_CLASS_NAME = {i: name for i, name in enumerate(UCF101_CLASSES)}


class UCF101_DVS(data.Dataset):
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
        UCF101-DVS Dataset Loader
        
        Args:
            dataset_dir: Path to UCF101_DVS dataset directory containing action class folders
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
        
        # Collect all .mat files organized by class
        self.all_seqs = []
        self.all_labels = []
        
        for class_idx, class_name in enumerate(UCF101_CLASSES):
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all .mat files and group by video (gXX)
            mat_files = sorted(class_dir.glob('*.mat'))
            
            # Group clips by video to avoid data leakage
            # Filename format: v_<ClassName>_g<XX>_c<YY>.mat
            from collections import defaultdict
            import re
            
            video_groups = defaultdict(list)
            for mat_file in mat_files:
                match = re.search(r'_g(\d+)_c(\d+)\.mat', mat_file.name)
                if match:
                    group_id = match.group(1)
                    video_groups[group_id].append(mat_file)
            
            # Sort video groups for consistent splits
            sorted_groups = sorted(video_groups.keys())
            
            # Split at video level (not clip level) to prevent data leakage
            num_videos = len(sorted_groups)
            num_train_videos = int(num_videos * train_split)
            
            if purpose == 'train':
                selected_groups = sorted_groups[:num_train_videos]
            elif purpose == 'validation':
                selected_groups = sorted_groups[num_train_videos:]
            elif purpose == 'all':
                selected_groups = sorted_groups
            else:
                raise ValueError(f"Unknown purpose: {purpose}")
            
            # PRE-SCAN: Detect corrupted video sequences
            # If any clip from a video sequence is corrupted,exclude the entire sequence
            corrupted_videos = set()
            for group_id in selected_groups:
                for file_path in video_groups[group_id]:
                    try:
                        # Quick check: load and verify file has events
                        mat_data = loadmat(str(file_path))
                        x = mat_data.get('x', np.array([]))
                        if len(x) == 0:
                            # Empty file - mark entire video sequence as corrupted
                            corrupted_videos.add(group_id)
                            print(f"⚠️  Excluding video sequence g{group_id} from {class_name} (empty file: {file_path.name})")
                            break  # No need to check other clips from this video
                    except Exception as e:
                        # File loading failed - mark entire video sequence as corrupted
                        corrupted_videos.add(group_id)
                        print(f"⚠️  Excluding video sequence g{group_id} from {class_name} (load error: {file_path.name})")
                        break
            
            # Add clips only from non-corrupted video sequences
            for group_id in selected_groups:
                if group_id in corrupted_videos:
                    continue  # Skip entire corrupted video sequence
                for file_path in video_groups[group_id]:
                    self.all_seqs.append(str(file_path))
                    self.all_labels.append(class_idx)
        
        if len(self.all_seqs) == 0:
            raise ValueError(f"No .mat files found in {self.dataset_dir}")

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        """Load and return raw events with class label"""
        file_path = self.all_seqs[idx]
        label = self.all_labels[idx]
        
        # Load events from .mat file
        # Format: {'x', 'y', 'ts', 'pol'} all as [N, 1] arrays
        mat_data = loadmat(file_path)
        
        # Extract and flatten to 1D arrays
        x = mat_data['x'].flatten().astype(np.float32)
        y = mat_data['y'].flatten().astype(np.float32)
        ts = mat_data['ts'].flatten().astype(np.float32)  # timestamps in microseconds
        pol = mat_data['pol'].flatten().astype(np.float32)  # polarity (0 or 1)
        
        
        # All corrupted files have been filtered out during __init__,
        # so this should never be empty. If it is, raise an error.
        if len(x) == 0:
            raise ValueError(f"Unexpected empty events (should have been filtered): {file_path}")
        
        # Combine into [N, 2] for xy coordinates
        events_xy_all = np.stack([x, y], axis=1)
        events_t_all = ts
        events_p_all = pol
        
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
    dataset = UCF101_DVS(
        dataset_dir="~/Downloads/UCF101_DVS",
        purpose="train",
        height=180,
        width=240,
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {len(UCF101_CLASSES)}")
    
    # Get first item
    if len(dataset) > 0:
        sample = dataset[0]
        
        print(f"\nSample keys: {sample.keys()}")
        print(f"Label: {sample['label']} ({UCF101_CLASSES[sample['label']]})")
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

import torch.utils.data as data
import numpy as np
import os
import random

GESTURE_CLASSES = [
    "hand_clapping",
    "right_hand_wave",
    "left_hand_wave",
    "right_arm_clockwise",
    "right_arm_counter_clockwise",
    "left_arm_clockwise",
    "left_arm_counter_clockwise",
    "arm_roll",
    "air_drums",
    "air_guitar",
    "other_gestures"
]

CLASS_NAME_TO_INT = {name: i for i, name in enumerate(GESTURE_CLASSES)}
INT_TO_CLASS_NAME = {i: name for i, name in enumerate(GESTURE_CLASSES)}


class DVSGesture(data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        purpose: str = 'train',
        height: int = 128,
        width: int = 128,
        use_flip_augmentation = False,
        accumulation_interval_ms = 100.0,
    ):
        """
        DVS Gesture Dataset Loader

        Args:
            dataset_dir: Path to dataset directory containing user folders
            purpose: 'train', 'validation', or 'test'
            train_split: Training data split ratio
            val_split: Validation data split ratio
        """
        self.dataset_dir = dataset_dir
        self.purpose = purpose
        self.height = height
        self.width = width
        self.use_flip_augmentation = use_flip_augmentation
        self.accumulation_interval_ms = accumulation_interval_ms

        if purpose == 'train':
            self.root = os.path.join(dataset_dir, 'ibmGestureTrain')
        else:
            self.root = os.path.join(dataset_dir, 'ibmGestureTest')

        self.all_seqs = []
        self.all_labels = []
        for user_folder in os.listdir(self.root):
            for i in range(11):
                target_path = os.path.join(self.root, user_folder, f'{i}.npy')
                if os.path.exists(target_path):
                    self.all_seqs.append(target_path)
                    self.all_labels.append(i)

        if purpose == 'all':
            for user_folder in os.listdir(os.path.join(dataset_dir, 'ibmGestureTrain')):
                for i in range(11):
                    target_path = os.path.join(os.path.join(dataset_dir, 'ibmGestureTrain'), user_folder, f'{i}.npy')
                    if os.path.exists(target_path):
                        self.all_seqs.append(target_path)
                        self.all_labels.append(i)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        """Load and return raw events with class label"""
        file_path = self.all_seqs[idx]
        label = self.all_labels[idx]

        # Load events from .npy file
        # Expected format: [x, y, polarity, timestamp]
        events = np.load(file_path).astype(np.float32)

        if len(events) == 0:
            raise ValueError(f"Found empty events at path: {file_path}")

        # the raw events are int x,y,p,t order,
        events_xy_all = events[:, 0:2]
        events_t_all = events[:, 3]
        events_p_all = events[:, 2]
    
        augmentation_method = ''
        if self.use_flip_augmentation:
            # Augmentation. If rand_num_horizontal <= 0.5 then flip events horizontally
            rand_num_horizontal = random.random()
            if rand_num_horizontal <= 0.5:
                augmentation_method += "H"

            # Augmentation. If rand_num_vertical <= 0.5 then flip events vertically
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

        cur_time = 0.0
        t_end = events_t_all[-1]
        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []
        while cur_time < t_end:
            new_time = cur_time + self.accumulation_interval_ms

            idx0 = np.searchsorted(events_t_all, cur_time, side='left')
            idx1 = np.searchsorted(events_t_all, new_time, side='left')

            if idx1 > idx0:
                events_p_sliced.append(events_p_all[idx0: idx1])
                events_t_sliced.append(events_t_all[idx0: idx1])
                events_xy_sliced.append(events_xy_all[idx0: idx1])
            
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
    dataset = DVSGesture(
        dataset_dir="/home/matt/DATA/DVSGesture/ibmGestureTrain/",
        purpose="train"
    )

    print(f"Dataset length: {len(dataset)}")

    # Get first item
    events, label = dataset[0]

    print(f"Events shape: {events.shape}")
    print(f"Events dtype: {events.dtype}")
    print(f"Label: {label} ({GESTURE_CLASSES[label]})")
    print(f"First 5 events:\n{events[:5]}")
    print(f"Event ranges - X: [{events[:, 0].min():.1f}, {events[:, 0].max():.1f}], Y: [{events[:, 1].min():.1f}, {events[:, 1].max():.1f}], Time: [{events[:, 3].min():.3f}, {events[:, 3].max():.3f}]")
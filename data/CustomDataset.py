from torch.utils.data import Dataset

import os
import numpy as np
import random
from tqdm import tqdm

import json

CLASS_NAME_TO_INT = {
    "knife_bread": 1,
    "knife_cleaver": 2,
    "knife_paring": 3,
    "knife_steak": 4,
    "ladle": 5,
    "spatula": 6,
    "spoon": 7,
    "ice_cream_scoop": 8,
    "pasta_server": 9,
    "skimmer": 10,
    "bottle": 11,
    "champagne_glass": 12,
    "espresso": 13,
    "coffee_mug": 14,
    "shot": 15,
    "wine_glass": 16,
}

INT_TO_CLASS_NAME = {
    1: "knife_bread",
    2: "knife_cleaver",
    3: "knife_paring",
    4: "knife_steak",
    5: "ladle",
    6: "spatula",
    7: "spoon",
    8: "ice_cream_scoop",
    9: "pasta_server",
    10: "skimmer",
    11: "bottle",
    12: "champagne_glass",
    13: "espresso",
    14: "coffee_mug",
    15: "shot",
    16: "wine_glass",
}


class Preprocessor(Dataset):
    def __init__(self, 
                dataset_dir,
                height,
                width, 
                accumulation_interval_ms=100, 
                use_flip_augmentation=True,
                use_RGB=False,
                use_events=True,
                purpose='train',
                val_person=None  # ← allow manual control if desired
            ):
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.accumulation_interval_ms = accumulation_interval_ms
        self.use_RGB = use_RGB
        self.use_events = use_events
        self.purpose = purpose
        self.use_flip_augmentation = use_flip_augmentation

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
        
        all_sequences = os.listdir(dataset_dir)

        # --- GROUP BY PERSON ID ---
        person_to_sequences = {}
        for seq in all_sequences:
            # sequence_matt2_SIDE_STATIC_DARK_knife cleaver
            _, person_id, *_ = seq.split("_")
            if person_id not in person_to_sequences:
                person_to_sequences[person_id] = []
            person_to_sequences[person_id].append(seq)

        all_persons = sorted(person_to_sequences.keys())  # 8 of them
        
        # --- PICK validation person ---
        if val_person is None:
            # deterministic default: always take last person for val
            val_person = all_persons[-1]

        val_sequences = person_to_sequences[val_person]
        train_sequences = []
        for p, seqs in person_to_sequences.items():
            if p != val_person:
                train_sequences.extend(seqs)

        # --- Assign according to purpose ---
        if purpose == 'train':
            self.sequence_list = train_sequences
        else:
            self.sequence_list = val_sequences


    def get_sequence_folder(self, idx):
        return os.path.join(self.dataset_dir, self.sequence_list[idx], 'proc')

    def get_class_name(self, idx):
        sequence_name = self.sequence_list[idx]
        _, people_id, view_direction, background, lighting, *class_name_parts = sequence_name.split("_")
        class_name = "_".join(class_name_parts)

        return class_name 

    def get_sliced_labels(self, idx):
        boundaries_path = os.path.join(self.get_sequence_folder(idx), 'boundaries.json')
        class_name = self.get_class_name(idx)

        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)

        background_first = boundaries[0]['name'] == 'background'
        split_time_us = boundaries[0]['proph_end_time'] * 1e6

        events_t_path = os.path.join(self.get_sequence_folder[idx], 'events', 'events_t.npy')
        raw_t = np.load(events_t_path)
        raw_t = np.nan_to_num(raw_t, posinf=None, neginf=None)  # replace NaN with 0, +inf stays, -inf becomes MAXNEG
        events_t_all = np.round(raw_t).astype(np.int64)

        # slicing by accumulation interval
        interval_us = self.accumulation_interval_ms * 1000.0
        t_start = events_t_all[0]
        t_end   = events_t_all[-1]


        label_sliced = []

        cur_t0 = t_start

        while cur_t0 < t_end:
            cur_t1 = cur_t0 + interval_us
            idx0 = np.searchsorted(events_t_all, cur_t0, side='left')
            idx1 = np.searchsorted(events_t_all, cur_t1, side='left')

            if idx1 > idx0:  # non-empty slice
                if cur_t1 >= split_time_us:
                    label_sliced.append(0 if background_first else CLASS_NAME_TO_INT[class_name])
                else:
                    label_sliced.append(CLASS_NAME_TO_INT[class_name] if background_first else 0)

            cur_t0 = cur_t1  # move to next interval

        return np.array(label_sliced)
        
    
    def __len__(self):
        if self.purpose == 'train':
            return len(self.sequence_list)
        elif self.purpose == 'validation':
            return len(self.sequence_list)

    def __getitem__(self, idx):
        sequence_name = self.sequence_list[idx]
        _, people_id, view_direction, background, lighting, *class_name_parts = sequence_name.split("_")
        class_name = "_".join(class_name_parts)

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

        sequence_folder = os.path.join(self.dataset_dir, sequence_name, 'proc')
        boundaries_path = os.path.join(sequence_folder, 'boundaries.json')

        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)

        background_first = boundaries[0]['name'] == 'background'
        split_time_us = boundaries[0]['proph_end_time'] * 1e6

        return_dict = {
            "view_direction": view_direction.lower(),
            "background": background.lower(),
            "lighting": lighting.lower(),
            "sequence_folder": sequence_folder,
            "class_name": class_name,
            "augmentation_method": augmentation_method
        }
        
        if self.use_events:
            events_p_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_p.npy')
            events_t_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_t.npy')
            events_xy_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_xy.npy')

            events_p_all = np.load(events_p_path).astype(np.uint8)
            raw_t = np.load(events_t_path)
            raw_t = np.nan_to_num(raw_t, posinf=None, neginf=None)  # replace NaN with 0, +inf stays, -inf becomes MAXNEG
            events_t_all = np.round(raw_t).astype(np.int64)
            events_t_all = np.clip(events_t_all, a_min=0, a_max=None)
            events_xy_all = np.load(events_xy_path).astype(np.uint16)

            # Ensure time array is sorted (should already be)
            sort_idx = np.argsort(events_t_all)
            events_t_all = events_t_all[sort_idx]
            events_xy_all = events_xy_all[sort_idx]
            events_p_all = events_p_all[sort_idx]

            # Clip the event coordinates to required range
            events_xy_all[:, 0] = np.clip(events_xy_all[:, 0], 0, self.width - 1)
            events_xy_all[:, 1] = np.clip(events_xy_all[:, 1], 0, self.height - 1)
                
            if 'H' in augmentation_method:
                X_LIMIT = events_xy_all[:, 0].max()
                assert X_LIMIT <= self.width, f"X range does not match, given range: 0-{self.width}, actual range: 0-{X_LIMIT}"
                events_xy_all[:, 0] = self.width - events_xy_all[:, 0]

               
            if 'V' in augmentation_method:
                Y_LIMIT = events_xy_all[:, 1].max()
                assert Y_LIMIT <= self.height, f"Y range does not match, given range: 0-{self.height}, actual range: 0-{Y_LIMIT}"
                events_xy_all[:, 1] = self.height - events_xy_all[:, 1]

            events_p_sliced = []
            events_t_sliced = []
            events_xy_sliced = []

            # slicing by accumulation interval
            interval_us = self.accumulation_interval_ms * 1000.0
            t_start = events_t_all[0]
            t_end   = events_t_all[-1]

            events_p_sliced = []
            events_t_sliced = []
            events_xy_sliced = []
            label_sliced = []

            cur_t0 = t_start

            while cur_t0 < t_end:
                cur_t1 = cur_t0 + interval_us
                idx0 = np.searchsorted(events_t_all, cur_t0, side='left')
                idx1 = np.searchsorted(events_t_all, cur_t1, side='left')

                if idx1 > idx0:  # non-empty slice
                    events_p_sliced.append(events_p_all[idx0:idx1])
                    events_t_sliced.append(events_t_all[idx0:idx1])
                    events_xy_sliced.append(events_xy_all[idx0:idx1])
                    if cur_t1 >= split_time_us:
                        label_sliced.append(0 if background_first else CLASS_NAME_TO_INT[class_name])
                    else:
                        label_sliced.append(CLASS_NAME_TO_INT[class_name] if background_first else 0)

                cur_t0 = cur_t1  # move to next interval

            return_dict['events_t_sliced'] = events_t_sliced
            return_dict['events_xy_sliced'] = events_xy_sliced
            return_dict['events_p_sliced'] = events_p_sliced
            return_dict['label_sliced'] = np.array(label_sliced)

        if self.use_RGB:
            frame_folder = os.path.join(self.dataset_dir, sequence_name, 'proc', 'flir', 'frame')
            frame_files = sorted(os.listdir(frame_folder))  # ensure temporal order
            frame_t_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'flir', 'flir_t.npy')

            # Load frame timestamps (µs, same unit as events)
            frame_t = np.load(frame_t_path).astype(np.float64)

            # Load all frames into a list
            all_frames = [np.load(os.path.join(frame_folder, f)) for f in frame_files]

            # --- Apply augmentation consistent with events ---
            if 'H' in augmentation_method:
                all_frames = [np.flip(frame, axis=1) for frame in all_frames]  # flip horizontally (width axis)

            if 'V' in augmentation_method:
                all_frames = [np.flip(frame, axis=0) for frame in all_frames]  # flip vertically (height axis)

            # If not using events, just return a single list of RGB frames
            if not self.use_events:
                return_dict['frames_all'] = all_frames
                return_dict['frames_t'] = frame_t
                return return_dict

            # If using events, slice frames by accumulation_interval_ms
            interval_us = self.accumulation_interval_ms * 1000.0
            sliced_frames = []
            sliced_frame_t = []

            cur_t0 = events_t_all[0]
            t_end = events_t_all[-1]

            while cur_t0 < t_end:
                cur_t1 = cur_t0 + interval_us

                # find all frames with timestamp inside [cur_t0, cur_t1)
                mask = (frame_t >= cur_t0) & (frame_t < cur_t1)
                frames_in_interval = [all_frames[i] for i in np.where(mask)[0]]

                if len(frames_in_interval) > 0:
                    sliced_frames.append(frames_in_interval)
                    sliced_frame_t.append(frame_t[mask])

                cur_t0 = cur_t1

            return_dict['frames_all'] = sliced_frames
            return_dict['frames_t'] = sliced_frame_t


        return return_dict


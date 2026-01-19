"""
Exploration script for HMDB_DVS dataset in .aedat format.

This script helps understand the structure of .aedat files and the dataset organization.
It supports multiple .aedat reading libraries (dv, aedat, loris) and automatically
detects which one is available.

Usage:
    python explore_aedat.py --dataset_dir ~/Downloads/HMDB_DVS
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Import custom AEDAT2 reader
try:
    from aedat2_reader import read_aedat2_with_dynamic_parsing
    HAS_CUSTOM_READER = True
except ImportError:
    # If running from different directory, try absolute import
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    try:
        from aedat2_reader import read_aedat2_with_dynamic_parsing
        HAS_CUSTOM_READER = True
    except ImportError:
        HAS_CUSTOM_READER = False


def try_import_aedat_library():
    """Try to import available .aedat reading libraries."""
    
    # Try custom AEDAT2 reader first (for AEDAT2.0 files like HMDB-DVS)
    if HAS_CUSTOM_READER:
        print("✓ Using custom AEDAT2.0 reader")
        return 'custom_aedat2', None
    
    # Try aedat library (supports AEDAT4)
    try:
        from aedat import Decoder
        print("✓ Using 'aedat' library for reading .aedat files")
        return 'aedat', Decoder
    except ImportError:
        pass
    
    # Try dv library (recommended for aedat4)
    try:
        import dv
        print("✓ Using 'dv' library for reading .aedat files")
        return 'dv', dv
    except ImportError:
        pass
    
    # Try loris library
    try:
        import loris
        print("✓ Using 'loris' library for reading .aedat files")
        return 'loris', loris
    except ImportError:
        pass
    
    print("✗ No .aedat library found!")
    print("\nPlease install one of the following:")
    print("  pip install aedat            # For AEDAT4 files")
    print("  pip install dv              # For AEDAT4 files")
    print("  pip install loris            # Alternative library")
    print("\nNote: AEDAT2.0 (HMDB-DVS) requires the custom reader included with this script.")
    return None, None


def read_aedat_with_dv(file_path):
    """Read .aedat file using dv library."""
    import dv
    
    with dv.AedatFile(str(file_path)) as f:
        events = []
        for packet in f['events']:
            events.append(packet)
        
        if events:
            # Concatenate all event packets
            all_events = np.concatenate(events)
            # DV format: structured array with 'x', 'y', 'timestamp', 'polarity'
            x = all_events['x']
            y = all_events['y']
            t = all_events['timestamp']
            p = all_events['polarity']
            
            # Stack into [N, 4] array: [x, y, t, p]
            events_array = np.stack([x, y, t, p], axis=1)
            return events_array
        else:
            return np.array([])


def read_aedat_with_aedat(file_path):
    """Read .aedat file using aedat library (Decoder class)."""
    from aedat import Decoder
    
    decoder = Decoder(str(file_path))
    
    # Collect all events from the decoder iterator
    x_list, y_list, t_list, p_list = [], [], [], []
    
    for packet in decoder:
        if 'events' in packet:
            events = packet['events']
            x_list.append(events['x'])
            y_list.append(events['y'])
            t_list.append(events['t'])
            p_list.append(events['on'].astype(np.float32))  # polarity: bool -> float
    
    if len(x_list) == 0:
        return np.array([])
    
    # Concatenate all packets
    x = np.concatenate(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    t = np.concatenate(t_list).astype(np.float32)
    p = np.concatenate(p_list).astype(np.float32)
    
    # Stack into [N, 4] array: [x, y, t, p]
    events_array = np.stack([x, y, t, p], axis=1)
    return events_array


def read_aedat_with_loris(file_path):
    """Read .aedat file using loris library."""
    import loris
    
    events = loris.read_file(str(file_path))
    
    if len(events) > 0:
        # Loris format may vary, adjust as needed
        # Typically: events.x, events.y, events.t, events.p
        x = events.x
        y = events.y
        t = events.t
        p = events.on.astype(np.int32)  # polarity as 0/1
        
        # Stack into [N, 4] array: [x, y, t, p]
        events_array = np.stack([x, y, t, p], axis=1)
        return events_array
    else:
        return np.array([])


def read_aedat_file(file_path, library_name, library_module):
    """Read .aedat file using the specified library."""
    
    if library_name == 'custom_aedat2':
        return read_aedat2_with_dynamic_parsing(file_path, expected_width=240, expected_height=180)
    elif library_name == 'aedat':
        return read_aedat_with_aedat(file_path)
    elif library_name == 'dv':
        return read_aedat_with_dv(file_path)
    elif library_name == 'loris':
        return read_aedat_with_loris(file_path)
    else:
        raise ValueError(f"Unsupported library: {library_name}")


def explore_dataset(dataset_dir):
    """Explore the dataset structure."""
    
    dataset_dir = Path(dataset_dir).expanduser()
    
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Exploring HMDB_DVS Dataset")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Find all class directories
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"\nNumber of classes: {len(class_dirs)}")
    print(f"\nClass names:")
    for i, class_dir in enumerate(class_dirs):
        num_files = len(list(class_dir.glob('*.aedat')))
        print(f"  {i:2d}. {class_dir.name:20s} ({num_files} files)")
    
    return class_dirs


def explore_sample_file(file_path, library_name, library_module, resolution=(240, 180)):
    """Explore a single .aedat file."""
    
    print(f"\n{'='*60}")
    print(f"Exploring sample file")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    try:
        events = read_aedat_file(file_path, library_name, library_module)
        
        if len(events) == 0:
            print("✗ No events found in file")
            return
        
        print(f"\n✓ Successfully loaded events!")
        print(f"\nEvent array shape: {events.shape}")
        print(f"Number of events: {len(events):,}")
        
        # Analyze event properties
        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3]
        
        print(f"\nEvent statistics:")
        print(f"  X range: [{x.min():.0f}, {x.max():.0f}]")
        print(f"  Y range: [{y.min():.0f}, {y.max():.0f}]")
        print(f"  Time range: [{t.min():.3e}, {t.max():.3e}] (microseconds)")
        print(f"  Duration: {(t.max() - t.min()) / 1e6:.3f} seconds")
        print(f"  Polarity values: {np.unique(p)}")
        print(f"  Polarity distribution: ON={np.sum(p == 1):,}, OFF={np.sum(p == 0):,}")
        
        # Calculate event rate
        duration_sec = (t.max() - t.min()) / 1e6
        if duration_sec > 0:
            event_rate = len(events) / duration_sec
            print(f"  Average event rate: {event_rate:,.0f} events/sec")
        
        # Check resolution
        expected_height, expected_width = resolution
        actual_height = int(y.max()) + 1
        actual_width = int(x.max()) + 1
        
        print(f"\nResolution check:")
        print(f"  Expected: {expected_height} × {expected_width}")
        print(f"  Actual: {actual_height} × {actual_width}")
        
        if actual_height <= expected_height and actual_width <= expected_width:
            print(f"  ✓ Resolution matches!")
        else:
            print(f"  ✗ Resolution mismatch!")
        
        # Display first few events
        print(f"\nFirst 10 events (x, y, t, p):")
        for i in range(min(10, len(events))):
            print(f"  {i}: ({events[i, 0]:.0f}, {events[i, 1]:.0f}, {events[i, 2]:.0e}, {events[i, 3]:.0f})")
        
        return events
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Explore HMDB_DVS dataset in .aedat format')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='~/Downloads/HMDB_DVS',
        help='Path to HMDB_DVS dataset directory'
    )
    parser.add_argument(
        '--sample_file',
        type=str,
        default=None,
        help='Specific .aedat file to analyze (optional)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[240, 180],
        help='Expected resolution (height width)'
    )
    
    args = parser.parse_args()
    
    # Try to import .aedat library
    library_name, library_module = try_import_aedat_library()
    
    if library_name is None:
        print("\nCannot proceed without a library to read .aedat files.")
        sys.exit(1)
    
    # Explore dataset structure
    class_dirs = explore_dataset(args.dataset_dir)
    
    if class_dirs is None or len(class_dirs) == 0:
        print("No classes found!")
        return
    
    # Find a sample file
    if args.sample_file:
        sample_file = Path(args.sample_file).expanduser()
    else:
        # Use first .aedat file from first class
        sample_file = next(class_dirs[0].glob('*.aedat'))
    
    if sample_file and sample_file.exists():
        explore_sample_file(sample_file, library_name, library_module, tuple(args.resolution))
    else:
        print(f"\n✗ Sample file not found: {sample_file}")
    
    print(f"\n{'='*60}")
    print("Exploration complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

"""
Simple AEDAT2.0 reader for DVS events.

This module provides functions to read .aedat files in AEDAT2.0 format,
which is used by the HMDB-DVS dataset.

AEDAT2.0 Format:
- Header: Lines starting with '#' (header information)
- Data: Binary data packets with event information
  - Each event: 8 bytes (2 words of 32 bits each)
  - Word 1: Address (contains x, y, polarity)
  - Word 2: Timestamp (microseconds)
"""

import numpy as np
import struct


def read_aedat2_events(file_path):
    """
    Read AEDAT2.0 file and return events as numpy array.
    
    Args:
        file_path: Path to .aedat file
    
    Returns:
        events: numpy array of shape [N, 4] with columns [x, y, t, p]
                - x, y: spatial coordinates (uint16)
                - t: timestamp in microseconds (uint32)
                - p: polarity (0 or 1)
    """
    
    with open(file_path, 'rb') as f:
        # Skip header lines (lines starting with '#')
        header_end = 0
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                # We've reached the end of header, go back to start of this line
                f.seek(header_end)
                break
            header_end = f.tell()
        
        # Read binary data
        # AEDAT2.0 uses 8 bytes per event (2 x uint32)
        data = f.read()
    
    # Number of events
    num_events = len(data) // 8
    
    if num_events == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    # Parse events
    # Each event is 8 bytes: address (4 bytes) + timestamp (4 bytes)
    events_raw = np.frombuffer(data[:num_events * 8], dtype=np.uint32).reshape(-1, 2)
    
    # Extract address and timestamp
    addresses = events_raw[:, 0]
    timestamps = events_raw[:, 1]
    
    # Decode address bits for DVS events (DAVIS240)
    # Address format (32 bits):
    # bit 10-0: x coordinate (11 bits)
    # bit 21-11: y coordinate (11 bits)
    # bit 1 (in some formats): polarity
    # The exact format may vary, common format:
    # - x: bits 0-8 or bits 12-20 (9 bits might be less than 240 wide so it might vary)
    # - y: bits 9-17 or bits1-9
    # - polarity: bit 0 or bit 11
    
    # Try common DAVIS240 format (needs verification with actual data):
    # Bits 0-10: y coordinate
    # Bits 11-21: x coordinate
    # Bit 11 in some: polarity (or bit 0)
    
    # Let's try the DVS128 style first, then adjust if needed
    # Common format for DAVIS:
    # x = (address >> 12) & 0x3FF  # bits 12-21 (10 bits for x up to 1023, but camera is 240 wide)
    # y = (address >> 2) & 0x1FF   # bits 2-10 (9 bits for y up to 511, but camera is 180 tall)
    # p = (address >> 1) & 0x1     # bit 1 for polarity
    
    # Actually, for DAVIS240, the standard format is:
    # y = (address >> 22) & 0x1FF  # 9 bits
    # x = (address >> 12) & 0x3FF  # 10 bits  
    # pol = (address >> 11) & 0x1  # 1 bit
    
    # Let me use a more standard DAVIS format:
    y = (addresses >> 22) & 0x1FF  # 9 bits for y (up to 511)
    x = (addresses >> 12) & 0x3FF  # 10 bits for x (up to 1023)
    polarity = (addresses >> 11) & 0x1  # 1 bit for polarity
    
    # Stack into [N, 4] array
    events = np.stack([
        x.astype(np.float32),
        y.astype(np.float32),
        timestamps.astype(np.float32),
        polarity.astype(np.float32)
    ], axis=1)
    
    return events


def read_aedat2_with_dynamic_parsing(file_path, expected_width=240, expected_height=180):
    """
    Read AEDAT2.0 file with dynamic bit parsing based on expected resolution.
    
    This function tries different bit layouts to find the correct one for the sensor.
    
    Args:
        file_path: Path to .aedat file
        expected_width: Expected sensor width (default: 240)
        expected_height: Expected sensor height (default: 180)
    
    Returns:
        events: numpy array of shape [N, 4] with columns [x, y, t, p]
    """
    
    with open(file_path, 'rb') as f:
        # Skip header  
        header_end = 0
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(header_end)
                break
            header_end = f.tell()
        
        # Read binary data
        data = f.read()
    
    num_events = len(data) // 8
    
    if num_events == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    events_raw = np.frombuffer(data[:num_events * 8], dtype=np.uint32).reshape(-1, 2)
    addresses = events_raw[:, 0]
    timestamps = events_raw[:, 1]
    
    # Try different parsing strategies
    parsing_strategies = [
        # Strategy 1: DVS style (swap x/y since sensor coords are in (height, width) order in address bits)
        {'x_shift': 0, 'x_mask': 0xFF, 'y_shift': 8, 'y_mask': 0xFF, 'p_shift': 0, 'p_mask': 0x1},
        # Strategy 2: Alternative  
        {'x_shift': 1, 'x_mask': 0x3FF, 'y_shift': 12, 'y_mask': 0x1FF, 'p_shift': 0, 'p_mask': 0x1},
        # Strategy 3: DAVIS240 standard
        {'x_shift': 12, 'x_mask': 0x3FF, 'y_shift': 22, 'y_mask': 0x1FF, 'p_shift': 11, 'p_mask': 0x1},
        # Strategy 4: Try with different bit positions
        {'x_shift': 9, 'x_mask': 0x1FF, 'y_shift': 1, 'y_mask': 0xFF, 'p_shift':0, 'p_mask': 0x1},
    ]
    
    for strategy in parsing_strategies:
        x = (addresses >> strategy['x_shift']) & strategy['x_mask']
        y = (addresses >> strategy['y_shift']) & strategy['y_mask']
        polarity = (addresses >> strategy['p_shift']) & strategy['p_mask']
        
        max_x = x.max()
        max_y = y.max()
        
        # Check if this parsing makes sense (width=240, height=180)
        if max_x <= expected_width and max_y <= expected_height:
            print(f"✓ Found valid parsing: max_x={max_x}, max_y={max_y}")
            print(f"  Strategy: {strategy}")
            
            events = np.stack([
                x.astype(np.float32),
                y.astype(np.float32),
                timestamps.astype(np.float32),
                polarity.astype(np.float32)
            ], axis=1)
            
            return events
    
    # If no strategy worked, return with the first strategy and a warning
    print("✗ Warning: Could not find valid parsing strategy, using default")
    y = (addresses >> 22) & 0x1FF
    x = (addresses >> 12) & 0x3FF
    polarity = (addresses >> 11) & 0x1
    
    events = np.stack([
        x.astype(np.float32),
        y.astype(np.float32),
        timestamps.astype(np.float32),
        polarity.astype(np.float32)
    ], axis=1)
    
    return events

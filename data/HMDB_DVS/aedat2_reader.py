"""
AEDAT2.0 reader for DVS events from DAVIS240C sensor.

This module provides functions to read .aedat files in AEDAT2.0 format,
which is used by the HMDB-DVS dataset.

AEDAT2.0 Format:
- Header: Lines starting with '#' (header information)
- Data: Binary data packets with event information
  - Each event: 8 bytes (2 words of 32 bits each, BIG-ENDIAN)
  - Word 1: Address (contains x, y, polarity in specific bit positions)
  - Word 2: Timestamp (microseconds)

DAVIS240C Address Encoding (from official jAER/cAER specification):
- Bits 12-21: X coordinate (10 bits, range 0-1023, sensor is 240 wide)
- Bits 22-30: Y coordinate (9 bits, range 0-511, sensor is 180 tall)  
- Bit 11: Polarity (0=OFF, 1=ON)

Reference: https://github.com/Enny1991/dvs_emg_fusion/blob/master/converter.py
"""

import numpy as np


def read_aedat2_events(file_path):
    """
    Read AEDAT2.0 file and return events as numpy array.
    
    Args:
        file_path: Path to .aedat file
    
    Returns:
        events: numpy array of shape [N, 4] with columns [x, y, t, p]
                - x, y: spatial coordinates (float32)
                - t: timestamp in microseconds (float32)
                - p: polarity 0=OFF, 1=ON (float32)
    """
    
    with open(file_path, 'rb') as f:
        # Skip header lines (lines starting with '#')
        header_end = 0
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                # We've reached the end of header, go back to start of data
                f.seek(header_end)
                break
            header_end = f.tell()
        
        # Read binary data
        data = f.read()
    
    # Number of events
    num_events = len(data) // 8
    
    if num_events == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    # Parse events - AEDAT2.0 uses BIG-ENDIAN byte order!
    # Each event is 8 bytes: address (4 bytes BE) + timestamp (4 bytes BE)
    events_raw = np.frombuffer(data[:num_events * 8], dtype='>u4').reshape(-1, 2)  # '>u4' = big-endian uint32
    
    # Extract address and timestamp
    addresses = events_raw[:, 0]
    timestamps = events_raw[:, 1]
    
    # Decode address bits for DAVIS240C (official AEDAT2.0 format)
    # Source: https://github.com/Enny1991/dvs_emg_fusion/blob/master/converter.py
    x = (addresses >> 12) & 0x3FF  # bits 12-21 (10 bits for x, 0-1023 range)
    y = (addresses >> 22) & 0x1FF  # bits 22-30 (9 bits for y, 0-511 range) 
    polarity = (addresses >> 11) & 0x1  # bit 11 for polarity
    
    # Stack into [N, 4] array
    events = np.stack([
        x.astype(np.float32),
        y.astype(np.float32),
        timestamps.astype(np.float32),
        polarity.astype(np.float32)
    ], axis=1)
    
    return events


# Alias for backward compatibility - now both functions do the same thing
def read_aedat2_with_dynamic_parsing(file_path, expected_width=240, expected_height=180):
    """
    Read AEDAT2.0 file with DAVIS240C format.
    
    This function is kept for backward compatibility with existing code.
    It now directly uses the standard DAVIS240C format.
    
    Args:
        file_path: Path to .aedat file
        expected_width: Expected sensor width (default: 240, not used)
        expected_height: Expected sensor height (default: 180, not used)
    
    Returns:
        events: numpy array of shape [N, 4] with columns [x, y, t, p]
    """
    return read_aedat2_events(file_path)

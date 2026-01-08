"""
Quick example script showing different filtering configurations.
Copy and paste these into configs/custom_gesture_config.yaml under PRECOMPUTING.filter
"""

# Example 1: Test with STATIC backgrounds only (RECOMMENDED FIRST TEST)
# This removes dynamic backgrounds which may be confusing the model
# Dataset size: 512 sequences (50% of full dataset)
EXAMPLE_1_STATIC_ONLY = """
  filter:
    view: both
    lighting: both
    background: STATIC  # Remove dynamic backgrounds
"""

# Example 2: Test with TOP view only
# Dataset size: 512 sequences (50% of full dataset)
EXAMPLE_2_TOP_VIEW = """
  filter:
    view: TOP
    lighting: both
    background: both
"""

# Example 3: Easiest subset - TOP + STATIC + LIGHT
# Dataset size: 128 sequences (12.5% of full dataset)
EXAMPLE_3_EASIEST = """
  filter:
    view: TOP
    lighting: LIGHT
    background: STATIC
"""

# Example 4: Hardest subset - SIDE + DYNAMIC + DARK
# Dataset size: 128 sequences (12.5% of full dataset)
EXAMPLE_4_HARDEST = """
  filter:
    view: SIDE
    lighting: DARK
    background: DYNAMIC
"""

# Example 5: Use ALL data (default)
# Dataset size: 1024 sequences (100% of full dataset)
EXAMPLE_5_ALL_DATA = """
  filter:
    view: both
    lighting: both
    background: both
"""

# Example 6: Multiple specific values using lists
# Dataset size: varies based on selection
EXAMPLE_6_CUSTOM_LIST = """
  filter:
    view: ["TOP", "SIDE"]     # List form (same as "both")
    lighting: LIGHT           # Single value
    background: ["STATIC"]    # List with one item (same as "STATIC")
"""

print("=" * 80)
print("SEQUENCE FILTERING EXAMPLES")
print("=" * 80)
print("\nTo use any example, copy the filter config into:")
print("  configs/custom_gesture_config.yaml")
print("\nunder the PRECOMPUTING section.")
print("\nAlways test first with: python test_sequence_filters.py")
print("=" * 80)

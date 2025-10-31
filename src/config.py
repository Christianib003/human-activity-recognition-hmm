# src/config.py
STATES = ("standing", "walking", "jumping", "still")
TARGET_HZ = 100        # harmonized rate
EDGE_TRIM_SEC = 1.0             # trim at start & end
MERGE_TOL_SEC = 0.01            # accel↔gyro merge tolerance (~10 ms)
RANDOM_SEED = 42

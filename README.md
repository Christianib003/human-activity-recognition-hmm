# Human Activity Recognition with Hidden Markov Models

> **Modeling Human Activity States Using Hidden Markov Models**  
> _Inferring real-world motion states from smartphone inertial sensor data_

---

## Project Overview

This project implements a complete **Hidden Markov Model (HMM)** pipeline to recognize four human activities — **standing, walking, jumping, and still** — from smartphone accelerometer and gyroscope data. The system demonstrates how HMMs can decode hidden activity states from noisy, continuous sensor measurements.

### Key Features

-   Real-world data collection using smartphone sensors (100 Hz sampling)
-   41 engineered features combining time-domain and frequency-domain characteristics
-   Full HMM implementation with Viterbi decoding and Baum-Welch training
-   100% test accuracy on unseen data
-   Comprehensive visualizations of transitions, emissions, and confusion matrices

---

## Use Case

**Wearable Health Monitoring**: This system can be deployed in fitness trackers, smartwatches, or health monitoring apps to automatically detect user activity patterns. Applications include:

-   Tracking daily activity levels (steps, sedentary time, exercise)
-   Fall detection for elderly care
-   Rehabilitation progress monitoring
-   Personalized fitness coaching

---

## Project Structure

```
hmm-human-activity/
├── notebooks/
│   └── HMM_Project.ipynb          # Complete analysis pipeline
├── src/
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Global configuration (sampling rate, states, etc.)
│   ├── data_loader.py             # Data ingestion, cleaning, resampling
│   ├── feature_extractor.py       # Time/frequency feature engineering
│   ├── model.py                   # HMM algorithms (Viterbi, Baum-Welch)
│   └── visualizer.py              # Plotting functions (transitions, emissions, confusion)
├── data/                          # Data directory (not in repo)
│   ├── raw/
│   │   ├── train/                 # Raw training recordings (.zip)
│   │   └── test/                  # Raw test recordings (.zip)
│   └── processed/
│       ├── train/                 # Cleaned CSVs (resampled, merged sensors)
│       └── test/                  # Cleaned test CSVs
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## Getting Started

### Prerequisites

-   **Python 3.9+** (tested on Python 3.9, 3.10, 3.11)
-   **pip** or **conda** for package management

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Christianib003/hmm-human-activity.git
    cd hmm-human-activity
    ```

2. **Create a virtual environment** (recommended)

    ```bash
    # Using venv
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Or using conda
    conda create -n hmm-activity python=3.10
    conda activate hmm-activity
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up data directory** (if you have your own data)
    ```bash
    mkdir -p data/raw/train data/raw/test
    # Place your .zip recordings in data/raw/train/ and data/raw/test/
    ```

---

## Usage

### Running the Complete Pipeline

1. **Launch Jupyter Notebook**

    ```bash
    jupyter notebook notebooks/HMM_Project.ipynb
    ```

2. **Run all cells** or step through the notebook:
    - **Section 1**: Data cleaning and harmonization (100 Hz resampling)
    - **Section 2**: Device metadata and sampling rate verification
    - **Section 3**: Exploratory data analysis with visualizations
    - **Section 4**: Feature extraction (41 features) and Z-score normalization
    - **Section 5**: HMM training (supervised init + Baum-Welch EM)
    - **Section 6**: Evaluation metrics and reflection

### Expected Outputs

-   Accelerometer/Gyroscope plots for each activity
-   Transition probability heatmaps
-   Emission probability distributions (means + top features)
-   Confusion matrices (val/test splits)
-   Timeline plots showing predicted vs true activities
-   Metrics table: sensitivity, specificity, overall accuracy

---

## Data Collection

### Recording Protocol

We used the **Sensor Logger** app (iOS/Android) to collect motion data:

| Activity     | Duration | Phone Placement                        | Notes                                    |
| ------------ | -------- | -------------------------------------- | ---------------------------------------- |
| **Standing** | 10-17s   | Front pocket (screen out, upside-down) | Keep phone steady at waist level         |
| **Walking**  | 10-17s   | Front pocket                           | Maintain consistent pace (~2 Hz cadence) |
| **Jumping**  | 10-17s   | Front pocket                           | Continuous vertical jumps                |
| **Still**    | 10-17s   | Flat surface                           | Phone stationary on desk                 |

### Dataset Summary

-   **Total recordings**: 64 files (52 train, 12 test)
-   **Sensors**: Accelerometer (ax, ay, az) + Gyroscope (gx, gy, gz)
-   **Sampling rate**: Harmonized to **100 Hz**
-   **Window size**: 2.0 s with 75% overlap → ~20-26 windows per recording
-   **Total windows**: 1,144 train, 277 test

---

## Technical Details

### Feature Engineering

We extract **41 features** per 2-second window:

**Time-Domain (35 features)**:

-   Per-axis stats: mean, std, variance, peak-to-peak, RMS (6 axes × 5 = 30)
-   Accelerometer magnitude stats: mean, std, var, ptp, RMS (5)
-   Axis correlations: xy, xz, yz (3)
-   Signal Magnitude Area (SMA): 1

**Frequency-Domain (2 features)**:

-   Dominant frequency (from accel magnitude FFT)
-   Spectral energy (from accel magnitude FFT)

**Normalization**: Z-score using train-only statistics (mean=0, std=1)

### HMM Implementation

**Model Structure**:

-   **States (Z)**: 4 hidden states (standing, walking, jumping, still)
-   **Observations (X)**: 41-dimensional feature vectors
-   **Emissions (B)**: Diagonal Gaussian per state (mean vector + variance vector)
-   **Transitions (A)**: 4×4 state transition matrix
-   **Initial (π)**: Starting state probabilities

**Training**:

1. **Supervised initialization**: Estimate parameters from labeled windows
2. **Baum-Welch EM**: Refine parameters (converges in 4 iterations with Δlog-likelihood < 1e-3)

**Decoding**:

-   **Viterbi algorithm** (log-space) to find most likely state sequence per recording

---

## Results

### Performance Metrics (Test Set)

| Activity | Samples | Sensitivity | Specificity | Overall Accuracy |
| -------- | ------- | ----------- | ----------- | ---------------- |
| Jumping  | 63      | 1.000       | 1.000       | 1.000            |
| Standing | 78      | 1.000       | 1.000       | 1.000            |
| Still    | 66      | 1.000       | 1.000       | 1.000            |
| Walking  | 70      | 1.000       | 1.000       | 1.000            |

**Overall Test Accuracy**: 100.0%

### Key Insights

-   Walking and jumping are easiest to distinguish due to strong periodicity and high-energy impulses
-   Standing vs still are most similar (both low-variance) but still perfectly separated
-   Transition probabilities reflect single-activity clips (high diagonal values)
-   Feature quality drives performance; emissions dominate over transitions for this dataset

---

## Contributors

**Team Members**:

-   **Christian B.** - Model implementation (Viterbi, Baum-Welch), evaluation metrics, results analysis
-   **Reponse I.** - Data collection coordination, feature engineering, visualizations, preprocessing pipeline

**Devices Used**:

-   Christian B.: Android Pixel (100 Hz, front pocket - screen out, upside-down)
-   Reponse I.: iPhone 15 (100 Hz, back pocket - screen in, upright)

---

## Future Improvements

1. Robustness testing: Test with varied phone placements (hand-held, backpack), walking speeds, and environments
2. Additional activities: Include running, cycling, sitting, stairs
3. Real-time deployment: Implement sliding window inference for live activity tracking
4. Model compression: Reduce feature set for embedded/mobile deployment
5. Transfer learning: Test model generalization to new users without retraining

---

## References

-   Rabiner, L. R. (1989). A tutorial on hidden Markov models. _Proceedings of the IEEE_, 77(2), 257-286.
-   Baum, L. E., et al. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. _The Annals of Mathematical Statistics_.
-   Forney, G. D. (1973). The Viterbi algorithm. _Proceedings of the IEEE_, 61(3), 268-278.

---

## License

This project is for **educational purposes** as part of coursework at African Leadership University (ALU). All data was collected with informed consent from participants.

---

## Acknowledgments

Special thanks to the ALU Machine Learning course instructors for guidance on HMM theory and implementation best practices.

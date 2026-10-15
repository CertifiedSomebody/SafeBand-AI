# SafeBand AI - Autoencoder Practical

## Objective

Apply an undercomplete Autoencoder to SafeBand multisensor data and
demonstrate reconstruction-error-based anomaly detection.

## Files

- `train_autoencoder.py`
  - Loads the SafeBand BNO055 synthetic dataset.
  - Selects 23 numerical sensor features.
  - Standardizes the features.
  - Trains an undercomplete Autoencoder.
  - Saves the trained model and plots.
  - Calculates a 95th-percentile reconstruction-error threshold.

- `anomaly_demo.py`
  - Loads the trained Autoencoder.
  - Uses real SafeBand sensor samples.
  - Creates controlled abnormal sensor readings.
  - Compares reconstruction error between normal and abnormal data.
  - Generates the final anomaly-detection plot.

## Expected SafeBand application

The Autoencoder learns normal multisensor relationships. A sensor pattern
that differs substantially from learned patterns should have a larger
reconstruction error. This can be used as an additional anomaly signal in
SafeBand's safety pipeline.

This experiment is a practical demonstration and does not constitute
medical diagnosis.

## Running

From the SafeBand project root:

    python experiments/autoencoder/train_autoencoder.py

Then:

    python experiments/autoencoder/anomaly_demo.py

Required packages:

    pip install tensorflow pandas numpy scikit-learn matplotlib

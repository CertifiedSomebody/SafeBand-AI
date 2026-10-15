"""
SafeBand AI - Autoencoder Practical
Experiment 2: Consistent anomaly-detection demonstration.

IMPORTANT:
The anomaly threshold is NOT recalculated from the 100 demonstration
samples. It is loaded from the validation-set threshold produced during
training. This makes the experiment methodologically consistent.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "synthetic" / "bno055" / "bno055_synthetic_demo.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_PATH = RESULTS_DIR / "safeband_autoencoder.keras"
SCALER_PATH = RESULTS_DIR / "scaler.npz"
METRICS_PATH = RESULTS_DIR / "metrics.json"

FEATURES = [
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
    "euler_heading_deg", "euler_roll_deg", "euler_pitch_deg",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "linear_accel_x_mps2", "linear_accel_y_mps2", "linear_accel_z_mps2",
    "gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2",
    "temperature_c",
]


def main():
    print("=" * 70)
    print("SafeBand AI - Consistent Autoencoder Anomaly Demonstration")
    print("=" * 70)

    for required in [MODEL_PATH, SCALER_PATH, METRICS_PATH, DATA_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    model = keras.models.load_model(MODEL_PATH)

    scaler_data = np.load(SCALER_PATH, allow_pickle=True)
    mean = scaler_data["mean"]
    scale = scaler_data["scale"]

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Use the SAME threshold determined during training.
    threshold = float(metrics["anomaly_threshold_95_percentile"])

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].astype(np.float32).dropna().copy()

    # Reproducible set of real SafeBand sensor samples.
    normal = X.sample(n=min(100, len(X)), random_state=42).to_numpy()

    # Create a controlled abnormal pattern in standardized space.
    # This avoids unrealistic raw-unit multiplication and makes the
    # demonstration easier to explain academically.
    normal_scaled = (normal - mean) / scale
    anomalous_scaled = normal_scaled.copy()

    # Inject a strong but controlled multisensor deviation:
    # acceleration, gyroscope and temperature become unusual.
    anomalous_scaled[:, 0:3] += 3.0
    anomalous_scaled[:, 3:6] += 3.0
    anomalous_scaled[:, -1] += 3.0

    # Reconstruction.
    normal_recon = model.predict(normal_scaled, verbose=0)
    anomaly_recon = model.predict(anomalous_scaled, verbose=0)

    normal_error = np.mean(
        np.square(normal_scaled - normal_recon), axis=1
    )
    anomaly_error = np.mean(
        np.square(anomalous_scaled - anomaly_recon), axis=1
    )

    normal_detected = normal_error > threshold
    anomaly_detected = anomaly_error > threshold

    print(f"\nTraining-derived 95% threshold : {threshold:.6f}")
    print(f"Normal mean reconstruction error: {normal_error.mean():.6f}")
    print(f"Anomaly mean reconstruction error: {anomaly_error.mean():.6f}")

    print("\nDetection summary")
    print("-" * 40)
    print(
        f"Normal samples flagged: "
        f"{normal_detected.sum()}/{len(normal_detected)}"
    )
    print(
        f"Injected anomalous samples detected: "
        f"{anomaly_detected.sum()}/{len(anomaly_detected)}"
    )

    results = pd.DataFrame({
        "normal_reconstruction_error": normal_error,
        "anomalous_reconstruction_error": anomaly_error,
        "normal_flag": normal_detected,
        "anomaly_flag": anomaly_detected,
    })
    results.to_csv(
        RESULTS_DIR / "anomaly_detection_results_fixed.csv",
        index=False,
    )

    # Use a logarithmic y-axis so both normal and abnormal errors remain
    # visually readable if their magnitudes differ substantially.
    plt.figure(figsize=(10, 5))
    plt.plot(normal_error, label="Normal Sensor Pattern")
    plt.plot(anomaly_error, label="Injected Anomaly")
    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Training-derived 95% Threshold = {threshold:.4f}",
    )
    plt.xlabel("Sample")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("SafeBand Autoencoder: Normal vs Anomalous Sensor Patterns")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "normal_vs_anomaly_fixed.png",
        dpi=200,
    )
    plt.close()

    print("\nSaved:")
    print(RESULTS_DIR / "anomaly_detection_results_fixed.csv")
    print(RESULTS_DIR / "normal_vs_anomaly_fixed.png")


if __name__ == "__main__":
    main()

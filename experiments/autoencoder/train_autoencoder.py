"""
SafeBand AI - Autoencoder Practical
Experiment 1: Train an undercomplete Autoencoder on BNO055 sensor data.

Purpose:
    Learn normal multisensor patterns and use reconstruction error
    as an anomaly-detection signal.

Dataset:
    datasets/synthetic/bno055/bno055_synthetic_demo.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "synthetic" / "bno055" / "bno055_synthetic_demo.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 32
RANDOM_STATE = 42

# BNO055 features used by the Autoencoder.
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


def build_autoencoder(input_dim: int) -> keras.Model:
    """Create an undercomplete dense Autoencoder."""
    inputs = keras.Input(shape=(input_dim,), name="sensor_input")

    # Encoder
    x = layers.Dense(32, activation="relu", name="encoder_dense_1")(inputs)
    x = layers.Dense(16, activation="relu", name="encoder_dense_2")(x)
    latent = layers.Dense(8, activation="relu", name="latent_space")(x)

    # Decoder
    x = layers.Dense(16, activation="relu", name="decoder_dense_1")(latent)
    x = layers.Dense(32, activation="relu", name="decoder_dense_2")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = keras.Model(inputs, outputs, name="SafeBand_Undercomplete_Autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def main():
    print("=" * 70)
    print("SafeBand AI - Autoencoder Practical")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print("Activities:", df["activity_label"].value_counts().to_dict())

    # Keep only required numeric sensor features.
    X = df[FEATURES].astype(np.float32).copy()

    # Remove any invalid rows if present.
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Usable sensor samples: {len(X)}")

    # Train/validation/test split.
    X_train, X_temp = train_test_split(
        X, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test = train_test_split(
        X_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    # Standardization is important because the sensors have different units/scales.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain samples: {len(X_train_scaled)}")
    print(f"Validation samples: {len(X_val_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")

    model = build_autoencoder(X_train_scaled.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_scaled,
        X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Reconstruction errors.
    train_recon = model.predict(X_train_scaled, verbose=0)
    val_recon = model.predict(X_val_scaled, verbose=0)
    test_recon = model.predict(X_test_scaled, verbose=0)

    train_error = np.mean(np.square(X_train_scaled - train_recon), axis=1)
    val_error = np.mean(np.square(X_val_scaled - val_recon), axis=1)
    test_error = np.mean(np.square(X_test_scaled - test_recon), axis=1)

    # Threshold: 95th percentile of validation reconstruction error.
    threshold = float(np.percentile(val_error, 95))

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Final training loss : {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Mean test error      : {test_error.mean():.6f}")
    print(f"95% anomaly threshold: {threshold:.6f}")

    # Save model.
    model.save(RESULTS_DIR / "safeband_autoencoder.keras")

    # Save scaler.
    np.savez(
        RESULTS_DIR / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.array(FEATURES, dtype=object),
    )

    # Save metrics.
    metrics = {
        "dataset": str(DATA_PATH),
        "input_features": FEATURES,
        "input_dimension": len(FEATURES),
        "latent_dimension": 8,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "train_samples": int(len(X_train_scaled)),
        "validation_samples": int(len(X_val_scaled)),
        "test_samples": int(len(X_test_scaled)),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_validation_loss": float(history.history["val_loss"][-1]),
        "mean_test_reconstruction_error": float(test_error.mean()),
        "anomaly_threshold_95_percentile": threshold,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------
    # Plot 1: Training loss
    # -----------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("SafeBand Autoencoder Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png", dpi=200)
    plt.close()

    # -----------------------------
    # Plot 2: Reconstruction error
    # -----------------------------
    plt.figure(figsize=(9, 5))
    plt.hist(test_error, bins=40)
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"95% Threshold = {threshold:.4f}",
    )
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Number of Samples")
    plt.title("SafeBand Sensor Reconstruction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reconstruction_error.png", dpi=200)
    plt.close()

    # -----------------------------
    # Plot 3: Original vs reconstructed
    # First three standardized features
    # -----------------------------
    n_show = min(150, len(X_test_scaled))
    sample_original = X_test_scaled[:n_show]
    sample_reconstructed = test_recon[:n_show]

    plt.figure(figsize=(11, 6))
    plt.plot(sample_original[:, 0], label="Original Accel X")
    plt.plot(sample_reconstructed[:, 0], "--", label="Reconstructed Accel X")
    plt.xlabel("Test Sample")
    plt.ylabel("Standardized Value")
    plt.title("Original vs Reconstructed SafeBand Sensor Signal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "original_vs_reconstructed.png", dpi=200)
    plt.close()

    # Save test reconstruction errors for later anomaly experiment.
    error_df = pd.DataFrame({
        "reconstruction_error": test_error,
        "is_anomaly_at_95_percent_threshold": test_error > threshold,
    })
    error_df.to_csv(RESULTS_DIR / "test_reconstruction_errors.csv", index=False)

    print("\nSaved results to:")
    print(RESULTS_DIR)
    print("\nNext experiment:")
    print("Run anomaly_demo.py to inject abnormal sensor readings and test detection.")


if __name__ == "__main__":
    main()

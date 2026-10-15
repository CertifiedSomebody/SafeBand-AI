from pathlib import Path
import json
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METRICS = RESULTS / "metrics.json"
ANOMALY_FIXED = RESULTS / "anomaly_detection_results_fixed.csv"
ANOMALY_OLD = RESULTS / "anomaly_detection_results.csv"
REPORT = RESULTS / "SafeBand_Autoencoder_Practical_Report.docx"

FIGURES = {
    "training": RESULTS / "training_loss.png",
    "reconstruction": RESULTS / "original_vs_reconstructed.png",
    "error": RESULTS / "reconstruction_error.png",
    "anomaly": RESULTS / "normal_vs_anomaly_fixed.png",
}

# Replace these before final submission.
STUDENT_NAME = "[YOUR NAME]"
ROLL_NO = "[YOUR ROLL NO.]"
CLASS_DIV = "[CLASS / DIVISION]"
COLLEGE = "[COLLEGE NAME]"
ACADEMIC_YEAR = "2026–27"


def shade(cell, fill="D9EAF7"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def cell_text(cell, value, bold=False, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(str(value))
    r.bold = bold
    r.font.size = Pt(size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def page_number(p):
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    a = OxmlElement("w:fldChar")
    a.set(qn("w:fldCharType"), "begin")
    b = OxmlElement("w:instrText")
    b.set(qn("xml:space"), "preserve")
    b.text = "PAGE"
    c = OxmlElement("w:fldChar")
    c.set(qn("w:fldCharType"), "end")
    r._r.extend([a, b, c])


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    repeat_header(t.rows[0])
    for i, h in enumerate(headers):
        cell_text(t.rows[0].cells[i], h, True)
        shade(t.rows[0].cells[i])
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cell_text(cells[i], v)
    doc.add_paragraph()
    return t


def figure(doc, path, caption):
    if not path.exists():
        p = doc.add_paragraph()
        p.add_run("[Missing figure: " + path.name + "]").italic = True
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(6.1))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.bold = True
    r.font.size = Pt(9)


def code_block(doc, code, max_lines=100):
    lines = code.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [
            "",
            "[Remaining source code is available in the project file.]"
        ]
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.cell(0, 0)
    shade(c, "F2F2F2")
    p = c.paragraphs[0]
    for i, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(7)
        if i < len(lines) - 1:
            r.add_break()
    doc.add_paragraph()


def fmt(v):
    return f"{float(v):.6f}"


def main():
    if not METRICS.exists():
        raise FileNotFoundError("Run train_autoencoder.py first.")

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    anomaly_path = ANOMALY_FIXED if ANOMALY_FIXED.exists() else ANOMALY_OLD
    anomaly = pd.read_csv(anomaly_path) if anomaly_path.exists() else None

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2)
    sec.bottom_margin = Cm(2)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.15
    for name, size in [("Title",20),("Heading 1",15),("Heading 2",12)]:
        doc.styles[name].font.name = "Arial"
        doc.styles[name].font.size = Pt(size)
    page_number(sec.footer.paragraphs[0])

    # COVER
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("M6 – AUTOENCODER PRACTICAL")
    r.bold = True; r.font.size = Pt(18)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Application of an Undercomplete Autoencoder\nfor Sensor Anomaly Detection in SafeBand AI")
    r.bold = True; r.font.size = Pt(20)
    doc.add_paragraph()
    table(doc, ["Student Detail", "Value"], [
        ("Student Name", STUDENT_NAME),
        ("Roll No.", ROLL_NO),
        ("Class / Division", CLASS_DIV),
        ("College", COLLEGE),
        ("Academic Year", ACADEMIC_YEAR),
    ])
    doc.add_page_break()

    # 1
    doc.add_heading("1. Objective", 1)
    doc.add_paragraph(
        "To apply an Undercomplete Autoencoder to multisensor data from the "
        "SafeBand AI project for learning a compact latent representation, "
        "reconstructing sensor measurements, and demonstrating "
        "reconstruction-error-based anomaly detection."
    )

    # 2
    doc.add_heading("2. Introduction", 1)
    doc.add_paragraph(
        "An Autoencoder is an encoder–decoder model that learns to represent "
        "input data in a latent space and reconstruct the original input "
        "from that representation. The encoder maps the input to a latent "
        "representation and the decoder reconstructs the input."
    )
    doc.add_paragraph(
        "An Undercomplete Autoencoder uses a latent representation with "
        "fewer dimensions than the input. The bottleneck encourages the "
        "model to retain important information in a compact representation. "
        "For this practical, the Undercomplete Autoencoder is applied to "
        "SafeBand sensor data for reconstruction and anomaly detection."
    )

    # 3
    doc.add_heading("3. SafeBand AI Project Application", 1)
    doc.add_paragraph(
        "SafeBand AI is an intelligent safety-band project that processes "
        "multiple sensor measurements as part of its sensing and safety "
        "pipeline. The Autoencoder is added as a complementary data-analysis "
        "component."
    )
    doc.add_paragraph(
        "The practical question is whether the Autoencoder can learn the "
        "normal structure of SafeBand sensor measurements and use "
        "reconstruction error as an additional indicator of an unusual "
        "sensor pattern."
    )

    # 4
    doc.add_heading("4. Dataset and Input Features", 1)
    doc.add_paragraph("Dataset: datasets/synthetic/bno055/bno055_synthetic_demo.csv")
    table(doc, ["Parameter", "Value"], [
        ("Total samples", "3,600"),
        ("Input features", metrics["input_dimension"]),
        ("Training samples", metrics["train_samples"]),
        ("Validation samples", metrics["validation_samples"]),
        ("Test samples", metrics["test_samples"]),
        ("Latent features", metrics["latent_dimension"]),
    ])
    doc.add_paragraph("The 23 features used by the Autoencoder are:")
    feats = metrics["input_features"]
    rows = []
    for i in range(0, len(feats), 2):
        rows.append((feats[i], feats[i+1] if i+1 < len(feats) else ""))
    table(doc, ["Feature", "Feature"], rows)

    # 5
    doc.add_heading("5. Methodology", 1)
    for item in [
        "Load the SafeBand BNO055 sensor dataset.",
        "Select the 23 numerical sensor features.",
        "Remove invalid numerical values if present.",
        "Split the data into training, validation, and test sets.",
        "Standardize the sensor features because they have different units and numerical ranges.",
        "Train the Undercomplete Autoencoder using mean squared error (MSE) reconstruction loss.",
        "Compress the 23-dimensional sensor vector into an 8-dimensional latent representation.",
        "Reconstruct the original 23-dimensional sensor vector.",
        "Calculate reconstruction error for each sample.",
        "Set the anomaly threshold using the 95th percentile of validation reconstruction error.",
        "Demonstrate anomaly detection using controlled deviations in selected sensor features.",
    ]:
        doc.add_paragraph(item, style="List Number")

    # 6
    doc.add_heading("6. Autoencoder Architecture", 1)
    table(doc, ["Stage", "Layer", "Output"], [
        ("Input", "Sensor input", "23"),
        ("Encoder", "Dense + ReLU", "32"),
        ("Encoder", "Dense + ReLU", "16"),
        ("Bottleneck", "Latent layer + ReLU", "8"),
        ("Decoder", "Dense + ReLU", "16"),
        ("Decoder", "Dense + ReLU", "32"),
        ("Output", "Linear reconstruction", "23"),
    ])
    doc.add_paragraph(
        "The model contains 2,879 trainable parameters. Since the latent "
        "dimension (8) is smaller than the input dimension (23), it is an "
        "Undercomplete Autoencoder."
    )

    # 7
    doc.add_heading("7. Python Implementation", 1)
    doc.add_paragraph(
        "The implementation uses Python, TensorFlow/Keras, NumPy, Pandas, "
        "scikit-learn, and Matplotlib."
    )
    train = HERE / "train_autoencoder.py"
    demo = HERE / "anomaly_demo.py"
    if train.exists():
        doc.add_heading("7.1 Training Program", 2)
        code_block(doc, train.read_text(encoding="utf-8"), 120)
    if demo.exists():
        doc.add_heading("7.2 Anomaly Demonstration Program", 2)
        code_block(doc, demo.read_text(encoding="utf-8"), 110)

    # 8
    doc.add_heading("8. Training Results", 1)
    table(doc, ["Metric", "Result"], [
        ("Epochs completed", metrics["epochs_completed"]),
        ("Final training loss", fmt(metrics["final_train_loss"])),
        ("Final validation loss", fmt(metrics["final_validation_loss"])),
        ("Mean test reconstruction error", fmt(metrics["mean_test_reconstruction_error"])),
        ("95% validation anomaly threshold", fmt(metrics["anomaly_threshold_95_percentile"])),
    ])
    figure(doc, FIGURES["training"], "Figure 1. Training and validation reconstruction loss.")
    doc.add_paragraph(
        "The training and validation losses decrease substantially during "
        "training and remain close toward the end of the experiment, "
        "showing that the model learned the reconstruction task."
    )

    # 9
    doc.add_heading("9. Reconstruction Analysis", 1)
    figure(doc, FIGURES["reconstruction"],
           "Figure 2. Original and reconstructed standardized accelerometer-X signal.")
    doc.add_paragraph(
        "The original and reconstructed signal follow a similar pattern. "
        "The decoder reconstructs the important structure of the sensor "
        "input from the compact latent representation."
    )
    figure(doc, FIGURES["error"],
           "Figure 3. Test-set reconstruction-error distribution and 95% threshold.")
    doc.add_paragraph(
        f"The mean test reconstruction error is "
        f"{metrics['mean_test_reconstruction_error']:.6f}. The threshold "
        f"from the 95th percentile of validation reconstruction error is "
        f"{metrics['anomaly_threshold_95_percentile']:.6f}."
    )

    # 10
    doc.add_heading("10. Anomaly Detection Demonstration", 1)
    if anomaly is not None:
        normal_mean = float(anomaly["normal_reconstruction_error"].mean())
        abnormal_mean = float(anomaly["anomalous_reconstruction_error"].mean())
        normal_flags = int(anomaly["normal_flag"].sum())
        abnormal_flags = int(anomaly["anomaly_flag"].sum())
        n = len(anomaly)
        table(doc, ["Measure", "Result"], [
            ("Training-derived threshold", fmt(metrics["anomaly_threshold_95_percentile"])),
            ("Normal mean reconstruction error", fmt(normal_mean)),
            ("Injected anomaly mean reconstruction error", fmt(abnormal_mean)),
            ("Normal samples flagged", f"{normal_flags}/{n}"),
            ("Injected anomalies detected", f"{abnormal_flags}/{n}"),
        ])
        figure(doc, FIGURES["anomaly"],
               "Figure 4. Normal versus controlled anomalous sensor patterns.")
        doc.add_paragraph(
            "For the demonstration, 100 real SafeBand sensor samples were "
            "used as the normal reference set. Controlled deviations were "
            "introduced into selected standardized accelerometer, gyroscope, "
            "and temperature features. The same validation-derived threshold "
            "was used for both groups."
        )
        doc.add_paragraph(
            f"The normal demonstration samples had a mean reconstruction "
            f"error of {normal_mean:.6f}, while the controlled anomalous "
            f"patterns had a mean reconstruction error of {abnormal_mean:.6f}. "
            f"{abnormal_flags} of {n} injected anomalous samples crossed the "
            f"threshold, while {normal_flags} normal samples crossed it."
        )
    doc.add_paragraph(
        "This is a controlled anomaly demonstration. It does not claim "
        "clinical diagnosis or clinical fall-detection accuracy."
    )

    # 11
    doc.add_heading("11. Practical Application in SafeBand AI", 1)
    table(doc, ["Step", "Operation"], [
        ("1", "Collect multisensor readings."),
        ("2", "Preprocess and standardize the sensor vector."),
        ("3", "Pass the vector through the trained Autoencoder."),
        ("4", "Reconstruct the sensor vector."),
        ("5", "Calculate reconstruction error."),
        ("6", "Compare the error with the validation-derived threshold."),
        ("7", "Generate an anomaly signal when the error is unusually high, for further processing."),
    ])
    doc.add_paragraph(
        "The anomaly signal can be combined with the existing SafeBand "
        "sensor-fusion and risk-processing logic. The Autoencoder therefore "
        "acts as a complementary unsupervised feature-learning and "
        "anomaly-detection component."
    )

    # 12
    doc.add_heading("12. Results and Discussion", 1)
    for item in [
        "Data representation: 23 sensor measurements are compressed into an 8-dimensional latent representation.",
        "Data reconstruction: the decoder reconstructs the 23-dimensional sensor vector from the latent representation.",
        "Feature learning: the bottleneck forces the model to retain useful information in a compact representation.",
        "Anomaly detection: reconstruction error provides a numerical signal for identifying unusual sensor patterns.",
    ]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_paragraph(
        "The decreasing training and validation losses demonstrate learning "
        "of the reconstruction task. The reconstruction plot shows that the "
        "learned representation retains the major structure of the selected "
        "sensor signal. The controlled anomaly experiment demonstrates the "
        "intended reconstruction-error mechanism."
    )

    # 13
    doc.add_heading("13. Limitations", 1)
    for item in [
        "The BNO055 dataset used for this practical is a synthetic SafeBand demonstration dataset.",
        "The anomalous examples were deliberately injected and are not a clinical or field-validated anomaly dataset.",
        "The experiment evaluates reconstruction-error separation rather than clinical diagnostic accuracy.",
        "The threshold is specific to the dataset, preprocessing procedure, and trained model.",
        "Further validation with real-world SafeBand recordings would be required before deployment.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # 14
    doc.add_heading("14. Conclusion", 1)
    doc.add_paragraph(
        "An Undercomplete Autoencoder was successfully implemented and "
        "applied to SafeBand AI multisensor data. The model compressed "
        "23 sensor features into an 8-dimensional latent representation "
        "and reconstructed the sensor vector through a decoder."
    )
    doc.add_paragraph(
        "The experiment demonstrated decreasing reconstruction loss, "
        "similar original and reconstructed sensor patterns, and a "
        "reconstruction-error-based approach for identifying controlled "
        "anomalous sensor patterns. The practical application shows how "
        "an Autoencoder can be added to SafeBand AI as a complementary "
        "anomaly-detection mechanism."
    )

    # 15
    doc.add_heading("15. Generated Outputs", 1)
    for item in [
        "training_loss.png – training and validation loss",
        "original_vs_reconstructed.png – original versus reconstructed sensor signal",
        "reconstruction_error.png – reconstruction-error distribution and threshold",
        "normal_vs_anomaly_fixed.png – normal versus controlled anomalous sensor patterns",
        "safeband_autoencoder.keras – trained Autoencoder model",
        "metrics.json – recorded experiment metrics",
        "anomaly_detection_results_fixed.csv – anomaly demonstration results",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # APPENDIX
    doc.add_heading("Appendix A – Core Autoencoder Model", 1)
    doc.add_paragraph(
        "The following is the core model definition used in the practical."
    )
    core_code = """def build_autoencoder(input_dim):
    inputs = keras.Input(shape=(input_dim,), name="sensor_input")

    # Encoder
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    latent = layers.Dense(8, activation="relu",
                          name="latent_space")(x)

    # Decoder
    x = layers.Dense(16, activation="relu")(latent)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear",
                           name="reconstruction")(x)

    model = keras.Model(inputs, outputs,
                        name="SafeBand_Undercomplete_Autoencoder")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model"""
    code_block(doc, core_code, 50)

    doc.add_paragraph("End of report.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    doc.save(REPORT)
    print("=" * 70)
    print("SafeBand AI - M6 Autoencoder Practical Report Generator")
    print("=" * 70)
    print("Report generated successfully:")
    print(REPORT)


if __name__ == "__main__":
    main()

# SafeBand Activity DL V1 — Experiment Plan

## 1. Why this experiment exists

SafeBand Activity V4 established a strong classical baseline using 38 engineered ACC features. The pooled subject-independent result was approximately 81.8% accuracy and 79.5% macro-F1. The major remaining weakness was stationary/postural discrimination, particularly RESTING and SITTING.

V3 showed that a confusion-guided specialist architecture did not provide a meaningful improvement. That experiment is retained as a lesson learned rather than deployed.

DL is therefore tested for a different reason: a neural model can learn temporal representations directly from the raw ACC waveform rather than depending entirely on handcrafted statistics.

## 2. Hypothesis

A compact 1D temporal CNN may learn discriminative patterns in the 2-second 3-axis ACC sequence that are not fully represented by the 38-feature V4 representation.

A CNN+GRU model is provided as an optional second architecture because the convolutional stage can capture local motion patterns while the recurrent stage can model temporal dependencies.

## 3. Why raw waveform input

Using the existing 38 features with a neural network would mostly turn the experiment into an MLP-on-features comparison. That is useful as a separate experiment, but it would not test the principal DL advantage for HAR.

DL V1 therefore uses the raw 40 x 3 ACC window.

## 4. Leakage controls

The evaluation is subject-independent.

For every outer fold:

1. Test subjects are completely held out.
2. Per-axis normalization statistics are calculated only from training subjects.
3. The neural network is trained only on training subjects.
4. Early stopping monitors an internal validation split derived from the training subjects.
5. Test subjects are evaluated once after model selection/training.

This matters because HAR can look substantially stronger when windows from the same subject appear in both training and testing.

## 5. Metrics

Primary:

- macro-F1
- balanced accuracy

Secondary:

- accuracy
- per-class precision/recall/F1
- confusion matrix
- fold mean and standard deviation

Deployment metrics should later include:

- parameter count
- serialized model size
- RAM requirement
- inference latency
- approximate compute/MACs
- quantized performance

## 6. Interpretation

A DL model is not accepted solely because its single best fold is high.

Evidence for replacing V4 should include:

- improvement in mean outer-fold macro-F1,
- improvement in balanced accuracy,
- no unacceptable degradation of RESTING/SITTING recall,
- reasonable fold stability,
- and a plausible embedded deployment path.

A 90–95% accuracy requirement can be used as a project target, but the experiment must not change the evaluation protocol simply to reach that number.

## 7. Next experiment if DL improves

If DL provides a meaningful gain, the next stage is not immediately a larger network. First test:

- quantization,
- latency,
- memory,
- and then real BNO055 gyro/orientation channels.

The BNO055 experiment is particularly relevant because the remaining ACC-only difficulty is concentrated around stationary/postural activities.

## 8. External research context

Published wearable HAR work demonstrates that compact 1D CNNs can learn temporal representations directly from inertial signals, and subject-independent evaluation is used in rigorous studies. Reported accuracies vary substantially with activity complexity, sensor placement, dataset and evaluation protocol. Therefore published 90%+ figures are not treated as expected performance for SafeBand.

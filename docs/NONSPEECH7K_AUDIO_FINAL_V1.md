# Nonspeech7k Audio Final V1

V3.1 grouped CV is frozen. It measured mean accuracy 71.574%, mean balanced accuracy 70.467%, mean Macro-F1 64.966%; pooled accuracy 72.529%, pooled balanced accuracy 71.759%, pooled Macro-F1 65.912%.

This final-training script creates one deployable `SmallLogMelCNN` using all 6,283 approved training recordings. The epoch count is fixed to 16 from the mean V3.1 best epoch (~15.6), so final training has no test-driven early stopping.

Preprocessing: 16 kHz mono -> 64-bin mel -> n_fft 512 -> hop 160 -> 50 Hz to 7600 Hz -> dB with per-recording max reference -> [-80,0] -> 128 frames -> standardization using the complete approved training set.

The `.pt` is a deployment artifact, not a new accuracy measurement. The locked V3.1 CV metrics remain the benchmark evidence.

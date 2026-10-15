# V6 experiment plan

Compare the original 11-class ACC-only task with the reduced SafeBand taxonomy. The goal is to determine whether some fine-grained AICOS-HAR labels are unnecessarily difficult for SafeBand.

Primary metrics: Macro-F1, balanced accuracy, then accuracy.

If the reduced taxonomy materially improves performance, proceed to ACC+gyro using the same taxonomy. If not, freeze ACC-only and add gyro anyway because BNO055 provides complementary rotational information.

MAX30102 is not a gyro; it is the PPG/physiological branch. BNO055 supplies accelerometer and gyroscope data.

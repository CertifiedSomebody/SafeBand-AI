# SisFall dataset notes

The compatibility layer uses the published SisFall sensor specifications:

- ADXL345: +/-16 g, 13-bit
- MMA8451Q: +/-8 g, 14-bit
- raw acquisition: 200 Hz

The SafeBand V6 BITS-2 model is accelerometer-first and its BITS-2 ingestion
pipeline preserves the released acceleration values. Therefore this adapter
converts SisFall counts into **g**, not m/s², before reproducing the V6
84-feature contract.

The 200 Hz stream is reduced to 20 Hz with non-overlapping 10-sample block
means. This preserves the V6 feature contract's 20 Hz FFT reference and
60-sample window length.

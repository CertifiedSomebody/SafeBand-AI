# PPG-DaLiA V2 Preparation Fix

## Fix
The V1 preparation script incorrectly expected the 4 Hz activity stream to contain `n_windows * 8` samples. That corresponds only to the 2-second shift between windows, not the full 8-second activity span represented by each HR target window.

PPG-DaLiA uses:
- Activity: 4 Hz
- HR target windows: 8 s
- HR target shift: 2 s

For `N` HR windows, the required full-recording activity length is:

`(N - 1) * (2 * 4) + (8 * 4)`

For S1:
- HR labels: 4603
- Activity samples: 36848
- Expected: `4602 * 8 + 32 = 36848`

For each HR window, the activity representative is taken at the midpoint (4 seconds after the HR window start), i.e. offset 16 samples at 4 Hz.

No BVP/ACC/HR framing or sensor values are changed by this fix.

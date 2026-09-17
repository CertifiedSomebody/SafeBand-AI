# PPG V4.1 Requirements

Use the project's existing virtual environment. Required Python packages:

- numpy
- pandas
- scipy
- scikit-learn
- joblib

No deep-learning framework is required.

The implementation uses `numpy.trapezoid` for spectral integration. If an older
NumPy version in the project lacks it, replace that one call with
`numpy.trapz`; do not change the numerical pipeline otherwise.

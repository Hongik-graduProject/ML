# Test Log

## Environment
- Python 3.14 caused compatibility issues.
- Python 3.12 virtual environment was used for stable execution.

## Dependency Issues
- XGBoost initially failed due to missing OpenMP runtime on macOS.
- The issue was resolved by installing libomp with Homebrew.

## UI Issues
- Streamlit initially crashed because the model training process was too heavy during app startup.
- The app was revised so that the model trains only after the user clicks the training button.

## Visualization Issues
- Plotly vertical-line annotation caused a Timestamp error.
- The chart code was modified to handle date values safely.

## Current Status
- Streamlit dashboard runs locally.
- Company selection, stock chart, financial summary, and model prediction functions are available.

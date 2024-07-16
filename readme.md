# Stock Price Forecasting

This project uses the Prophet model to forecast stock prices. The project includes steps for data preparation, model training, cross-validation, and generating forecasts.

## File Structure

- `data/`: Contains the raw and processed data files.
- `models/`: Contains the saved Prophet model and its serialized version.
- `notebooks/`: Jupyter Notebooks for interactive data analysis and model training.
- `scripts/`: Python scripts for various tasks such as data preparation, model training, and forecasting.
- `app/`: Flask API application and Dockerfile for deployment.
- `requirements.txt`: List of required libraries.
- `README.md`: Project documentation.
- `.gitignore`: Git ignore file.

## Saved Model Files

- `models/prophet_model.joblib`: The trained Prophet model saved using joblib.
- `models/serialized_model.json`: The serialized version of the trained Prophet model.

## Steps to Run

1. **Set up the virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the model training script**:
    ```bash
    cd scripts/
    python model_training.py
    ```

4. **Generate forecasts**:
    ```bash
    python generate_forecasts.py
    ```

## Usage

The saved model files can be loaded and used to generate future forecasts without retraining the model.

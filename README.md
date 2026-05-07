# Sales Forecasting System

This project is an automated, machine learning-driven Time Series Forecasting System. It predicts future sales for different US states using historical data. The system automatically trains multiple models, selects the best-performing one based on RMSE, and serves the predictions via a production-ready REST API.

## Features
- **Multi-Model Training**: Automatically trains **SARIMA**, **Prophet**, and **XGBoost** models for every state.
- **Auto Model Selection**: Evaluates models on a validation set and selects the most accurate model for each individual state.
- **REST API**: Provides a FastAPI-based web server to fetch future sales forecasts interactively.
- **Interactive Documentation**: Comes with a built-in Swagger UI dashboard to easily test predictions.

## Tech Stack
- **Python** (Recommended 3.10 or 3.11)
- **Data Science/ML**: Pandas, Numpy, Statsmodels (SARIMA), Prophet, XGBoost, Scikit-Learn
- **API**: FastAPI, Uvicorn

## Folder Structure
```
forecasting_system_code/
├── api/
│   └── main.py              # FastAPI server and endpoints
├── data/
│   └── sales_data.xlsx      # Raw historical dataset (Add this manually)
├── models/
│   ├── prophet_model.py
│   ├── sarima_model.py
│   └── xgboost_model.py
├── outputs/
│   └── trained_models/      # Saved .pkl models after training
├── data_preprocessing.py    # Data cleaning and feature engineering
├── train_pipeline.py        # Main pipeline to train and evaluate models
└── requirements.txt         # Project dependencies
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd forecasting_system_code
```

### 2. Add your Data
Create a folder named `data` inside the root directory and place your historical dataset there. It must be named `sales_data.xlsx`.

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```
Activate it:
- On Windows: `.\venv\Scripts\activate`
- On Mac/Linux: `source venv/bin/activate`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models
Before you can get predictions, you need to train the models on your dataset. Run the pipeline:
```bash
python train_pipeline.py
```
This will process the data, train all models for each state, pick the best ones, and save them into the `outputs/trained_models` folder.

### Starting the API Server
Once the models are trained, start the forecasting web server:
```bash
uvicorn api.main:app --reload
```

### Testing the API
Open your web browser and navigate to:
**http://127.0.0.1:8000/docs#/Forecasting/forecast_state_forecast__state__get**

This will open an interactive dashboard where you can test the endpoints:
- **`GET /models`**: See which model performed best for each state.
- **`GET /forecast/{state}`**: Get weekly sales predictions for a specific state (e.g., Texas).
- **`GET /forecast/batch/all`**: Get predictions for all states at once.

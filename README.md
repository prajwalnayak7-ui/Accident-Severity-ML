Accident Severity Prediction System (AI/ML Project)
A Machine Learning system that predicts the severity level of a road accident based on environmental, geographical, and traffic-related features.
Built as part of the AI–ML Internship at Dlithe Consultancy Services Pvt. Ltd.
This project includes:

✅ Data preprocessing + feature engineering
✅ ML model training (Random Forest & XGBoost)
✅ FastAPI backend
✅ Streamlit web interface
✅ Fully reproducible ML pipeline
✅ Clean project structure

⭐ Project Highlights

Built with Python, Scikit-Learn, XGBoost, SMOTE

Implements two ML models:

Random Forest Classifier

XGBoost Classifier

API built using FastAPI

User interface built using Streamlit

Handles real accident data: timestamps, weather, environment

Automatic datetime feature extraction

Probability distribution for each severity class

Clean and production-ready pipeline architecture

🎯 Problem Statement

Road accidents are a major public safety issue worldwide, leading to large numbers of injuries, deaths, and economic losses.
The severity of an accident depends on a combination of weather conditions, visibility, lighting, road structure, location, and time-based factors.

Manually analyzing these patterns is difficult — therefore, the goal of this project is to build an intelligent ML system that predicts accident severity in real-time, helping authorities and emergency services respond faster and allocate resources effectively.

🎯 Objective

To design an end-to-end machine learning pipeline

To predict accident severity: Minor, Moderate, Serious, Severe

To create a deployable API service for predictions

To build a user-friendly web interface

To compare ML models (RandomForest vs XGBoost)

To prepare a structured and scalable project architecture

📁 Dataset

This project uses the US Accidents Dataset (March 2023) containing millions of accident records.
However:

❗ The dataset is too large (>2GB) and is not included in the repository.
❗ The dataset must be downloaded manually and placed in:

data/raw/US_Accidents_March23.csv

Dataset Contains (Examples):

Timestamp of accident (Start_Time, End_Time)

Weather condition (Rain, Clear, Fog, etc.)

Temperature, visibility, wind speed

Location (City, County, State)

GPS Coordinates (Lat, Lng)

Road features (Junction, Crossing, etc.)

Target variable: Severity (1 to 4)

You can download similar datasets from Kaggle:
🔗 https://www.kaggle.com/sobhanmoosavi/us-accidents

🧠 ML Models Used
1. Random Forest Classifier

Handles high-dimensional data

Robust to noise

Baseline classifier

2. XGBoost Classifier

Gradient boosting

Handles imbalance well

Typically higher accuracy

Evaluation Metrics:

Accuracy

F1-score (macro & weighted)

Confusion matrix

Probability distribution of predictions

🏗️ Project Structure
accident-severity-ml/
│
├── data/
│   ├── raw/                # original dataset (not included)
│   ├── interim/            # sampled/cleaned data
│   └── processed/          # final processed dataset
│
├── models/
│   ├── rf_pipeline.joblib  # saved RandomForest model
│   └── xgb_pipeline.joblib # saved XGBoost model
│
├── src/
│   ├── data.py             # data loading/sampling
│   ├── features.py         # feature engineering
│   ├── train.py            # RF training
│   ├── train_both.py       # RF + XGB training
│   └── api.py              # FastAPI backend
│
├── streamlit_app.py        # UI
├── requirements.txt
└── README.md

⚙️ Running the Project
1️⃣ Create virtual env
python -m venv .venv
.venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the models

Random Forest only:

python -m src.train


Random Forest + XGBoost:

python -m src.train_both

4️⃣ Run the API
uvicorn src.api:app --reload


API available at:
👉 http://127.0.0.1:8000

5️⃣ Run Streamlit App
streamlit run streamlit_app.py


UI available at:
👉 http://localhost:8501

🖼️ Screenshots (Add Yours Here)

You can upload your own:

Streamlit interface

Model comparison charts

API testing demo

🚀 Future Enhancements

Deploy ML model to cloud (AWS / Render / Railway)

Add SHAP explainability

Add LSTM/Deep Learning models

Add accident location map visualization

Add live weather API for real-time predictions

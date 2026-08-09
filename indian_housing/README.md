# 🏠 Indian Housing Price Prediction

A machine learning project to predict housing prices across India using property features, location data, and amenities. The system follows a modular architecture from data ingestion to deployment, with a FastAPI backend and Streamlit frontend.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Implementation Plan](#implementation-plan)
- [Architecture](#architecture)
- [Success Metrics](#success-metrics)

## 🎯 Overview

This project predicts house prices (in Lakhs) for the Indian housing market using a dataset of 250,000 property listings. It leverages features such as location (City, Locality, State), structural details (BHK, Size, Floor), and environmental factors (Nearby Schools, Hospitals, Public Transport) to estimate property values.

The end-to-end pipeline includes:

1. **Data Ingestion** — Load and validate raw CSV data
2. **EDA** — Analyze distributions, correlations, and data quality
3. **Preprocessing** — Clean data, encode categoricals, scale numericals
4. **Modeling** — Train XGBoost/LightGBM models with hyperparameter tuning
5. **API** — Serve predictions via FastAPI
6. **Frontend** — Interactive Streamlit dashboard for buyers and sellers
7. **Deployment** — Docker containerization with docker-compose

## 📁 Project Structure

```
indian_housing/
├── data/
│   ├── raw/                      # Raw CSV data (gitignored)
│   │   └── india_housing_prices.csv
│   └── processed/                # Processed data (gitignored)
│       └── .gitkeep
├── models/                       # Trained model artifacts (gitignored)
│   └── .gitkeep
├── notebooks/                    # Jupyter notebooks for EDA and analysis
├── src/                          # Source code modules
│   ├── __init__.py
│   └── data_ingestion.py         # Data loading and validation
├── venv/                         # Python virtual environment (gitignored)
├── ARCHITECTURE.md               # System architecture documentation
├── Implementation-plan.md        # Phase-wise implementation plan
├── PROBLEM_Readme.md             # Problem statement
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📊 Dataset

- **File**: `data/raw/india_housing_prices.csv`
- **Records**: 250,000 property listings
- **Features**: 23 columns including:
  - **Location**: State, City, Locality
  - **Structure**: Property_Type, BHK, Size_in_SqFt, Floor_No, Total_Floors
  - **Price**: Price_in_Lakhs, Price_per_SqFt
  - **Age**: Year_Built, Age_of_Property
  - **Furnishing**: Furnished_Status
  - **Amenities**: Amenities (multi-label string)
  - **Environment**: Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility
  - **Other**: Parking_Space, Security, Facing, Owner_Type, Availability_Status

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Ajitkumar04/Indian-Housing-Machine-Learning-Project-
cd indian_housing
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Run Data Ingestion

```bash
python src/data_ingestion.py
```

This loads the raw CSV, validates mandatory columns, and prints the first 5 rows.

### Run Data Ingestion as a Module

```python
from src.data_ingestion import ingest_data

df = ingest_data("data/raw/india_housing_prices.csv")
```

### Run the FastAPI Service

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` to access the automatic Swagger UI and test `/predict`.

### Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://127.0.0.1:8501` to use the dashboard.

### Run with Docker Compose

```bash
docker compose up --build
```

This starts:
- `api` on port `8000`
- `streamlit` on port `8501`

## 📅 Implementation Plan

See [Implementation-plan.md](Implementation-plan.md) for the full phase-wise breakdown:

| Phase | Title | Status |
|-------|-------|--------|
| Phase 0 | Project Foundation & Environment Setup | 🟡 In Progress |
| Phase 1 | Project Setup & Data Ingestion | ✅ Completed |
| Phase 2 | Exploratory Data Analysis (EDA) | ⬜ Pending |
| Phase 3 | Preprocessing & Feature Engineering | ✅ Completed |
| Phase 4 | Model Development & Evaluation | ✅ Completed |
| Phase 5 | API Development (FastAPI) | ✅ Completed |
| Phase 6 | Frontend Development (Streamlit) | ✅ Completed |
| Phase 7 | Containerization & Documentation | ✅ Completed |

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system architecture, including the data flow from ingestion through preprocessing, modeling, deployment, and the user interface.

## 📈 Success Metrics

1. **Model Performance**: MAE < ₹10 Lakhs
2. **System Latency**: Prediction response time < 200ms
3. **Usability**: Intuitive Streamlit interface requiring zero technical knowledge

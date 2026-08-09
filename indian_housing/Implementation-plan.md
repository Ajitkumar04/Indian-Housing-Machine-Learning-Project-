# 📅 Phase-wise Implementation Plan: India Housing Price Prediction

This plan breaks down the project into 7 manageable phases to move from a raw dataset to a fully deployed ML application.

---

## 🟢 Phase 1: Project Setup & Data Ingestion
**Goal:** Establish the development environment and load the raw data.
- [x] Initialize Git repository and project structure.
- [x] Setup Python Virtual Environment (`venv` or `conda`).
- [x] Create `requirements.txt` with base libraries (pandas, numpy, scikit-learn, matplotlib, seaborn).
- [x] Write `data_ingestion.py` to load and validate the raw housing CSV file.

## 🟡 Phase 2: Exploratory Data Analysis (EDA)
**Goal:** Understand data distribution, correlations, and quality issues.
- [ ] Create a Jupyter Notebook `01_EDA.ipynb`.
- [ ] Analyze Target Variable (`Price_in_Lakhs`) for skewness and outliers.
- [ ] Correlation analysis: Relationship between `Size_in_SqFt`, `BHK`, and `Price`.
- [ ] Geospatial analysis: Compare average prices across different `Cities` and `States`.
- [ ] Identify missing values and data inconsistencies.

## 🟠 Phase 3: Preprocessing & Feature Engineering
**Goal:** Clean and transform raw data into a format suitable for ML models.
- [ ] **Data Cleaning**: Handle missing values and remove/cap price outliers.
- [ ] **Amenity Parsing**: Transform the `Amenities` string column into multiple binary flags (Multi-hot encoding).
- [ ] **Encoding**: 
    - Target encoding for `Locality` and `City`.
    - One-hot encoding for `Property_Type` and `Furnished_Status`.
- [ ] **Scaling**: Apply StandardScaler/RobustScaler to numerical features.
- [ ] Save the preprocessing pipeline using `joblib` for later use in production.

## 🔴 Phase 4: Model Development & Evaluation
**Goal:** Train, tune, and select the best predictive model.
- [ ] Baseline modeling using Linear Regression and Random Forest.
- [ ] Advanced modeling using **XGBoost** and **LightGBM**.
- [ ] Hyperparameter tuning using **Optuna** or `RandomizedSearchCV`.
- [ ] Evaluate using MAE, RMSE, and R² scores.
- [ ] **Interpretability**: Generate SHAP feature importance plots to explain model decisions.
- [ ] Save the final trained model (`best_model.pkl`).

## 🔵 Phase 5: API Development (FastAPI)
**Goal:** Create a backend service to serve model predictions.
- [x] Set up **FastAPI** structure.
- [x] Create a `/predict` endpoint that accepts property details and returns a price estimate.
- [x] Integrate the preprocessing pipeline and the trained model into the API logic.
- [x] Add basic request validation using Pydantic.
- [x] Test API locally using Swagger UI (`/docs`).

## 🟣 Phase 6: Frontend Development (Streamlit)
**Goal:** Build an interactive UI for buyers and sellers.
- [x] Create `app.py` using **Streamlit**.
- [x] **Buyer View**: Slider and dropdown inputs to check if a specific house is a "Fair Deal".
- [x] **Seller View**: Form to estimate the best listing price for their property.
- [x] Visualize price trends in the selected locality using simple charts.

## ⚪ Phase 7: Containerization & Documentation
**Goal:** Make the project reproducible and deployable.
- [x] Create a `Dockerfile` to containerize the FastAPI and Streamlit apps.
- [x] Write a `docker-compose.yml` to orchestrate the backend and frontend.
- [x] Finalize `README.md` with "How to Run" instructions.
- [x] Perform a final end-to-end test of the entire system.

---

## 📈 Success Metrics
1. **Model Performance**: MAE < ₹10 Lakhs (Target depends on dataset variance).
2. **System Latency**: Prediction response time < 200ms.
3. **Usability**: Intuitive Streamlit interface that requires zero technical knowledge.

# 🏠 Indian Housing Price Prediction — Problem Statement

## 🎯 Problem Overview

The goal of this project is to build a machine learning system that accurately predicts housing prices across India. Given a set of property features, the system should estimate the price of a property in Lakhs (₹1 Lakh = ₹100,000), enabling both **buyers** and **sellers** to make informed decisions.

## 📊 Dataset Description

- **Source File**: `data/raw/india_housing_prices.csv`
- **Total Records**: 250,000 property listings
- **Total Features**: 23 columns

### Feature Breakdown

| Feature | Type | Description |
|---------|------|-------------|
| `ID` | int | Unique identifier for each listing |
| `State` | str | State where the property is located |
| `City` | str | City where the property is located |
| `Locality` | str | Neighborhood/locality within the city |
| `Property_Type` | str | Type of property (Apartment, Independent House, etc.) |
| `BHK` | int | Number of Bedrooms, Hall, Kitchen |
| `Size_in_SqFt` | int | Total area of the property in square feet |
| `Price_in_Lakhs` | float | **Target variable** — Price of the property in Lakhs |
| `Price_per_SqFt` | float | Price per square foot |
| `Year_Built` | int | Year the property was built |
| `Furnished_Status` | str | Furnishing status (Furnished, Unfurnished, Semi-furnished) |
| `Floor_No` | int | Floor number of the property |
| `Total_Floors` | int | Total floors in the building |
| `Age_of_Property` | int | Age of the property in years |
| `Nearby_Schools` | int | Number of schools nearby |
| `Nearby_Hospitals` | int | Number of hospitals nearby |
| `Public_Transport_Accessibility` | str | Accessibility to public transport (High, Low, etc.) |
| `Parking_Space` | str | Availability of parking (Yes/No) |
| `Security` | str | Security availability (Yes/No) |
| `Amenities` | str | Comma-separated list of amenities (Pool, Gym, Garden, etc.) |
| `Facing` | str | Direction the property faces (North, South, East, West) |
| `Owner_Type` | str | Type of owner (Owner, Builder, Broker) |
| `Availability_Status` | str | Availability status (Ready_to_Move, Under_Construction) |

## 🎯 Target Variable

**`Price_in_Lakhs`** — The price of the property in Lakhs (₹1 Lakh = ₹100,000). This is a continuous numerical value that the model will predict.

## 🔑 Key Predictive Features

1. **`Size_in_SqFt`** — Larger properties generally command higher prices
2. **`BHK`** — More bedrooms typically mean higher value
3. **`City`** — Metropolitan cities (Mumbai, Delhi, Bangalore) tend to have higher prices
4. **`Locality`** — Specific neighborhoods can significantly impact pricing
5. **`Furnished_Status`** — Furnished properties are typically more expensive
6. **`Amenities`** — Presence of amenities like Pool, Gym, Clubhouse adds value
7. **`Age_of_Property`** — Newer properties may command premium prices
8. **`Nearby_Schools` / `Nearby_Hospitals`** — Proximity to essential services affects value

## 🏗️ Use Cases

### Buyer View
- Enter property details to check if a quoted price is a "Fair Deal", "Overpriced", or a "Steal"
- Compare the market value of a property against its listing price

### Seller View
- Estimate the optimal listing price for a property based on current market trends
- Understand which features drive the highest value

## 📈 Success Metrics

1. **Model Performance**: Mean Absolute Error (MAE) < ₹10 Lakhs
2. **System Latency**: Prediction response time < 200ms
3. **Usability**: Intuitive Streamlit interface requiring zero technical knowledge

## 🔄 Project Phases

| Phase | Title | Description |
|-------|-------|-------------|
| Phase 0 | Project Foundation | Environment setup, virtual environment, data ingestion, documentation |
| Phase 1 | Project Setup & Data Ingestion | Load and validate raw data |
| Phase 2 | EDA | Analyze distributions, correlations, and data quality |
| Phase 3 | Preprocessing & Feature Engineering | Clean data, encode categoricals, scale numericals |
| Phase 4 | Model Development & Evaluation | Train and tune XGBoost/LightGBM models |
| Phase 5 | API Development | FastAPI backend for serving predictions |
| Phase 6 | Frontend Development | Streamlit dashboard for buyers and sellers |
| Phase 7 | Containerization & Documentation | Docker deployment and final documentation |

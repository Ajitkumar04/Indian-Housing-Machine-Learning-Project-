import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.data_ingestion import load_data

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.joblib"


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


@st.cache_data
def load_dataset():
    df = load_data()
    df.columns = df.columns.str.lower()
    return df


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    return df


def predict_price(preprocessor, model, data: pd.DataFrame) -> float:
    features = preprocessor.transform(data)
    pred = model.predict(features)
    if hasattr(pred, "tolist"):
        pred = pred.tolist()[0]
    return float(pred)


def render_trends(df: pd.DataFrame):
    st.subheader("Locality Price Trends")
    city = st.selectbox("Choose a city", sorted(df["city"].unique()), index=0)
    city_data = df[df["city"] == city]
    locality_avg = city_data.groupby("locality")["price_in_lakhs"].mean().sort_values(ascending=False).head(12)

    st.write(f"Average price in {city} by top localities")
    st.bar_chart(locality_avg)


def buyer_view(preprocessor, model):
    st.header("Buyer View")
    st.write("Estimate whether a listing is a fair deal based on model predictions.")

    with st.form("buyer_form"):
        cols = st.columns(2)
        inputs = {}
        with cols[0]:
            inputs["state"] = st.selectbox("State", sorted(df["state"].unique()))
            inputs["city"] = st.selectbox("City", sorted(df["city"].unique()))
            inputs["locality"] = st.text_input("Locality", value="Locality_84")
            inputs["property_type"] = st.selectbox("Property Type", sorted(df["property_type"].unique()))
            inputs["bhk"] = st.slider("BHK", 1, 6, 2)
            inputs["size_in_sqft"] = st.number_input("Size (sqft)", min_value=100.0, max_value=10000.0, value=850.0)
            inputs["furnished_status"] = st.selectbox("Furnished Status", sorted(df["furnished_status"].unique()))
            inputs["amenities"] = st.text_area("Amenities (comma-separated)", value="Gym, Pool, Garden")
        with cols[1]:
            inputs["facing"] = st.selectbox("Facing", sorted(df["facing"].unique()))
            inputs["owner_type"] = st.selectbox("Owner Type", sorted(df["owner_type"].unique()))
            inputs["availability_status"] = st.selectbox("Availability Status", sorted(df["availability_status"].unique()))
            inputs["public_transport_accessibility"] = st.selectbox(
                "Public Transport", sorted(df["public_transport_accessibility"].unique())
            )
            inputs["parking_space"] = st.selectbox("Parking Space", sorted(df["parking_space"].unique()))
            inputs["security"] = st.selectbox("Security", sorted(df["security"].unique()))
            inputs["floor_no"] = st.number_input("Floor Number", min_value=0, max_value=50, value=2)
            inputs["total_floors"] = st.number_input("Total Floors", min_value=0, max_value=50, value=5)
            inputs["age_of_property"] = st.number_input("Age of Property", min_value=0, max_value=100, value=10)
            inputs["nearby_schools"] = st.number_input("Nearby Schools", min_value=0, max_value=50, value=5)
            inputs["nearby_hospitals"] = st.number_input("Nearby Hospitals", min_value=0, max_value=50, value=2)

        proposed_price = st.number_input("Listing price you found (Price in Lakhs)", min_value=0.0, value=80.0)
        submit = st.form_submit_button("Evaluate Listing")

    if submit:
        X = build_input_dataframe(inputs)
        predicted = predict_price(preprocessor, model, X)
        st.metric("Predicted Fair Price (Lakhs)", f"{predicted:.2f}")

        delta = proposed_price - predicted
        pct_diff = 100 * delta / predicted if predicted else 0.0
        if abs(pct_diff) <= 10:
            verdict = "Fair Deal 👍"
            color = "green"
        elif pct_diff < 0:
            verdict = "Good Deal — below model value"
            color = "green"
        else:
            verdict = "Overpriced — above model value"
            color = "red"

        st.markdown(f"**Verdict:** <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
        st.write(f"Price difference: {delta:.2f} Lakhs ({pct_diff:.1f}%)")


def seller_view(preprocessor, model):
    st.header("Seller View")
    st.write("Estimate a competitive listing price for your property.")

    with st.form("seller_form"):
        cols = st.columns(2)
        inputs = {}
        with cols[0]:
            inputs["state"] = st.selectbox("State", sorted(df["state"].unique()), index=0, key="seller_state")
            inputs["city"] = st.selectbox("City", sorted(df["city"].unique()), index=0, key="seller_city")
            inputs["locality"] = st.text_input("Locality", value="Locality_84", key="seller_locality")
            inputs["property_type"] = st.selectbox("Property Type", sorted(df["property_type"].unique()), index=0, key="seller_property_type")
            inputs["bhk"] = st.slider("BHK", 1, 6, 3, key="seller_bhk")
            inputs["size_in_sqft"] = st.number_input("Size (sqft)", min_value=100.0, max_value=10000.0, value=1200.0, key="seller_size")
            inputs["furnished_status"] = st.selectbox("Furnished Status", sorted(df["furnished_status"].unique()), key="seller_furnished_status")
            inputs["amenities"] = st.text_area("Amenities (comma-separated)", value="Gym, Pool, Garden", key="seller_amenities")
        with cols[1]:
            inputs["facing"] = st.selectbox("Facing", sorted(df["facing"].unique()), key="seller_facing")
            inputs["owner_type"] = st.selectbox("Owner Type", sorted(df["owner_type"].unique()), key="seller_owner_type")
            inputs["availability_status"] = st.selectbox("Availability Status", sorted(df["availability_status"].unique()), key="seller_availability_status")
            inputs["public_transport_accessibility"] = st.selectbox(
                "Public Transport", sorted(df["public_transport_accessibility"].unique()), key="seller_public_transport_accessibility"
            )
            inputs["parking_space"] = st.selectbox("Parking Space", sorted(df["parking_space"].unique()), key="seller_parking_space")
            inputs["security"] = st.selectbox("Security", sorted(df["security"].unique()), key="seller_security")
            inputs["floor_no"] = st.number_input("Floor Number", min_value=0, max_value=50, value=3, key="seller_floor_no")
            inputs["total_floors"] = st.number_input("Total Floors", min_value=0, max_value=50, value=8, key="seller_total_floors")
            inputs["age_of_property"] = st.number_input("Age of Property", min_value=0, max_value=100, value=8, key="seller_age_of_property")
            inputs["nearby_schools"] = st.number_input("Nearby Schools", min_value=0, max_value=50, value=7, key="seller_nearby_schools")
            inputs["nearby_hospitals"] = st.number_input("Nearby Hospitals", min_value=0, max_value=50, value=4, key="seller_nearby_hospitals")

        submit = st.form_submit_button("Estimate Listing Price")

    if submit:
        X = build_input_dataframe(inputs)
        predicted = predict_price(preprocessor, model, X)
        st.metric("Suggested Listing Price (Lakhs)", f"{predicted:.2f}")
        st.write("Use this as a starting point to set a competitive price." )


st.set_page_config(page_title="Indian Housing Price Estimator", layout="wide")

preprocessor, model = load_artifacts()
ensemble_label = "best_model" if MODEL_PATH.exists() else "baseline_best"

df = load_dataset()

st.title("Indian Housing Price Prediction")
st.write(
    "Use the buyer or seller workspace to estimate property price and evaluate listing competitiveness. "
    "The model uses location, structure, furnishing, amenities, and nearby services to make predictions."
)

view = st.sidebar.radio("Choose a mode", ["Buyer", "Seller", "Market Trends"])

if view == "Buyer":
    buyer_view(preprocessor, model)
elif view == "Seller":
    seller_view(preprocessor, model)
else:
    render_trends(df)

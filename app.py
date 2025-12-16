import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =========================
#  CUSTOM STYLING
# =========================

def set_bg_image(image_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.90);
            padding: 2rem 2.5rem;
            border-radius: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =========================
#  CONFIG & CONSTANTS
# =========================

st.set_page_config(
    page_title="Solar Power Prediction Dashboard",
    page_icon="☀️",
    layout="wide"
)

# 🌄 Solar panel + green field background image
set_bg_image(
    "https://images.unsplash.com/photo-1509391366360-2e959784a276"
)

# This is just for UI layout (nice grouping of sliders)
UI_FEATURE_ORDER = [
    "distance-to-solar-noon",
    "humidity",
    "average-wind-speed-(period)",
    "sky-cover",
    "wind-direction",
    "wind-speed",
    "temperature",
]

# =========================
#  LOAD MODEL & SCALER
# =========================

@st.cache_resource
def load_model_and_scaler():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# 🔥 Get the true feature orders from the saved objects
if hasattr(scaler, "feature_names_in_"):
    SCALED_FEATURES = list(scaler.feature_names_in_)
else:
    SCALED_FEATURES = UI_FEATURE_ORDER  # fallback

if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = SCALED_FEATURES  # fallback

# For reference / debugging:
ALL_FEATURES = sorted(set(SCALED_FEATURES) | set(MODEL_FEATURES))

# =========================
#  HELPERS
# =========================

def prepare_input_df(data_dict: dict) -> pd.DataFrame:
    """
    data_dict: {feature_name: value from UI}
    Returns a DataFrame with:
      - scaled columns in SCALED_FEATURES order
      - columns reordered to MODEL_FEATURES for prediction
    """
    # Start from all features the model expects
    base = {feat: data_dict.get(feat, 0.0) for feat in MODEL_FEATURES}
    df = pd.DataFrame([base])

    # Scale only the SCALED_FEATURES columns (scaler's training order)
    to_scale = df[SCALED_FEATURES]
    scaled_arr = scaler.transform(to_scale)
    scaled_df = pd.DataFrame(scaled_arr, columns=SCALED_FEATURES, index=df.index)

    # Replace scaled columns
    df[SCALED_FEATURES] = scaled_df[SCALED_FEATURES]

    # Finally ensure columns are exactly in the order model saw during fit
    df = df[MODEL_FEATURES]
    return df


def predict_single(data_dict: dict) -> float:
    df_for_model = prepare_input_df(data_dict)
    y_pred = model.predict(df_for_model)[0]
    return float(y_pred)


def batch_predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame containing at least all MODEL_FEATURES.
    Returns original DF with extra 'predicted_solar_power' column.
    """
    df = df_raw.copy()

    # Ensure all expected model features exist
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Build dict row-by-row so we reuse the same pipeline
    preds = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        preds.append(predict_single(row_dict))

    df_raw["predicted_solar_power"] = preds
    return df_raw


def get_default_inputs_from_time() -> dict:
    """Generate some default values based on current time."""
    from datetime import datetime
    now = datetime.now()
    hour = now.hour
    distance_to_solar_noon = abs(12 - hour)

    return {
        "distance-to-solar-noon": float(distance_to_solar_noon),
        "humidity": 55.0,
        "average-wind-speed-(period)": 5.0,
        "sky-cover": 40.0,
        "wind-direction": 180.0,
        "wind-speed": 5.0,
        "temperature": 30.0,
    }

# =========================
#  SIDEBAR
# =========================

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Interactive Form", "Batch Upload (CSV)", "Real-Time View"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Solar Power Prediction\n**Method:** Machine Learning")

# =========================
#  MAIN LAYOUT – HEADER
# =========================

st.markdown(
    """
    <h1 style="text-align:center; color:#0b6623; font-size:48px;">
        ☀️ Solar Power Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Interactively estimate solar power output from environmental conditions, "
    "analyze batch data, and view real-time style predictions."
)

st.markdown("---")

# =========================
#  MODE 1 – INTERACTIVE FORM
# =========================

if mode == "Interactive Form":
    st.subheader("🎛️ Manual Input")

    defaults = get_default_inputs_from_time()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        distance = st.slider(
            "Distance to Solar Noon (hours)",
            min_value=0.0,
            max_value=12.0,
            value=float(defaults["distance-to-solar-noon"]),
            step=0.5,
        )
        humidity = st.slider(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["humidity"]),
            step=1.0,
        )

    with col2:
        avg_wind_speed = st.slider(
            "Average Wind Speed (period, m/s)",
            min_value=0.0,
            max_value=40.0,
            value=float(defaults["average-wind-speed-(period)"]),
            step=0.5,
        )
        sky_cover = st.slider(
            "Sky Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["sky-cover"]),
            step=1.0,
        )

    with col3:
        wind_direction = st.slider(
            "Wind Direction (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=float(defaults["wind-direction"]),
            step=5.0,
        )
        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=40.0,
            value=float(defaults["wind-speed"]),
            step=0.5,
        )

    with col4:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=float(defaults["temperature"]),
            step=0.5,
        )

    if st.button("🔮 Predict Solar Power"):
        user_data = {
            "distance-to-solar-noon": distance,
            "humidity": humidity,
            "average-wind-speed-(period)": avg_wind_speed,
            "sky-cover": sky_cover,
            "wind-direction": wind_direction,
            "wind-speed": wind_speed,
            "temperature": temperature,
        }

        prediction = predict_single(user_data)

        # Fancy prediction card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4CAF50, #81C784);
                padding: 25px;
                border-radius: 20px;
                text-align: center;
                color: white;
                font-size: 28px;
                font-weight: bold;
                box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
                margin-bottom: 20px;
            ">
                ☀️ Predicted Solar Power<br>
                {prediction:.2f} units
            </div>
            """,
            unsafe_allow_html=True
        )

        met1, met2 = st.columns(2)
        with met1:
            st.metric("Temperature", f"{temperature:.1f} °C")
        with met2:
            st.metric("Humidity", f"{humidity:.0f} %")

        st.success("Prediction generated successfully!")

        st.write("### Input Summary")
        st.dataframe(pd.DataFrame([user_data]))

# =========================
#  MODE 2 – BATCH CSV UPLOAD
# =========================

elif mode == "Batch Upload (CSV)":
    st.subheader("📂 Batch Prediction from CSV")

    st.markdown(
        """
        **Instructions:**
        - Upload a CSV file containing at least these columns (exact names):
        `distance-to-solar-noon`, `humidity`, `average-wind-speed-(period)`,
        `sky-cover`, `wind-direction`, `wind-speed`, `temperature`
        """
    )

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df_raw = pd.read_csv(file)
        st.write("#### Preview of Uploaded Data")
        st.dataframe(df_raw.head())

        missing = [col for col in MODEL_FEATURES if col not in df_raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("🚀 Run Batch Prediction"):
                try:
                    result_df = batch_predict(df_raw)
                except Exception as e:
                    st.error(f"Error during batch prediction: {e}")
                else:
                    st.write("#### Results (first 10 rows)")
                    st.dataframe(result_df.head(10))

                    # Simple distribution chart
                    st.write("#### Distribution of Predicted Solar Power")
                    fig, ax = plt.subplots()
                    ax.hist(result_df["predicted_solar_power"], bins=20)
                    ax.set_xlabel("Predicted Solar Power")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

                    # Download button
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_out,
                        file_name="solar_power_predictions.csv",
                        mime="text/csv",
                    )

# =========================
#  MODE 3 – REAL-TIME STYLE VIEW
# =========================

elif mode == "Real-Time View":
    st.subheader("⏱️ Real-Time Style Prediction")

    st.markdown(
        """
        This mode simulates a real-time prediction loop.
        Adjust current conditions and see the prediction update.
        """
    )

    try:
        from streamlit_autorefresh import st_autorefresh
        refresh_rate = st.slider("Auto-refresh every (seconds)", 5, 60, 15)
        st_autorefresh(interval=refresh_rate * 1000, key="realtime-refresh")
    except Exception:
        st.info("Install `streamlit-autorefresh` to enable auto-refresh behaviour.")

    defaults = get_default_inputs_from_time()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        distance = st.number_input(
            "Distance to Solar Noon (hours)",
            min_value=0.0,
            max_value=12.0,
            value=float(defaults["distance-to-solar-noon"]),
            step=0.5,
        )
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["humidity"]),
            step=1.0,
        )

    with col2:
        avg_wind_speed = st.number_input(
            "Average Wind Speed (period, m/s)",
            min_value=0.0,
            max_value=40.0,
            value=float(defaults["average-wind-speed-(period)"]),
            step=0.5,
        )
        sky_cover = st.number_input(
            "Sky Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["sky-cover"]),
            step=1.0,
        )

    with col3:
        wind_direction = st.number_input(
            "Wind Direction (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=float(defaults["wind-direction"]),
            step=5.0,
        )
        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=40.0,
            value=float(defaults["wind-speed"]),
            step=0.5,
        )

    with col4:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=float(defaults["temperature"]),
            step=0.5,
        )

    user_data = {
        "distance-to-solar-noon": distance,
        "humidity": humidity,
        "average-wind-speed-(period)": avg_wind_speed,
        "sky-cover": sky_cover,
        "wind-direction": wind_direction,
        "wind-speed": wind_speed,
        "temperature": temperature,
    }

    prediction = predict_single(user_data)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #4CAF50, #81C784);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        ">
            ⏱️ Real-Time Predicted Solar Power<br>
            {prediction:.2f} units
        </div>
        """,
        unsafe_allow_html=True
    )

    met1, met2, met3 = st.columns(3)
    met1.metric("Humidity", f"{humidity:.0f} %")
    met2.metric("Temperature", f"{temperature:.1f} °C")
    met3.metric("Sky Cover", f"{sky_cover:.0f} %")

    st.write("### Current Input State")
    st.dataframe(pd.DataFrame([user_data]))

# =========================
#  FEATURE IMPORTANCE (FOOTER)
# =========================

st.markdown("---")
st.subheader("📊 Model Feature Importance (Descending Order)")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_

    # Use MODEL_FEATURES for correct alignment, sort desc
    importance_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.bar(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance Score")
    ax.set_title("Feature Importance (High → Low)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Sorted Feature Importance Table")
    st.dataframe(importance_df)
else:
    st.info("The current model type does not expose feature importances.")

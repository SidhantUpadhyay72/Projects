import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# Streamlit setup
st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

# Upload files
colA, colB = st.columns(2)
masked_file = colA.file_uploader("Upload 'masked_output1.csv'", type=["csv"])
forecast_file = colB.file_uploader("Upload 'oil_forecast_by_asset_well_field.csv' (Optional)", type=["csv"])

# Load and clean data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'Date' not in df.columns:
        raise ValueError("❌ 'Date' column is missing.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Oil_Production_MT'] = np.clip(df['Oil_Production_MT'], 0, df['Oil_Production_MT'].quantile(0.995))
    return df

def create_lags(df, lags=3):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Oil_Production_MT'].shift(lag)
    return df.dropna()

if masked_file is not None:
    try:
        df = load_data(masked_file)
    except Exception as e:
        st.error(f"❌ Error loading main data: {e}")
        st.stop()

    if forecast_file is not None:
        try:
            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
        except Exception as e:
            forecast_df = None
            st.warning(f"⚠️ Could not read forecast file: {e}")
    else:
        forecast_df = None

    # Required columns check
    required_cols = ['Masked_Asset', 'Masked_Well_no', 'Masked_Field', 'Oil_Production_MT', 'Date']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ CSV must contain: {', '.join(required_cols)}")
        st.stop()

    # Select asset → well → field
    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].dropna().unique()))
    well = st.selectbox("Select Well", sorted(df[df['Masked_Asset'] == asset]['Masked_Well_no'].dropna().unique()))
    field = st.selectbox("Select Field", sorted(df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].dropna().unique()))

    # Date inputs
    col1, col2, col3 = st.columns(3)
    year = col1.text_input("Forecast Start Year", value="2025")
    month = col2.text_input("Month (1–12)", value="6")
    day = col3.text_input("Day (1–31)", value="30")

    # Forecast on button
    if st.button("Generate Forecast"):
        try:
            start_date = datetime(int(year), int(month), int(day))
            subset = df[(df['Masked_Asset'] == asset) &
                        (df['Masked_Well_no'] == well) &
                        (df['Masked_Field'] == field)].sort_values("Date")

            # Create lag features
            lags = 3
            subset = create_lags(subset, lags=lags)

            if subset.shape[0] < lags:
                st.error("❌ Not enough data for forecasting.")
                st.stop()

            X = subset[[f'lag_{i}' for i in range(1, lags + 1)]]
            y = subset['Oil_Production_MT']
            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X, y)

            # Use last known values for forecast
            last_known = subset['Oil_Production_MT'].iloc[-lags:].tolist()
            forecast_dates = pd.date_range(start=start_date, periods=5, freq='MS')
            forecast_values = []

            for d in forecast_dates:
                X_input = np.array(last_known[-lags:]).reshape(1, -1)
                pred = model.predict(X_input)[0]
                forecast_values.append((d, pred))
                last_known.append(pred)

            forecast_df_model = pd.DataFrame(forecast_values, columns=["Date", "Forecast_Model"])

            # Plot
            actual_recent = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] < start_date)]
            fig = go.Figure()

            if not actual_recent.empty:
                fig.add_trace(go.Scatter(
                    x=actual_recent['Date'], y=actual_recent['Oil_Production_MT'],
                    mode='lines+markers', name="Actual (Last 30 days)", line=dict(color="steelblue")
                ))

            fig.add_trace(go.Scatter(
                x=forecast_df_model['Date'], y=forecast_df_model['Forecast_Model'],
                mode='lines+markers', name="Forecast (Model)", line=dict(color="crimson")
            ))

            # Optional forecast overlay
            if forecast_df is not None:
                forecast_match = forecast_df[
                    (forecast_df['Masked_Asset'] == asset) &
                    (forecast_df['Masked_Well_no'] == well) &
                    (forecast_df['Masked_Field'] == field)
                ]
                if not forecast_match.empty:
                    fig.add_trace(go.Scatter(
                        x=forecast_match['Date'], y=forecast_match['Forecast_Oil_Production_MT'],
                        mode='lines+markers', name="Forecast (Uploaded)", line=dict(color="orange", dash="dot")
                    ))

            fig.update_layout(
                title=f"Forecast from {start_date.strftime('%d-%m-%Y')} for {asset} / {well} / {field}",
                xaxis_title="Date", yaxis_title="Oil Production (MT)",
                template="plotly_white", height=550
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df_model.set_index("Date"))

        except Exception as e:
            st.error(f"❌ Forecast error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

col1, col2 = st.columns(2)
historical_file = col1.file_uploader("Upload Historical Data: masked_output1.csv", type=["csv"])
uploaded_forecast_file = col2.file_uploader("Upload External Forecast (next_6_months.csv)", type=["csv"])

if historical_file is not None:
    df = pd.read_csv(historical_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Oil_Production_MT'] = np.clip(df['Oil_Production_MT'], 0, df['Oil_Production_MT'].quantile(0.995))

    external_forecast_df = None
    if uploaded_forecast_file is not None:
        try:
            external_forecast_df = pd.read_csv(uploaded_forecast_file)
            external_forecast_df['Date'] = pd.to_datetime(external_forecast_df['Date'], errors='coerce')
        except Exception as e:
            st.warning(f"⚠️ Could not read forecast file: {e}")
            external_forecast_df = None

    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].dropna().unique()))
    well = st.selectbox("Select Well", sorted(df[df['Masked_Asset'] == asset]['Masked_Well_no'].dropna().unique()))
    field = st.selectbox("Select Field", sorted(df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].dropna().unique()))

    subset = df[(df['Masked_Asset'] == asset) & 
                (df['Masked_Well_no'] == well) & 
                (df['Masked_Field'] == field)].sort_values("Date")

    def create_features(data, lags=3):
        for i in range(1, lags + 1):
            data[f'lag_{i}'] = data['Oil_Production_MT'].shift(i)
        return data.dropna()

    subset = create_features(subset)

    if subset.shape[0] < 10:
        st.warning("❌ Not enough data to build model.")
        st.stop()

    # Train model
    X = subset[[f'lag_{i}' for i in range(1, 4)]]
    y = subset['Oil_Production_MT']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    # Forecast next 6 months
    last_known = subset['Oil_Production_MT'].iloc[-3:].tolist()
    last_date = subset['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
    forecast_values = []

    for date in forecast_dates:
        X_input = np.array(last_known[-3:]).reshape(1, -1)
        pred = model.predict(X_input)[0]
        forecast_values.append((date, pred))
        last_known.append(pred)

    model_forecast_df = pd.DataFrame(forecast_values, columns=["Date", "Forecast_Model"])
    model_forecast_df['Masked_Asset'] = asset
    model_forecast_df['Masked_Well_no'] = well
    model_forecast_df['Masked_Field'] = field

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset['Date'], y=subset['Oil_Production_MT'],
        name="Actual", mode='lines+markers', line=dict(color="steelblue")
    ))

    fig.add_trace(go.Scatter(
        x=model_forecast_df['Date'], y=model_forecast_df['Forecast_Model'],
        name="Model Forecast", mode='lines+markers', line=dict(color="crimson")
    ))

    if external_forecast_df is not None:
        match = external_forecast_df[
            (external_forecast_df['Masked_Asset'] == asset) &
            (external_forecast_df['Masked_Well_no'] == well) &
            (external_forecast_df['Masked_Field'] == field)
        ]
        if not match.empty:
            fig.add_trace(go.Scatter(
                x=match['Date'], y=match['Forecast_Oil_Production_MT'],
                name="Uploaded Forecast", mode='lines+markers', line=dict(color="orange", dash="dot")
            ))

    fig.update_layout(title="Oil Production Forecast vs Actual", xaxis_title="Date", yaxis_title="Oil Production (MT)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast input by date
    st.subheader("🔍 Lookup Forecast by Date")
    forecast_input_date = st.date_input("Enter a date (within next 6 months):", value=datetime.today())

    forecast_for_date = None
    input_dt = pd.to_datetime(forecast_input_date)

    # Check in model forecast
    match_model = model_forecast_df[model_forecast_df['Date'] == input_dt]
    if not match_model.empty:
        forecast_for_date = match_model['Forecast_Model'].values[0]
        st.success(f"📈 Model Forecast for {input_dt.strftime('%d-%b-%Y')}: **{forecast_for_date:.2f} MT**")

    # Check in uploaded forecast
    if external_forecast_df is not None:
        match_uploaded = external_forecast_df[
            (external_forecast_df['Masked_Asset'] == asset) &
            (external_forecast_df['Masked_Well_no'] == well) &
            (external_forecast_df['Masked_Field'] == field) &
            (external_forecast_df['Date'] == input_dt)
        ]
        if not match_uploaded.empty:
            forecast_uploaded = match_uploaded['Forecast_Oil_Production_MT'].values[0]
            st.info(f"📤 Uploaded Forecast for {input_dt.strftime('%d-%b-%Y')}: **{forecast_uploaded:.2f} MT**")

    if forecast_for_date is None and (external_forecast_df is None or match_uploaded.empty):
        st.warning("❌ No forecast available for the selected date.")

    # Download button
    st.download_button(
        label="📥 Download Model Forecast CSV",
        data=model_forecast_df.to_csv(index=False).encode('utf-8'),
        file_name='model_forecast.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload the required historical CSV file to begin.")
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

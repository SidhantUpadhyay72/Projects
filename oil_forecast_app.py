import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

# Upload both required files
colA, colB = st.columns(2)
masked_file = colA.file_uploader("Upload 'masked_output1.csv' (Raw Production Data)", type=["csv"])
forecast_file = colB.file_uploader("Upload 'oil_forecast_by_asset_well_field.csv' (Optional Forecast)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df['Oil_Production_MT'] = np.clip(df['Oil_Production_MT'], 0, df['Oil_Production_MT'].quantile(0.995))
    return df

def create_lag_features(df, lags=3):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Oil_Production_MT'].shift(lag)
    return df.dropna()

if masked_file is not None:
    df = load_data(masked_file)

    # Load forecast file if uploaded
    forecast_df = None
    if forecast_file is not None:
        try:
            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        except Exception as e:
            st.warning(f"⚠️ Could not read forecast file: {e}")
            forecast_df = None

    # Selection Filters
    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
    wells = df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()
    well = st.selectbox("Select Well", sorted(wells))
    fields = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()
    field = st.selectbox("Select Field", sorted(fields))

    # Forecast Start Date
    col1, col2, col3 = st.columns(3)
    year = col1.text_input("Forecast Start Year", value="2025")
    month = col2.text_input("Month (1-12)", value="6")
    day = col3.text_input("Day (1-31)", value="30")

    # Forecast Button
    if st.button("Generate Forecast"):
        try:
            start_date = datetime(int(year), int(month), int(day))

            subset = df[(df['Masked_Asset'] == asset) &
                        (df['Masked_Well_no'] == well) &
                        (df['Masked_Field'] == field)].sort_values("Date")

            subset = create_lag_features(subset)
            if subset.shape[0] < 10:
                st.error("❌ Not enough data after lag creation.")
            else:
                X = subset[[f'lag_{i}' for i in range(1, 4)]]
                y = subset['Oil_Production_MT']
                model = XGBRegressor(n_estimators=100, learning_rate=0.1)
                model.fit(X, y)

                history = subset[subset['Date'] < start_date].copy()
                if history.shape[0] < 3:
                    st.error("❌ Not enough data before forecast start date.")
                else:
                    last_known = history.iloc[-3:]['Oil_Production_MT'].tolist()
                    forecast_dates = pd.date_range(start=start_date, periods=5, freq='MS')
                    forecast_vals = []

                    for d in forecast_dates:
                        X_input = np.array(last_known[-3:]).reshape(1, -1)
                        pred = model.predict(X_input)[0]
                        forecast_vals.append((d, pred))
                        last_known.append(pred)

                    model_forecast = pd.DataFrame(forecast_vals, columns=["Date", "Forecast_Model"])

                    # Plotting
                    actual = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] <= start_date)]
                    fig = go.Figure()

                    if not actual.empty:
                        fig.add_trace(go.Scatter(
                            x=actual['Date'], y=actual['Oil_Production_MT'],
                            mode='lines+markers', name="Actual (Last 30 days + Start Date)",
                            line=dict(color="steelblue")
                        ))

                    fig.add_trace(go.Scatter(
                        x=model_forecast["Date"], y=model_forecast["Forecast_Model"],
                        mode='lines+markers', name="Forecast (Model)", line=dict(color="crimson")
                    ))

                    if forecast_df is not None:
                        uploaded_forecast = forecast_df[
                            (forecast_df['Masked_Asset'] == asset) &
                            (forecast_df['Masked_Well_no'] == well) &
                            (forecast_df['Masked_Field'] == field)
                        ]
                        if not uploaded_forecast.empty:
                            fig.add_trace(go.Scatter(
                                x=uploaded_forecast['Date'],
                                y=uploaded_forecast['Forecast_Oil_Production_MT'],
                                mode='lines+markers',
                                name="Forecast (Uploaded)",
                                line=dict(color="orange", dash="dot")
                            ))

                    fig.update_layout(
                        title=f"Forecast from {start_date.strftime('%d-%m-%Y')} for {asset} / {well} / {field}",
                        xaxis_title="Date",
                        yaxis_title="Oil Production (MT)",
                        template="plotly_white",
                        height=550
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(model_forecast.set_index("Date"))

        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

# File upload
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

    # Optional forecast CSV
    forecast_df = None
    if forecast_file is not None:
        try:
            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        except Exception as e:
            st.warning(f"⚠️ Could not read forecast file: {e}")

    # Selection
    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
    wells = df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()
    well = st.selectbox("Select Well", sorted(wells))
    fields = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()
    field = st.selectbox("Select Field", sorted(fields))

    # Date input
    forecast_date = st.date_input("Select Forecast Start Date", value=datetime.today())

    if st.button("🔮 Generate 30-Day Forecast"):
        subset = df[(df['Masked_Asset'] == asset) &
                    (df['Masked_Well_no'] == well) &
                    (df['Masked_Field'] == field)].sort_values("Date")

        subset = create_lag_features(subset)
        if subset.shape[0] < 10:
            st.error("❌ Not enough data after lag feature creation.")
        else:
            X = subset[[f'lag_{i}' for i in range(1, 4)]]
            y = subset['Oil_Production_MT']
            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X, y)

            history = subset[subset['Date'] < pd.to_datetime(forecast_date)].copy()
            if history.shape[0] < 3:
                st.error("❌ Not enough past data to start forecast from this date.")
            else:
                last_known = history.iloc[-3:]['Oil_Production_MT'].tolist()
                forecast_dates = [forecast_date + timedelta(days=i) for i in range(30)]
                forecast_vals = []

                for d in forecast_dates:
                    X_input = np.array(last_known[-3:]).reshape(1, -1)
                    pred = model.predict(X_input)[0]
                    forecast_vals.append((d, pred))
                    last_known.append(pred)

                forecast_df_30 = pd.DataFrame(forecast_vals, columns=["Date", "Forecast_Model"])
                forecast_point = forecast_df_30[forecast_df_30['Date'] == pd.to_datetime(forecast_date)]

                st.success(f"📅 Forecasted Oil Production on {forecast_date.strftime('%d-%m-%Y')}: **{forecast_point['Forecast_Model'].values[0]:.2f} MT**")

                # Plot
                fig = go.Figure()

                actual = subset[(subset['Date'] >= pd.to_datetime(forecast_date) - timedelta(days=30)) & 
                                (subset['Date'] < pd.to_datetime(forecast_date))]

                if not actual.empty:
                    fig.add_trace(go.Scatter(
                        x=actual['Date'], y=actual['Oil_Production_MT'],
                        mode='lines+markers', name="Actual (Last 30 days)",
                        line=dict(color="steelblue")
                    ))

                fig.add_trace(go.Scatter(
                    x=forecast_df_30["Date"], y=forecast_df_30["Forecast_Model"],
                    mode='lines+markers', name="Forecast (Next 30 days)", line=dict(color="crimson")
                ))

                # Optional uploaded forecast
                if forecast_df is not None:
                    uploaded = forecast_df[
                        (forecast_df['Masked_Asset'] == asset) &
                        (forecast_df['Masked_Well_no'] == well) &
                        (forecast_df['Masked_Field'] == field)
                    ]
                    if not uploaded.empty:
                        fig.add_trace(go.Scatter(
                            x=uploaded['Date'], y=uploaded['Forecast_Oil_Production_MT'],
                            mode='lines+markers', name="Forecast (Uploaded)", line=dict(color="orange", dash="dot")
                        ))

                fig.update_layout(
                    title=f"⛽ Forecast from {forecast_date.strftime('%d-%m-%Y')} for {asset} / {well} / {field}",
                    xaxis_title="Date",
                    yaxis_title="Oil Production (MT)",
                    height=550,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df_30.set_index("Date"))

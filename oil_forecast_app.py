import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# Streamlit page config
st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

# Upload section
col1, col2 = st.columns(2)
raw_file = col1.file_uploader("Upload 'masked_output1.csv'", type=["csv"])
optional_forecast_file = col2.file_uploader("Upload optional 'oil_forecast_by_asset_well_field.csv'", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df['Oil_Production_MT'] = np.clip(df['Oil_Production_MT'], 0, df['Oil_Production_MT'].quantile(0.995))
    return df

def create_lag_features(df, lags=3):
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Oil_Production_MT'].shift(i)
    return df.dropna()

if raw_file is not None:
    df = load_data(raw_file)

    # Forecast CSV (optional)
    uploaded_forecast = None
    if optional_forecast_file is not None:
        try:
            uploaded_forecast = pd.read_csv(optional_forecast_file)
            uploaded_forecast['Date'] = pd.to_datetime(uploaded_forecast['Date'])
        except Exception as e:
            st.warning(f"⚠️ Could not read forecast file: {e}")

    # UI: Asset → Well → Field
    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
    well = st.selectbox("Select Well", sorted(df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()))
    field = st.selectbox("Select Field", sorted(df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()))

    # Forecast date
    forecast_date = st.date_input("Select Forecast Start Date", value=datetime.today().date())

    if st.button("🔮 Generate Forecast"):
        subset = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well) & (df['Masked_Field'] == field)].sort_values("Date")

        subset = create_lag_features(subset)
        if subset.shape[0] < 10:
            st.error("❌ Not enough data for selected well/field/asset.")
        else:
            X = subset[[f'lag_{i}' for i in range(1, 4)]]
            y = subset['Oil_Production_MT']

            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X, y)

            # Use data before the forecast date
            history = subset[subset['Date'].dt.date < forecast_date]
            if history.shape[0] < 3:
                st.error("❌ Not enough history before selected date.")
            else:
                last_known = history.tail(3)['Oil_Production_MT'].tolist()
                forecast_dates = [datetime.combine(forecast_date, datetime.min.time()) + timedelta(days=i) for i in range(30)]

                forecast_values = []
                for d in forecast_dates:
                    X_input = np.array(last_known[-3:]).reshape(1, -1)
                    pred = model.predict(X_input)[0]
                    forecast_values.append((d, pred))
                    last_known.append(pred)

                forecast_df = pd.DataFrame(forecast_values, columns=['Date', 'Forecast_Oil_Production_MT'])

                # ✅ Extract exact date forecast
                forecast_df['Only_Date'] = forecast_df['Date'].dt.date
                match_row = forecast_df[forecast_df['Only_Date'] == forecast_date]

                if not match_row.empty:
                    forecast_val = match_row.iloc[0]['Forecast_Oil_Production_MT']
                    st.success(f"📅 Forecasted Oil Production on {forecast_date.strftime('%d-%m-%Y')}: **{forecast_val:.2f} MT**")
                else:
                    st.warning(f"⚠️ Could not find forecast for {forecast_date}")

                # 📈 Plot
                fig = go.Figure()

                # Last 30 days actual
                actual = subset[(subset['Date'] >= pd.to_datetime(forecast_date) - timedelta(days=30)) &
                                (subset['Date'] < pd.to_datetime(forecast_date))]
                if not actual.empty:
                    fig.add_trace(go.Scatter(
                        x=actual['Date'], y=actual['Oil_Production_MT'],
                        mode='lines+markers', name='Actual (Last 30 Days)',
                        line=dict(color='steelblue')
                    ))

                # Forecast plot
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'], y=forecast_df['Forecast_Oil_Production_MT'],
                    mode='lines+markers', name='Forecast (Next 30 Days)',
                    line=dict(color='crimson')
                ))

                # Optional uploaded forecast
                if uploaded_forecast is not None:
                    uploaded_subset = uploaded_forecast[
                        (uploaded_forecast['Masked_Asset'] == asset) &
                        (uploaded_forecast['Masked_Well_no'] == well) &
                        (uploaded_forecast['Masked_Field'] == field)
                    ]
                    if not uploaded_subset.empty:
                        fig.add_trace(go.Scatter(
                            x=uploaded_subset['Date'], y=uploaded_subset['Forecast_Oil_Production_MT'],
                            mode='lines+markers', name='Forecast (Uploaded)',
                            line=dict(color='orange', dash='dot')
                        ))

                fig.update_layout(
                    title=f"Forecast from {forecast_date.strftime('%d-%m-%Y')} — {asset} / {well} / {field}",
                    xaxis_title="Date",
                    yaxis_title="Oil Production (MT)",
                    template="plotly_white",
                    height=550
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df[['Date', 'Forecast_Oil_Production_MT']].set_index('Date'))

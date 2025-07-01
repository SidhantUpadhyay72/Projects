import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# LangChain (Updated for new version)
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Load API key securely
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard")

# File Upload
col1, col2 = st.columns(2)
masked_file = col1.file_uploader("Upload Raw Production Data (masked_output1.csv)", type="csv")
forecast_file = col2.file_uploader("Upload Optional Forecast (oil_forecast_by_asset_well_field.csv)", type="csv")

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

# Main Logic
if masked_file is not None:
    df = load_data(masked_file)

    if forecast_file is not None:
        forecast_df = pd.read_csv(forecast_file)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    else:
        forecast_df = None

    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
    well = st.selectbox("Select Well", sorted(df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()))
    field = st.selectbox("Select Field", sorted(df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()))

    col3, col4, col5 = st.columns(3)
    year = col3.text_input("Forecast Start Year", value="2025")
    month = col4.text_input("Month (1-12)", value="6")
    day = col5.text_input("Day (1-31)", value="30")

    if st.button("Generate Forecast"):
        try:
            start_date = datetime(int(year), int(month), int(day))
            subset = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well) & (df['Masked_Field'] == field)].sort_values("Date")
            subset = create_lag_features(subset)

            if subset.shape[0] < 10:
                st.error("❌ Not enough data for training.")
            else:
                X = subset[[f'lag_{i}' for i in range(1, 4)]]
                y = subset['Oil_Production_MT']

                model = XGBRegressor(n_estimators=100, learning_rate=0.1)
                model.fit(X, y)

                history = subset[subset['Date'] < start_date].copy()
                if history.shape[0] < 3:
                    st.error("❌ Not enough history to forecast.")
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

                    actual = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] < start_date)]

                    fig = go.Figure()
                    if not actual.empty:
                        fig.add_trace(go.Scatter(x=actual['Date'], y=actual['Oil_Production_MT'], mode='lines+markers', name="Actual (Last 30 days)", line=dict(color="steelblue")))

                    fig.add_trace(go.Scatter(x=model_forecast['Date'], y=model_forecast['Forecast_Model'], mode='lines+markers', name="Forecast (Model)", line=dict(color="crimson")))

                    if forecast_df is not None:
                        uploaded_forecast = forecast_df[(forecast_df['Masked_Asset'] == asset) & (forecast_df['Masked_Well_no'] == well) & (forecast_df['Masked_Field'] == field)]
                        if not uploaded_forecast.empty:
                            fig.add_trace(go.Scatter(x=uploaded_forecast['Date'], y=uploaded_forecast['Forecast_Oil_Production_MT'], mode='lines+markers', name="Forecast (Uploaded)", line=dict(color="orange", dash="dot")))

                    fig.update_layout(title=f"Forecast from {start_date.strftime('%d-%m-%Y')} for {asset} / {well} / {field}", xaxis_title="Date", yaxis_title="Oil Production (MT)", template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(model_forecast.set_index("Date"))

        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")

    # Sidebar Chatbot
    with st.sidebar:
        st.header("🤖 Ask about your dataset")

        if "agent" not in st.session_state:
            try:
                agent = create_pandas_dataframe_agent(
                    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                    df,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=False
                )
                st.session_state.agent = agent
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")

        if "agent" in st.session_state:
            user_question = st.text_input("Ask a question about the data:")
            if user_question:
                with st.spinner("Generating answer..."):
                    response = st.session_state.agent.run(user_question)
                    st.success(response)

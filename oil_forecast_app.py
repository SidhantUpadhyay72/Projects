import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime
import os

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set page config
st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard with AI Assistant")

# Load API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Upload files
col1, col2 = st.columns(2)
masked_file = col1.file_uploader("Upload 'masked_output1.csv' (Raw Production Data)", type=["csv"])
forecast_file = col2.file_uploader("Upload 'oil_forecast_by_asset_well_field.csv' (Optional Forecast)", type=["csv"])

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
    forecast_df = load_data(forecast_file) if forecast_file else None

    # Chatbot Section
    with st.sidebar:
        st.header("🤖 Ask AI about the Data")
        user_question = st.text_area("Enter your question:")
        if user_question and df is not None:
            chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            agent = create_pandas_dataframe_agent(chat_model, df, verbose=False)
            with st.spinner("Thinking..."):
                response = agent.run(user_question)
            st.success(response)

    # Selection inputs
    asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
    wells = df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()
    well = st.selectbox("Select Well", sorted(wells))
    fields = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()
    field = st.selectbox("Select Field", sorted(fields))

    # Date input
    colA, colB, colC = st.columns(3)
    year = colA.text_input("Forecast Start Year (e.g., 2025)", value="2025")
    month = colB.text_input("Month (1-12)", value="6")
    day = colC.text_input("Day (1-31)", value="30")

    if st.button("Generate Forecast"):
        try:
            start_date = datetime(int(year), int(month), int(day))
            subset = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well) & (df['Masked_Field'] == field)].sort_values("Date")

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
                    actual = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] < start_date)]
                    fig = go.Figure()

                    if not actual.empty:
                        fig.add_trace(go.Scatter(x=actual['Date'], y=actual['Oil_Production_MT'],
                                                 mode='lines+markers', name="Actual (Last 30 days)", line=dict(color="steelblue")))

                    fig.add_trace(go.Scatter(x=model_forecast['Date'], y=model_forecast['Forecast_Model'],
                                             mode='lines+markers', name="Forecast (Model)", line=dict(color="crimson")))

                    if forecast_df is not None:
                        uploaded_forecast = forecast_df[
                            (forecast_df['Masked_Asset'] == asset) &
                            (forecast_df['Masked_Well_no'] == well) &
                            (forecast_df['Masked_Field'] == field)]

                        if not uploaded_forecast.empty:
                            fig.add_trace(go.Scatter(x=uploaded_forecast['Date'], y=uploaded_forecast['Forecast_Oil_Production_MT'],
                                                     mode='lines+markers', name="Forecast (Uploaded)",
                                                     line=dict(color="orange", dash="dot")))

                    fig.update_layout(title=f"Forecast from {start_date.strftime('%d-%m-%Y')} for {asset} / {well} / {field}",
                                      xaxis_title="Date", yaxis_title="Oil Production (MT)",
                                      template="plotly_white", height=550)

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(model_forecast.set_index("Date"))

        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")

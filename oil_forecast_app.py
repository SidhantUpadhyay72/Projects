import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# LangChain & OpenAI Chatbot imports
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain.agents.agent_types import AgentType

st.set_page_config(layout="wide")
st.title("🛢️ Oil Production Forecast Dashboard with AI Chatbot")

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

    # Forecast CSV uploaded (optional)
    forecast_df = None
    if forecast_file is not None:
        try:
            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        except Exception as e:
            st.warning(f"⚠️ Could not read forecast file: {e}")
            forecast_df = None

    # Create chatbot agent
    if "OPENAI_API_KEY" in st.secrets:
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        merged_df = df.copy()
        if forecast_df is not None:
            merged_df = pd.concat([df, forecast_df], axis=0, ignore_index=True, sort=False)
        agent = create_pandas_dataframe_agent(chat_model, merged_df, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=False)
    else:
        st.error("❌ OPENAI_API_KEY not found in st.secrets")

    # Main Layout
    col_main, col_bot = st.columns([3, 1])

    with col_main:
        asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
        wells = df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()
        well = st.selectbox("Select Well", sorted(wells))
        fields = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()
        field = st.selectbox("Select Field", sorted(fields))

        col1, col2, col3 = st.columns(3)
        year = col1.text_input("Forecast Start Year (e.g., 2025)", value="2025")
        month = col2.text_input("Month (1-12)", value="6")
        day = col3.text_input("Day (1-31)", value="30")

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

                        actual = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] < start_date)]
                        fig = go.Figure()

                        if not actual.empty:
                            fig.add_trace(go.Scatter(
                                x=actual['Date'], y=actual['Oil_Production_MT'],
                                mode='lines+markers', name="Actual (Last 30 days)",
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
                                    x=uploaded_forecast['Date'], y=uploaded_forecast['Forecast_Oil_Production_MT'],
                                    mode='lines+markers', name="Forecast (Uploaded)", line=dict(color="orange", dash="dot")
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

    with col_bot:
        st.subheader("🤖 Ask the AI about your Data")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for i, (user, bot) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {user}")
            st.markdown(f"**Bot:** {bot}")

        user_question = st.text_input("Ask a question about the uploaded data:", key="user_input")
        if user_question and 'agent' in locals():
            try:
                response = agent.run(user_question)
                st.session_state.chat_history.append((user_question, response))
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Chatbot error: {e}")

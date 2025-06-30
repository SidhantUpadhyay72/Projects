import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from streamlit_chat import message

# Secure API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page Setup
st.set_page_config(layout="wide")
st.title("🛢️ Oil Forecast Dashboard + 🤖 AI Chatbot")

# Upload CSVs
col1, col2 = st.columns(2)
masked_file = col1.file_uploader("Upload 'masked_output1.csv'", type="csv")
forecast_file = col2.file_uploader("Upload 'oil_forecast_by_asset_well_field.csv'", type="csv")

# Load Helper
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

# Main logic
if masked_file is not None:
    df = load_data(masked_file)
    forecast_df = None

    if forecast_file is not None:
        try:
            forecast_df = pd.read_csv(forecast_file)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        except Exception as e:
            st.warning(f"⚠️ Error reading forecast file: {e}")

    # === Tabs for Forecasting and Chatbot ===
    tab1, tab2 = st.tabs(["📈 Forecasting Dashboard", "💬 AI Chatbot"])

    # === Tab 1: Forecasting ===
    with tab1:
        st.subheader("Oil Production Forecasting")

        asset = st.selectbox("Select Asset", sorted(df['Masked_Asset'].unique()))
        wells = df[df['Masked_Asset'] == asset]['Masked_Well_no'].unique()
        well = st.selectbox("Select Well", sorted(wells))
        fields = df[(df['Masked_Asset'] == asset) & (df['Masked_Well_no'] == well)]['Masked_Field'].unique()
        field = st.selectbox("Select Field", sorted(fields))

        colA, colB, colC = st.columns(3)
        year = colA.text_input("Forecast Start Year", value="2025")
        month = colB.text_input("Month", value="6")
        day = colC.text_input("Day", value="30")

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

                        actual = subset[(subset['Date'] >= start_date - pd.Timedelta(days=30)) & (subset['Date'] < start_date)]
                        fig = go.Figure()

                        if not actual.empty:
                            fig.add_trace(go.Scatter(x=actual['Date'], y=actual['Oil_Production_MT'],
                                                     mode='lines+markers', name="Actual", line=dict(color="steelblue")))
                        fig.add_trace(go.Scatter(x=model_forecast["Date"], y=model_forecast["Forecast_Model"],
                                                 mode='lines+markers', name="Forecast (Model)", line=dict(color="crimson")))

                        if forecast_df is not None:
                            uploaded_forecast = forecast_df[
                                (forecast_df['Masked_Asset'] == asset) &
                                (forecast_df['Masked_Well_no'] == well) &
                                (forecast_df['Masked_Field'] == field)
                            ]
                            if not uploaded_forecast.empty:
                                fig.add_trace(go.Scatter(x=uploaded_forecast['Date'], y=uploaded_forecast['Forecast_Oil_Production_MT'],
                                                         mode='lines+markers', name="Forecast (Uploaded)", line=dict(color="orange", dash="dot")))

                        fig.update_layout(title=f"Forecast from {start_date.strftime('%d-%m-%Y')}", xaxis_title="Date", yaxis_title="Oil Production (MT)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(model_forecast.set_index("Date"))

            except Exception as e:
                st.error(f"❌ Forecasting Error: {e}")

    # === Tab 2: Chatbot ===
    with tab2:
        st.subheader("Ask the AI about your uploaded data")

        # Prepare documents
        df_masked_text = df.to_csv(index=False)
        docs = [Document(page_content=df_masked_text, metadata={"source": "masked_output1.csv"})]

        if forecast_df is not None:
            df_forecast_text = forecast_df.to_csv(index=False)
            docs.append(Document(page_content=df_forecast_text, metadata={"source": "forecast_file.csv"}))

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=retriever,
            return_source_documents=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask a question:")
        if user_input:
            result = qa_chain({"query": user_input})
            response = result["result"]
            st.session_state.chat_history.append((user_input, response))

        for i, (q, a) in enumerate(st.session_state.chat_history):
            message(q, is_user=True, key=f"user_{i}")
            message(a, key=f"bot_{i}")

else:
    st.warning("⬆️ Please upload at least `masked_output1.csv` to use the dashboard and chatbot.")

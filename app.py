
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import io
import base64

st.set_page_config(page_title="Carbon Footprint Forecast App", layout="wide")
st.title("🌍 Carbon Footprint Forecasting App (2022–2030)")
st.markdown("""
Upload your Excel file and select a region to view historical CO₂ data and forecasted emissions.

💡 Make sure your file follows the required format.
📥 [Download Sample Template](https://chat.openai.com/c/sandbox:/mnt/data/sample_co2_emission_template.xlsx)
""")

uploaded_file = st.file_uploader("📁 Upload Excel File (IEA format with 'World' sheet)", type=['xlsx'])

def preprocess_data(file):
    try:
        df = pd.read_excel(file, sheet_name="World", header=1)
        df = df[df['Category'] == 'CO2 combustion and process']
        df = df[df['Product'] == 'Total']
        df = df[df['Flow'] == 'Total energy supply']
        df = df[df['Unit'] == 'Mt CO2']
        year_cols = [col for col in df.columns if isinstance(col, int)]
        df_long = df.melt(id_vars=['Region'], value_vars=year_cols,
                          var_name='Year', value_name='CO2 (Mt)')
        df_long.dropna(inplace=True)
        return df_long
    except Exception:
        return None

def forecast(df):
    model = LinearRegression()
    X = df['Year'].values.reshape(-1, 1)
    y = df['CO2 (Mt)'].values
    model.fit(X, y)
    future_years = np.arange(2026, 2031)
    y_pred = model.predict(future_years.reshape(-1, 1))
    return pd.DataFrame({'Year': future_years, 'CO2 (Mt)': y_pred})

def get_image_download_link(fig, format='png'):
    buf = io.BytesIO()
    fig.write_image(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/{format};base64,{b64}" download="co2_chart.{format}">📥 Download {format.upper()}</a>'

if uploaded_file:
    df = preprocess_data(uploaded_file)
    if df is not None and 'Region' in df.columns and 'Year' in df.columns:
        regions = sorted(df['Region'].dropna().unique())
        selected_region = st.selectbox("🌐 Select Region", regions)
        df_region = df[df['Region'] == selected_region]
        editable_df = df_region[df_region['Year'] < 2025].copy()

        st.subheader("✏️ Edit Historical Data (2022–2024)")
        edited_df = st.data_editor(editable_df, num_rows="dynamic", key="input_table")

        st.subheader("🔮 Forecast CO₂ Emissions (2026–2030)")
        forecast_df = forecast(edited_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edited_df['Year'], y=edited_df['CO2 (Mt)'],
                                 mode='lines+markers', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['CO2 (Mt)'],
                                 mode='lines+markers', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title=f"CO₂ Emission Forecast for {selected_region}",
                          xaxis_title="Year", yaxis_title="CO₂ (Mt)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("📤 **Export Visualization:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(get_image_download_link(fig, format='png'), unsafe_allow_html=True)
        with col2:
            st.markdown(get_image_download_link(fig, format='pdf'), unsafe_allow_html=True)

        st.subheader("📊 Data Tables")
        st.markdown("**Edited Historical Data**")
        st.dataframe(edited_df)
        st.markdown("**Forecasted Data**")
        st.dataframe(forecast_df)
    else:
        st.warning("⚠️ The uploaded file does not match the required template format. Please check your file or download the sample template above.")
else:
    st.info("⬆️ Upload a file to begin.")

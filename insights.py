import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import openai
from io import BytesIO

# Set OpenAI API Key
openai.api_key ="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"  # Replace with your actual API key

# Ensure required libraries are installed
try:
    import streamlit
    import pandas
    import seaborn
    import matplotlib
except ModuleNotFoundError:
    raise ModuleNotFoundError("Required libraries are not installed. Please run: pip install streamlit pandas seaborn matplotlib numpy")

# Custom CSS for improved styling
custom_css = """
<style>
body {
    background-color: #f5f5f5;
}
header, footer, .stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2c3e50, #3498db);
    color: white;
}
h1, h2, h3 {
    color: #2c3e50;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Function to generate graph description
def generate_graph_description(df, plot_type, selected_col=None, correlation=None):
    if plot_type == 'heatmap':
        return "This heatmap represents missing values in the dataset. Yellow cells indicate missing values, while purple cells represent present data. It helps identify columns with missing data."
    elif plot_type == 'distribution':
        return f"This is the distribution of the numeric column: {selected_col}. The histogram shows the frequency of values, with the KDE curve indicating the estimated distribution shape."
    elif plot_type == 'correlation_heatmap':
        return "This heatmap displays correlations between numeric features in the dataset. Positive correlations are shown in warm colors, while negative correlations are in cool colors, helping identify relationships between variables."
    elif plot_type == 'bar_plot':
        return f"This bar plot shows the top 10 categories in the column {selected_col}. It helps to visualize the distribution of the most frequent categories."
    elif plot_type == 'time_series':
        return f"This time series plot shows the trend of {selected_col} over time. It helps in analyzing the temporal patterns of the data, such as seasonality or trends."
    else:
        return "Graph description not available."

# Create two columns
col1, col2 = st.columns([1, 2])  

with col1:
    st.image(r"C:\Users\Gayatri\Desktop\Streamlit\statistics-clipart-statistics.jpg", use_column_width=True)

with col2:
    st.markdown("## Streamlit Insights Application")
    st.write("An interactive and user-friendly web application designed for data analysis and visualization.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Visualizations", "LLM Insights"])

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())
    
    if page == "Home":
        st.subheader("🏠 Home")
        st.write("Welcome to the Streamlit Insights Application! Use the sidebar to navigate between Data Analysis, Visualizations, and LLM Insights")
    
    elif page == "Data Analysis":
        st.subheader("🛠 Data Cleaning and Analysis")
        df.dropna(axis=1, how='all', inplace=True)  
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write("*Missing Values Detected:*")
            st.write(missing_values[missing_values > 0])
            df.fillna(df.median(numeric_only=True), inplace=True)
            st.write("Missing numeric values filled with column medians.")
        else:
            st.write("No missing values detected.")
        
        st.write("*Dataset Info:*")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write("*Basic Statistics:*")
        st.write(df.describe())
    
    elif page == "Visualizations":
        st.subheader("📊 Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values Heatmap")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax1)
            st.pyplot(fig1)
            st.write(generate_graph_description(df, 'heatmap'))

        with col2:
            st.markdown("#### Numeric Feature Distribution")
            numeric_cols = df.select_dtypes(include=np.number).columns
            if numeric_cols.size > 0:
                selected_col = st.selectbox("Select a numeric column", numeric_cols)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.histplot(df[selected_col], kde=True, ax=ax2)
                st.pyplot(fig2)
                st.write(generate_graph_description(df, 'distribution', selected_col=selected_col))
            else:
                st.write("No numeric columns found")
        
        st.markdown("---")
        st.markdown("#### Additional Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if numeric_cols.size > 1:
            st.markdown("##### Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
            st.pyplot(fig3)
            st.write(generate_graph_description(df, 'correlation_heatmap'))
            
        categorical_cols = df.select_dtypes(include='object').columns
        if categorical_cols.size > 0:
            st.markdown("##### Categorical Insights")
            selected_cat = st.selectbox("Select a categorical column", categorical_cols)
            top_categories = df[selected_cat].value_counts().head(10)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_categories.index, y=top_categories.values, ax=ax4)
            plt.xticks(rotation=45)
            st.pyplot(fig4)
            st.write(generate_graph_description(df, 'bar_plot', selected_col=selected_cat))
            
        st.markdown("##### Time Series Analysis")
        date_cols = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(date_cols) > 0:
            selected_date = st.selectbox("Select a date column", date_cols)
            df[selected_date] = pd.to_datetime(df[selected_date], errors='coerce')
            df.dropna(subset=[selected_date], inplace=True)
            if numeric_cols.size > 0:
                time_metric = st.selectbox("Select a numeric column for time analysis", numeric_cols)
                if time_metric:
                    time_df = df.groupby(df[selected_date].dt.to_period('M'))[time_metric].mean()
                    fig5, ax5 = plt.subplots(figsize=(6, 4))
                    time_df.plot(ax=ax5, marker='o', linestyle='-', color='b')
                    ax5.set_xlabel("Time (Month)")
                    ax5.set_ylabel(time_metric)
                    ax5.set_title(f"{time_metric} Over Time")
                    st.pyplot(fig5)
                    st.write(generate_graph_description(df, 'time_series', selected_col=selected_date))

    elif page == "LLM Insights":
        st.subheader("🤖 LLM Insights")
        summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        summary += "Columns: " + ", ".join(df.columns) + "\n"
        summary += "Basic Statistics:\n" + df.describe().to_string() + "\n"
        prompt = f"Please analyze the following CSV dataset summary:\n\n{summary}"

        st.markdown("*Dataset Summary:*")
        st.text(summary)
        st.write("Generating LLM insights. Please wait...")
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are an expert data analyst."},
                          {"role": "user", "content": prompt}],
                temperature=0.7
            )
            llm_insights = response.choices[0].message['content']
            st.markdown("### LLM Insights")
            st.write(llm_insights)
        except Exception as e:
            st.error(f"Error generating LLM insights: {e}")

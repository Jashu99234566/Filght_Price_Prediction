import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_eda(df):
    st.title("Exploratory Data Analysis")

    # Ensure necessary features exist
    if 'Duration_Minutes' not in df.columns:
        df['Duration_Minutes'] = (
            df['Duration'].str.extract(r'(\\d+)h').fillna(0).astype(int) * 60 +
            df['Duration'].str.extract(r'(\\d+)m').fillna(0).astype(int)
        )

    if 'Total_Stops_Num' not in df.columns:
        stop_map = {
            'non-stop': 0, '1 stop': 1, '2 stops': 2,
            '3 stops': 3, '4 stops': 4
        }
        df['Total_Stops_Num'] = df['Total_Stops'].map(stop_map)

    # Price Distribution
    st.subheader("Distribution of Flight Prices")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Price'], kde=True, bins=40, ax=ax1)
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = ['Price', 'Dep_Hour', 'Dep_Minute', 'Arrival_Hour', 'Arrival_Minute', 'Duration_Minutes', 'Total_Stops_Num']
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # Boxplot: Price vs Total Stops
    st.subheader("Price vs Total Stops")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x=df['Total_Stops'], y=df['Price'], ax=ax3)
    ax3.set_xlabel("Total Stops")
    ax3.set_ylabel("Price")
    st.pyplot(fig3)

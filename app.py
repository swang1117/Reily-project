import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
import pickle
from PIL import Image
import plotly.graph_objs as go
from scipy import stats
import random


st.set_page_config(page_title="Kroger Sales Prediction", layout="wide")

st.title("Kroger Mayo Sales Prediction")

df = pd.read_csv("usableData.csv")
df.Sales=df.Sales*30
df['yearmonth'] = df['yearmonth'].apply(lambda x: str(x))
df['yearmonth'] = df['yearmonth'].apply(lambda x: datetime.strptime(x, '%Y%m'))

A, B, C, D = st.tabs(['Data Preprocessing', 'Modeling', 'Overview', "Prediction"])

with A:
    st.subheader("Kroger Mayo Sales Data Sample")

    st.dataframe(df.sample(5))

    st.caption('Source of Inflation: http://USINFLATIONCalculator.com')

    st.caption('Source of Employment: https://data.bls.gov/pdq/SurveyOutputServlet')

    st.write('Sales Data Range - 29 March 2021 to 22 February 2023')

with B:

    st.write('After Data Preprocessing step, we tried different models to predict sales')

    st.subheader("Model Evaluation (R Squared)")
    col1, col2, col3, col4= st.columns(4)
    col1.metric('Linear Regression', "0.32")
    col2.metric('Ridge', "0.18")
    col3.metric('Lasso', "0.19")
    col4.metric('DecisionTreeRegressor', "0.82")

    col1, col2, col3 = st.columns(3)
    col1.metric('XGBoost', "0.82")
    col2.metric('AdaBosst', "0.89")
    col3.metric('Random Forest', "0.83")

    image = Image.open('tree.jpg')
    st.image(image, caption='Decision Tree Plot', use_column_width=True)


with C:
    metric = st.selectbox('select a variable',['BP_Price', 'Average_Price', 'Employment', 'Inflation', 'Facebook_Spend','Amazon_Spend', 'Instacart_Spend'])

    fig = go.Figure()

    x = np.arange(len(df['yearmonth']))

    if metric == 'Average_Price':
        y = df['Average_Price']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Average_Price'))
    elif metric == 'Employment':
        y = df['Employment']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Employment'))
    elif metric == 'Amazon_Spend':
        y = df['Amazon_Spend']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Amazon_Spend'))
    elif metric == 'Instacart_Spend':
        y = df['Instacart_Spend']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Instacart_Spend'))
    elif metric == 'BP_Price':
        y = df['BP_Price']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='BP_Price'))
    elif metric == 'Inflation':
        y = df['Inflation']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Inflation'))
    elif metric == 'Facebook_Spend':
        y = df['Facebook_Spend']
        fig.add_trace(go.Scatter(x=df['yearmonth'], y=y, mode='lines', name='Facebook_Spend'))

    # Calculate the trend line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    trendline = slope * x + intercept

    # Add the trend line to the plot
    fig.add_trace(go.Scatter(x=df['yearmonth'], y=trendline, mode='lines', name='Trend Line', line=dict(color='red', dash='dash')))

    # Customize the plot layout
    fig.update_layout(title=metric, xaxis_title='Year-Month', yaxis_title=metric, plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Feature Importance')
    st.dataframe({
        'Variables': ['Employment', 'Average_Price', 'Amazon_Spend', 'BP_Price', 'Inflation', 'Instacart_Spend', 'Facebook_Spend'],
        'Feature Importance': ['51.11%', '17.33%', '12.42%', '11.36%', '4.37%', '3.25%', '0.15%']
    })

pipeline = None

with D:
    if st.button('Build Prediction Model', key='build_model'):
        X = df[["BP_Price", "Average_Price", "Inflation", "Facebook_Spend", "Employment", "Amazon_Spend", "Instacart_Spend"]]
        y = df.Sales
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        pipeline = Pipeline([
            ('ada', AdaBoostRegressor(n_estimators=20, random_state=1234))
        ])
        pipeline.fit(X_train, y_train)

        with open('adapipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

    # Input min-max ranges for each feature
    central_bp_price = st.number_input("BP_Price Value")
    central_avg_price = st.number_input("Average_Price Value")
    central_inflation = st.number_input("Inflation Value")
    central_facebook_spend = st.number_input("Facebook_Spend Value")
    central_employment = st.number_input("Employment Value")
    central_amazon_spend = st.number_input("Amazon_Spend Value")
    central_instacart_spend = st.number_input("Instacart_Spend Value")

    range_width_bp_price = 1
    range_width_avg_price = 1
    range_width_inflation = 0.1
    range_width_facebook_spend = 100
    range_width_employment = 1
    range_width_amazon_spend = 10
    range_width_instacart_spend = 100


    # Predict sales for the specified number of months
    num_months = st.slider('Select the number of months for prediction:', min_value=2, max_value=6)
    future_sales = np.zeros(num_months)

    with open("adapipeline.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)

    # Randomly choose values within the specified ranges for each feature
    future_bp_price = random.uniform(central_bp_price - range_width_bp_price, central_bp_price + range_width_bp_price)
    future_avg_price = random.uniform(central_avg_price - range_width_avg_price, central_avg_price + range_width_avg_price)
    future_inflation = random.uniform(central_inflation - range_width_inflation, central_inflation + range_width_inflation)
    future_facebook_spend = random.uniform(central_facebook_spend - range_width_facebook_spend, central_facebook_spend + range_width_facebook_spend)
    future_employment = random.uniform(central_employment - range_width_employment, central_employment + range_width_employment)
    future_amazon_spend = random.uniform(central_amazon_spend - range_width_amazon_spend, central_amazon_spend + range_width_amazon_spend)
    future_instacart_spend = random.uniform(central_instacart_spend - range_width_instacart_spend, central_instacart_spend + range_width_instacart_spend)

    if st.button("Predict Future Sales", key='predict_sales'):
        if not (future_bp_price and future_avg_price and future_inflation and future_facebook_spend and future_employment and future_amazon_spend and future_instacart_spend):
            st.write('Please enter future parameters first')
        else:
            for i in range(num_months):
                future_data = np.array([[future_bp_price, future_avg_price, future_inflation, future_facebook_spend, future_employment, future_amazon_spend, future_instacart_spend]])
                future_sale = loaded_pipeline.predict(future_data)
                future_sales[i] = future_sale[0]

    # Plot the original sales data
    plt.figure(figsize=(10, 5))
    plt.plot(df['yearmonth'], df['Sales'], label='Original Sales Data')
    plt.xlabel('Year')
    plt.ylabel('Sales')

    # Plot the predicted sales
    future_dates = pd.date_range(df['yearmonth'].iloc[-1] + timedelta(weeks=4), periods=num_months, freq='M')
    plt.plot(future_dates, future_sales, label='Predicted Sales for the Next {} Months'.format(num_months))
    plt.legend()
    plt.title('Kroger Mayo Sales Prediction')
    st.pyplot(plt.gcf())
    st.write(f"Predicted sales value: {future_sales[-1]:.2f}")
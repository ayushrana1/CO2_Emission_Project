import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# Sampling Library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Data Transformation Libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Custom Transformation 
from sklearn.base import BaseEstimator, TransformerMixin

# Data Pipelines 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 

# Machine Learning Models and Evaluation Metrices
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the dataset
file_path = 'C:\\Users\\KIIT\\Reducing heavy vehicle carbon footprint'
df = pd.read_csv('co2 Emissions.csv')

# Preprocess the data
df["Transmission"] = np.where(df["Transmission"].isin(["A4", "A5", "A6", "A7", "A8", "A9", "A10"]), "Automatic", df["Transmission"])
df["Transmission"] = np.where(df["Transmission"].isin(["AM5", "AM6", "AM7", "AM8", "AM9"]), "Automated Manual", df["Transmission"])
df["Transmission"] = np.where(df["Transmission"].isin(["AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10"]), "Automatic with Select Shift", df["Transmission"])
df["Transmission"] = np.where(df["Transmission"].isin(["AV", "AV6", "AV7", "AV8", "AV10"]), "Continuously Variable", df["Transmission"])
df["Transmission"] = np.where(df["Transmission"].isin(["M5", "M6", "M7"]), "Manual", df["Transmission"])

df["Fuel Type"] = np.where(df["Fuel Type"]=="Z", "Premium Gasoline", df["Fuel Type"])
df["Fuel Type"] = np.where(df["Fuel Type"]=="X", "Regular Gasoline", df["Fuel Type"])
df["Fuel Type"] = np.where(df["Fuel Type"]=="D", "Diesel", df["Fuel Type"])
df["Fuel Type"] = np.where(df["Fuel Type"]=="E", "Ethanol(E85)", df["Fuel Type"])
df["Fuel Type"] = np.where(df["Fuel Type"]=="N", "Natural Gas", df["Fuel Type"])

# Reset index
df.reset_index(drop=True, inplace=True)

# Features and target
X = df.drop("CO2 Emissions(g/km)", axis=1)
y = df["CO2 Emissions(g/km)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=145)

# Pipeline for preprocessing
num_features = ["Engine Size(L)","Cylinders","Fuel Consumption Comb (L/100 km)"]
ord_features = ["Fuel Type"]
nominal_features = []
pass_through_cols = []
drop_cols = ["Fuel Consumption City (L/100 km)","Fuel Consumption Hwy (L/100 km)","Fuel Consumption Comb (mpg)","Transmission","Make","Model","Vehicle Class"]

numerical_pipeline = Pipeline([("imputer", SimpleImputer()), ("std scaler", StandardScaler())])
ordinal_pipeline = Pipeline([("ordinal encoder", OrdinalEncoder()), ("std scaling", StandardScaler())])
nominal_pipeline = Pipeline([("one hot encoding", OneHotEncoder())])

preprocessing_pipeline = ColumnTransformer([
    ("numerical pipeline", numerical_pipeline, num_features),
    ("ordinal pipeline", ordinal_pipeline, ord_features),
    ("nominal pipeline", nominal_pipeline, nominal_features),
    ("passing columns", "passthrough", pass_through_cols),
    ("drop columns", "drop", drop_cols)
])

# Transform data
X_train_tr = preprocessing_pipeline.fit_transform(X_train)
X_test_tr = preprocessing_pipeline.transform(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train_tr, y_train)

# Streamlit app
st.title("CO2 Emission Prediction App")

# Input form
st.sidebar.header("Enter Vehicle Details:")
engine_size = st.sidebar.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)
fuel_consumption = st.sidebar.number_input("Fuel Consumption (L/100 km)", min_value=0.0, max_value=30.0, value=10.0)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Premium Gasoline", "Regular Gasoline", "Diesel", "Ethanol(E85)"])
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Automated Manual", "Automatic with Select Shift", "Continuously Variable", "Manual"])

# Prepare input data for prediction
input_data = pd.DataFrame({"Engine Size(L)": [engine_size],
                            "Cylinders": [cylinders],
                            "Fuel Consumption Comb (L/100 km)": [fuel_consumption],
                            "Fuel Type": [fuel_type],
                            "Transmission": [transmission]})

# Transform input data
input_data_tr = preprocessing_pipeline.transform(input_data)

# Predict CO2 emission for the inputted fuel type
predicted_co2 = rf_model.predict(input_data_tr)[0]
st.header("Predicted CO2 Emission:")
st.write(f"The predicted CO2 emission for the given vehicle is: {predicted_co2:.2f} g/km")

# Predict CO2 emission if natural gas was used instead
input_data_natural_gas = input_data.copy()
input_data_natural_gas["Fuel Type"] = "Natural Gas"
input_data_natural_gas_tr = preprocessing_pipeline.transform(input_data_natural_gas)
predicted_co2_natural_gas = rf_model.predict(input_data_natural_gas_tr)[0]
percent_decrease = abs(((predicted_co2 - predicted_co2_natural_gas) / predicted_co2) * 100)
# Display CO2 emission if natural gas was used
st.header("Predicted CO2 Emission with Natural Gas:")

# Display percentage decrease compared to the original prediction

st.warning(f"If Natural Gas was used instead, the CO2 emission would decrease by {percent_decrease:.2f}%.")
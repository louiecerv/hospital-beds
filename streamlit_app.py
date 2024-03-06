import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    if "user_inputs" not in st.session_state:
        st.session_state['user_inputs'] = []

    if "scaler" not in st.session_state:
        st.session_state["scaler"] = StandardScaler()

    if "clf" not in st.session_state:
        st.session_state["clf"] = []

    #initialize the slider variables
    if "initial_payment" not in st.session_state:        
        st.session_state['initial_payment'] = 200
    if "last_payment" not in st.session_state:
        st.session_state['last_payment'] = 12000
    if "credit_score" not in st.session_state:
        st.session_state['credit_score'] = 500
    if "house_number" not in st.session_state:
        st.session_state['house_number'] = 4000

    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")

    # Display the DataFrame with formatting
    form1.title("Predicting Hospital Bed Needs with Decision Trees and Random Forests")
    text = """(c) 2024 Louie F. Cervantes, M.Eng. [Information Engineering] 
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)
                
    form1.subheader('Regression Task: Predicting Hospital Bed Needs with Decision Trees and Random Forests')
    form1.write('Data Source:')
    text = """This data app explores the hospital bed capacity in 306 US hospital referral regions 
    under various COVID-19 infection scenarios. """
    form1.write(text)

    text = """The data originates from the Harvard Global Data Institute and represents various 
    models simulating a 40 percent adult population infection rate with COVID-19. Hospital bed 
    capacity information comes from the American Hospital Association and the American 
    Hospital Directory."""
    form1.write(text)
    
    form1.write('App Functionality:')

    text = """Data Exploration:
    * View information about different hospital referral regions.
    * Analyze statistics on current bed capacity across regions.
        \nScenario Selection: Choose from various COVID-19 infection scenario models 
        developed by Harvard researchers.
        \nPrediction Models:
        \nDecision Tree Regression: This model visually represents the decision-making process 
        for predicting bed needs based on different factors. Users can interactively explore the 
        tree structure and understand the reasoning behind the predictions.
        \nRandom Forest Regression: This model combines multiple decision trees for improved 
        prediction accuracy and robustness. 
        \nComparison: Compare the predictions from both the decision tree and random forest 
        models for each region and scenario. Identify regions with potential bed shortages
        based on the model outputs.
        \nPotential Use Cases:
    \nPublic health officials:** Allocate resources and plan for surge capacity based on predicted bed needs.
    \nHospital administrators:** Prepare for potential bed shortages and make informed staffing decisions.
    \nResearchers:** Gain insights into the factors influencing hospital bed demand during pandemics.
    \nThis data app leverages the power of decision trees and random forests to provide valuable 
    insights for various stakeholders involved in managing hospital bed capacity during
    public health emergencies."""

    # Create the selection of classifier
    
    clf = DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier']
    selected_option = form1.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)        
    else:
        clf = DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)

    #save the clf to session state
    st.session_state['clf'] = clf
                                                                              
    submit1 = form1.form_submit_button("Start")

    if submit1:
        form1 = [];
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2
    form2 = st.form("training")
    form2.subheader('Classifier Training')        

    #load the data and the labels
    dbfile = 'hospitalbeds.csv'
    df = pd.read_csv(dbfile, header=0)

    #display the data set
    form2.write('Browse the dataset')
    form2.write(df)

    form2.write('The dataset descriptive stats')
    form2.write(df.describe().T)

    # Separate features and target variable
    X = df.drop('Hospital Beds Needed, Six Months', axis=1)  # Target variable column name
    y = df['Hospital Beds Needed, Six Months']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features using StandardScaler (recommended)
    scaler = st.session_state["scaler"] 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #save the scaler object for later use in prediction
    st.session_state["scaler"] = scaler
    


    # Create and train the Decision Tree Classifier   
    clf = st.session_state.clf
    clf.fit(X_train_scaled, y_train)
    st.session_state["clf"] = clf

    # Make predictions on the test set
    y_test_pred = clf.predict(X_test_scaled)

    # Calculate R-squared
    r2 = r2_score(y_test, y_test_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_test_pred)

    form2.write("R-squared:", r2)
    form2.write("Mean Squared Error:", mse)

        
    submit2 = form2.form_submit_button("Predict")
    if submit2:        
        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.write('The trained model will predict if a debtor will repay the loan or not')

    initial_payment = form3.slider(
        label="Initial Payment:",
        min_value=100,
        max_value=500,
        on_change=update_values(),
        key="initial_payment",
        value = 300
    )

    last_payment = form3.slider(
        label="Last Payment:",
        min_value=10000,
        max_value=15000,
        on_change=update_values(),
        key="last_payment",
        value = 12000
    )

    credit_score = form3.slider(
        label="Credit Score:",
        min_value=100,
        max_value=1000,
        on_change=update_values(),
        key="credit_score",
        value = 500
    )

    house_number = form3.slider(
        label="House Number:",
        min_value=3000,
        max_value=5000,
        on_change=update_values(),
        key="house_number",
        value = 3700
    )

    update_values()

    predictbn = form3.form_submit_button("Predict")
    if predictbn:
        user_inputs = np.array(st.session_state['user_inputs'])

        scaler = st.session_state["scaler"]
        test_data_scaled =scaler.transform(user_inputs)
        test_data_scaled = np.array(test_data_scaled)
        form3.write('The raw user inputs (not yet encoded to numeric format):')        
        form3.write(user_inputs)
        form3.write('The user inputs encoded to numeric format:')
        form3.write(test_data_scaled)

        predicted =  st.session_state["clf"].predict(test_data_scaled)
        result = 'Will the debtor fully pay their loan? The model predicts: ' + predicted[0]
        form3.subheader(result)

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()
        form3.write('Click reset again to reset this app.')

def update_values():
    """Get the updated values from the sliders."""
    initial_payment = st.session_state['initial_payment']
    last_payment = st.session_state['last_payment']
    credit_score = st.session_state['credit_score']
    house_number = st.session_state['house_number']

    st.session_state['user_inputs'] = [[initial_payment, 
        last_payment, credit_score, house_number]]

if __name__ == "__main__":
    app()

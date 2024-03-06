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
    if "total_beds" not in st.session_state:
        st.session_state.total_beds = 0
                
    if "total_icu_beds" not in st.session_state:
        st.session_state.total_icu_beds = 0
        
    if "available_beds" not in st.session_state:
        st.session_state.available_beds = 0
        
    if "potentially_available" not in st.session_state:
        st.session_state.potentially_available = 0

    if "available_icu_beds" not in st.session_state:
        st.session_state.available_icu_beds = 0

    if "potentially_available_icu_beds" not in st.session_state:
        st.session_state.potentially_available_icu_beds = 0
        
    if "adult_population" not in st.session_state:
        st.session_state.adult_population = 0
        
    if "population_65plus" not in st.session_state:
        st.session_state.population_65plus = 0

    if "projected_infected" not in st.session_state:
        st.session_state.projected_infected = 0

    if "projected_hospitalized" not in st.session_state:
        st.session_state.projected_hospitalized = 0
        
    if "projected_needing_icu" not in st.session_state:
        st.session_state.projected_needing_icu = 0

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
    \nPublic health officials:\nAllocate resources and plan for surge capacity based on predicted bed needs.
    \nHospital administrators:\nPrepare for potential bed shortages and make informed staffing decisions.
    \nResearchers:\nGain insights into the factors influencing hospital bed demand during pandemics.
    \nThis data app leverages the power of decision trees and random forests to provide valuable 
    insights for various stakeholders involved in managing hospital bed capacity during
    public health emergencies."""
    form1.write(text)

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
    form2.subheader('Regressor Training')        

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

    form2.subheader('Regressor Performance')
    # Calculate R-squared
    r2 = r2_score(y_test, y_test_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_test_pred)

    form2.write("R-squared: " + f"{r2:.2f}")
    form2.write("Mean Squared Error: " + f"{mse:.2f}")

        
    submit2 = form2.form_submit_button("Predict")
    if submit2:        
        display_form3()



def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.write("""To test this trained model, input the data below. The trained model 
        will then predict the projected hospital beds required for the infection scenario.""")

    total_beds = form3.slider(
        label="Total Hospital Beds:",
        min_value=200,
        max_value=20000,
        on_change=update_values(),
        key="total_beds",
        value = 1000
    )
    total_icu_beds = form3.slider(
        label="Total ICU Beds:",
        min_value=18,
        max_value=2500,
        on_change=update_values(),
        key="total_icu_beds",
        value = 1500
    )

    available_beds = form3.slider(
        label="Available Hospital Beds:",
        min_value=100,
        max_value=6500,
        on_change=update_values(),
        key="available_beds",
        value = 300
    )

    potentially_available = form3.slider(
        label="Potentially Available Beds:",
        min_value=160,
        max_value=13000,
        on_change=update_values(),
        key="potentially_available",
        value = 5000
    )

    available_icu_beds = form3.slider(
        label="Available ICU Beds:",
        min_value=5,
        max_value=1000,
        on_change=update_values(),
        key="available_icu_beds",
        value = 600
    )

    potentially_available_icu_beds = form3.slider(
        label="Potentially Available ICU Beds:",
        min_value=5,
        max_value=1700,
        on_change=update_values(),
        key="potentially_available_icu_beds",
        value = 600
    )

    adult_population = form3.slider(
        label="Adult Population:",
        min_value=10000,
        max_value=7900000,
        step=100,
        on_change=update_values(),
        key="adult_population",
        value = 50000
    )

    population_65plus = form3.slider(
        label="Population 65+ :",
        min_value=20000,
        max_value=1268000,
        step=100,
        on_change=update_values(),
        key="population_65plus",
        value = 50000
    )    

    projected_infected = form3.slider(
        label="Projected Infected Individuals:",
        min_value=40000,
        max_value=3149000,
        step=100,
        on_change=update_values(),
        key="projected_infected",
        value = 50000
    )   

    projected_hospitalized = form3.slider(
        label="Projected Infected Individuals:",
        min_value=8600,
        max_value=650000,
        step=100,
        on_change=update_values(),
        key="projected_hospitalized",
        value = 150000
    )   

    projected_needing_icu = form3.slider(
        label="Projected Individuals Needing ICU:",
        min_value=1930,
        max_value=136000,
        step=10,
        on_change=update_values(),
        key="projected_needing_icu",
        value = 10000
    )       

    update_values()

    predictbn = form3.form_submit_button("Predict")
    if predictbn:
        user_inputs = np.array(st.session_state['user_inputs'])
        form3.write(user_inputs)
        scaler = st.session_state["scaler"]
        test_data_scaled =scaler.transform(user_inputs)
        test_data_scaled = np.array(test_data_scaled)
        form3.write('The raw user inputs (not yet encoded to numeric format):')        
        form3.write(user_inputs)
        form3.write('The user inputs encoded to numeric format:')
        form3.write(test_data_scaled)

        predicted =  st.session_state["clf"].predict(test_data_scaled)
        result = 'How many hospital beds are needed? The model predicts: ' +  f"{mspredicted[0]:.2f}" 
        form3.subheader(result)

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()
        form3.write('Click reset again to reset this app.')

def update_values():
    """Get the updated values from the sliders."""
    total_beds = st.session_state.total_beds
    total_icu_beds = st.session_state.total_icu_beds
    available_beds = st.session_state.available_beds
    potentially_available = st.session_state.potentially_available
    available_icu_beds = st.session_state.available_icu_beds
    potentially_available_icu_beds = st.session_state.potentially_available_icu_beds
    adult_population = st.session_state.adult_population
    population_65plus = st.session_state.population_65plus
    projected_infected = st.session_state.projected_infected
    projected_hospitalized = st.session_state.projected_hospitalized
    projected_needing_icu = st.session_state.projected_needing_icu

    st.session_state['user_inputs'] = [[
        total_beds,
        total_icu_beds,
        available_beds,
        potentially_available,
        available_icu_beds,
        potentially_available_icu_beds,
        adult_population,
        population_65plus,
        projected_infected,
        projected_hospitalized,
        projected_needing_icu]]

if __name__ == "__main__":
    app()

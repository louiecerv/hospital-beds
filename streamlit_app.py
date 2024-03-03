import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

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
    form1.title("Loan Repayment Predictor based on the Decision Tree Algorithm")
    text = """Louie F. Cervantes, M.Eng. \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)
                
    form1.write('Replace with the actual description')        
    #insert the rest of the information here

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
    dbfile = 'loan-repayment.csv'
    df = pd.read_csv(dbfile, header=0)

    #display the data set
    form2.write('Browse the dataset')
    form2.write(df)

    form2.write('The dataset descriptive stats')
    form2.write(df.describe().T)

    X = df.values[:,0:-1]
    y = df.values[:,-1]    
    
    fig, ax = plt.subplots(figsize=(6, 2))

    # Create the horizontal barplot
    sns.countplot(y='result', data=df, hue='result', palette='bright', ax=ax)

    # Add the title
    ax.set_title('Distribution of Paid/Not Paid')
    # Display the plot using Streamlit
    form2.pyplot(fig)
    form2.write("""The data shows that the debtors are almost equal between 
    those that paid their loans (yes) and those that did not (no).""")
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42)

    clf = DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    form2.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    form2.text(cm)

    form2.subheader('Performance Metrics')
    form2.text(classification_report(y_test, y_test_pred))
        
    submit2 = form2.form_submit_button("Predict")
    if submit2:        
        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.text('replace with the result of the prediction model.')

    n_clusters = form3.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )

    predictbn = form3.form_submit_button("Predict")
    if predictbn:                    
        form3.text('User selected nclusters = ' + str(n_clusters))

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()

if __name__ == "__main__":
    app()

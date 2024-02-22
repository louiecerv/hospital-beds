import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Support Vector Machine Classifier")
    text = """Louie F. Cervantes, M.Eng. \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.subheader('Description')
    st.write('Support Vector Machines (SVM):')
    text = """Supervised learning algorithm: Used for both classification and regression.
    Linear decision boundary: In high-dimensional spaces, it uses the 
    kernel trick to create a non-linear decision boundary by implicitly 
    mapping data points to higher dimensions.
    Maximizing margin: Aims to find the hyperplane that separates classes 
    with the largest margin (distance between the hyperplane and the closest data 
    points on either side). This makes it robust to noise and outliers.
    """
    st.write(text)
    st.write('Key Features:')
    st.write("""Dataset Generation:
    Randomly generates data points belonging to two clusters using user-defined settings:
    Number of clusters
    Number of data points per cluster
    Cluster means and standard deviations
    Overlap control (overlap_factor) to adjust cluster spread""")
    st.write('SVM Classification:')
    st.write("""Trains an SVM model with the chosen kernel (linear or radial basis function) and hyperparameters.
    Evaluates the model's performance using accuracy, precision, recall, and F1-score.""")
    st.write('Visualization:')
    st.write("""Interactive scatter plot displaying data points colored by 
    their true and predicted classes. Decision boundary overlayed on the plot. 
    Performance metrics displayed dynamically as cluster overlap changes.""")

    # Create a slider with a label and initial value
    n_samples = st.slider(
        label="Number of samples (200 to 4000):",
        min_value=200,
        max_value=4000,
        step=200,
        value=1000,  # Initial value
    )

    cluster_std = st.number_input("Standard deviation (between 0 and 1):")

    random_state = st.slider(
        label="Random seed (between 0 and 100):",
        min_value=0,
        max_value=100,
        value=42,  # Initial value
    )
   
    n_clusters = st.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )
    
    if st.button('Start'):
        centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                        cluster_std=cluster_std, centers = centers,
                        random_state=random_state)
                   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clfSVM = svm.SVC(kernel='linear', C=1000)
        clfSVM.fit(X_train, y_train)
        y_test_pred = clfSVM.predict(X_test)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.write(cm)
        st.subheader('Visualization')

        if n_clusters == 2:
            #use the Numpy array to merge the data and test columns
            dataset = np.column_stack((X, y))

            df = pd.DataFrame(dataset)
            # Add column names to the DataFrame
            df = df.rename(columns={0: 'X', 1: 'Y', 2: 'Class'})
            # Extract data and classes
            x = df['X']
            y = df['Y']
            classes = df['Class'].unique()

            # Create the figure and axes object
            fig, ax = plt.subplots(figsize=(9, 9))
    
            # Scatter plot of the data
            #ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
            sns.scatterplot(
                x = "X",
                y = "Y",
                hue = "Class",
                data = df,
                palette="Set1",
                ax=ax  # Specify the axes object
            )          

            # Plot the decision function directly on ax
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = clfSVM.decision_function(xy).reshape(XX.shape)
    
            ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '--', '--'])
    
            # Plot support vectors
            ax.scatter(clfSVM.support_vectors_[:, 0], clfSVM.support_vectors_[:, 1], s=100, linewidth=1, facecolor='none')
    
            st.pyplot(fig)
        else :
            st.write('Support vectors of n_classes > 2 cannot be plotted on a 2D graph.')

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points

if __name__ == "__main__":
    app()

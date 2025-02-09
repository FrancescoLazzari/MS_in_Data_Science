# ==========================================================================================================================================
# Here in this file we will define all the functions that we will import in our main file 
# in order to mantain the code clean and organized.
# ==========================================================================================================================================

# Importing the necessary libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from IPython.display import Image

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from prince import PCA

from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm

import time
import os
import tqdm.notebook as tqdm

import joblib

import jax
import jax.numpy as jnp

import tensorflow as tf
from flax import linen as nn
from flax.core import frozen_dict
from clu import metrics
from flax.training import train_state, checkpoints
import optax 
from flax import struct  

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------------------------------------------------

# Custom statistic function
def statistic(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The passed dataset must be a pandas DataFrame")

    stat = round(data.describe()[1:].T, 2)

    # Additional statistics
    stat['Range'] = round(stat['max'] - stat['min'], 2)
    stat['IQ Range'] = round(stat['75%'] - stat['25%'], 2)
    stat['% Null Values'] = (data.isnull().sum() / data.shape[0] * 100).round(2).astype(str) + ' %'

    # Capitalize the column names
    stat.columns = [col.capitalize() for col in stat.columns]

    return stat 

# ----------------------------------------------------------------------------------------------------------------------------------------

def corr_matrix(data):
    
    # Plot style and size
    sns.set_style("darkgrid")
    plt.figure(figsize=(25, 20))

    # Correlation matrix
    corr_m = data.corr()

    # Plot of the correlation matrix with a specified color map
    # here we added also a mask to remove the upper triangle of the matrix since it is redundant
    heatmap = sns.heatmap(corr_m, linewidths=0.5, vmin=-1, vmax=1, square=True, annot_kws={"fontsize":12}, linewidth=1.5,
                        cmap="viridis_r", linecolor='white', annot=True, fmt=".2f", cbar_kws={"shrink": 1}, mask=np.triu(corr_m))

    plt.yticks(rotation=0)

    # Color bar customization
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation', rotation=270, labelpad=55, fontsize=20, fontweight='bold')

    # Plot labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14, fontweight='bold', color='black')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14, fontweight='bold', color='black')

    plt.title('Correlation Matrix\n', fontsize=25, fontweight='bold')

    # Save the plot as a .png file in the current working directory
    plt.savefig('Correlation_Matrix.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

# Function to plot the violin plot (i.e. distribution + boxplot) of the features by class
def violin_plot(ds):

    # Plot style and size
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(25, 25))
    axes = axes.flat

    # Subplots execution for each feature
    for i, col in enumerate(ds.columns[:-1]):  
        sns.violinplot(data=ds, x="Class", y=col, 
                    ax=axes[i], palette={"0": "dodgerblue", "1": "darkred"}, 
                    linewidth=2.5, inner_kws=dict(box_width=12, whis_width=2, color=".8")) 
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        # Plot title and grid settings
        axes[i].set_title(f"{col}", color='darkblue', fontweight='bold', fontsize=15)
        axes[i].grid(True, linestyle='--', color='gray', alpha=0.25)
        
        # Set y-axis limits to maximize visualization
        col_min = ds[col].min()
        col_max = ds[col].max()
        axes[i].set_ylim(col_min, col_max)
        
        # Calculate and plot the mean for each class
        axes[i].scatter([0, 1], ds.groupby('Class')[col].mean(), color='orchid', marker='D', s=50, zorder=5, alpha=0.85)

    # Remove empty subplots
    fig.delaxes(axes[ds.shape[1]-1])
    sns.despine()
    fig.suptitle("Violin plot by Class\n", fontsize=35, fontweight="bold")
    fig.tight_layout()

    # Save and show the plot
    plt.savefig('Violin_Plot.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

# Function to plot the QQ plot for each feature and calculate the p-value of the Shapiro-Wilk test
def qq_plot(data):

    # Plot style and size 
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(25, 25))
    axes = axes.flat

    # Sub plots for each feature
    for i, col in enumerate(data.columns):
        qqplot(data[col], line="s", ax=axes[i])
        axes[i].set_title(f"{col}", fontsize=15, fontweight="bold", color="darkblue")
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, linestyle='--', color='gray', alpha=0.25)

        # Calculate the p-value of the Shapiro-Wilk test
        _, p_value = shapiro(data[col])

        # Add the p-value in a transparent box with solid border
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
        axes[i].text(0.05, 0.95, f'p-value: {p_value:.2f}', transform=axes[i].transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

    fig.delaxes(axes[29])
    sns.despine()
    fig.suptitle("QQ Plot with Shapiro-Wilk test p-value\n", fontsize=35, fontweight="bold")
    fig.tight_layout()

    # Save the plot
    plt.savefig('QQ_Plot.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

# Function to visualize if the two class in the train and test set are balanced
def label_proportion(y_train, y_test):

    # Count the occurrences of each label for y_train and y_test
    train_counts = pd.Series(y_train).value_counts(normalize=True)
    test_counts = pd.Series(y_test).value_counts(normalize=True)
    
    # Define colors for the classes
    colors = {0: 'dodgerblue', 1: 'darkred'}
    
    # Plot style and size
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Pie chart for y_train
    axes[0].pie(train_counts, labels=[f'Class {label}' for label in train_counts.index], autopct='%1.1f%%', 
                colors=[colors[label] for label in train_counts.index], startangle=90,
                textprops={'color': 'black'})
    axes[0].set_title('Label Distribution in Training Set', fontsize=25, fontweight='bold')
    
    # Pie chart for y_test
    axes[1].pie(test_counts, labels=[f'Class {label}' for label in test_counts.index], autopct='%1.1f%%', 
                colors=[colors[label] for label in test_counts.index], startangle=90,
                textprops={'color': 'black'})
    axes[1].set_title('Label Distribution in Test Set', fontsize=25, fontweight='bold')
    
    # Display the plots
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

# Function to plot the Kernel Density Estimation with the Normal distribution for each feature
def kde_plot(df):
    # Plot style and size
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(25, 25))
    axes = axes.flat

    # Generate a color palette with as different colors as the number of features
    palette = sns.color_palette("husl", df.shape[1])

    # Sub plots for each feature
    for i, col in enumerate(df.columns):
        # Plot the Kernel Density Estimation with the Normal distribution
        sns.distplot(df[col], kde=True, rug=True, fit=norm, color=palette[i], ax=axes[i])

        axes[i].set_xlabel("")
        axes[i].set_title(f"{col}", fontsize=15, fontweight="bold", color="darkblue")
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, linestyle="--", color="gray", alpha=0.25)

        # Calculate the p-value of the Shapiro-Wilk test
        _, p_value = shapiro(df[col])

        # Add the p-value in a transparent box with solid border
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
        axes[i].text(0.05, 0.95, f'p-value: {p_value:.2f}', transform=axes[i].transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)
    
    # Remove the empty subplots
    fig.delaxes(axes[df.shape[1]])
    fig.suptitle("Kernel Density Estimation vs theoretical Normal distribution\n", fontsize=35, fontweight="bold")
    sns.despine()
    fig.tight_layout()

    # Save the plot
    plt.savefig('KDE_Plot.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

# Function to plot the cumulative explained variance of the Principal Components
def scree_plot(cumulative_eigenvalues, chosen_pc=None):
    # Set the plot style and size
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    # Create the axis values of the plot
    xi = np.arange(1, len(cumulative_eigenvalues) + 1, step=1)

    # Set the plot
    sns.set(style="whitegrid")
    sns.lineplot(x=xi, y=cumulative_eigenvalues, marker='d', color='dodgerblue')
    plt.xlabel('Number of Principal Components', fontsize=14)
    plt.xticks(np.arange(1, len(cumulative_eigenvalues) + 1, step=1))
    plt.ylabel('Cumulative Variance Explained (%)', fontsize=14)
    plt.title('Scree Plot', fontsize=25, fontweight='bold')
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, step=10))

    # Add a light grid
    plt.grid(alpha=0.25, linestyle='--', color='gray', linewidth=0.5)

    # Add vertical and horizontal lines to highlight the chosen PC
    if chosen_pc is not None:
        chosen_variance = cumulative_eigenvalues[chosen_pc - 1]
        plt.axvline(x=chosen_pc, ymin=0, ymax=chosen_variance / 100, color='darkgreen', linestyle='--')
        plt.axhline(y=chosen_variance, xmin=0, xmax=chosen_pc / len(cumulative_eigenvalues), color='darkgreen', linestyle='--')

        # Add a text box with the y-axis value
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='darkgreen')
        plt.text(chosen_pc-0.85, chosen_variance + 3.5, f'{chosen_variance:.2f} %', fontsize=15, verticalalignment='center', bbox=props)

    plt.legend(['Cumulative Variance'], loc='upper left', fontsize=12, frameon=False)

    # Remove top and right borders
    sns.despine()

    # Show the plot
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------


# Function to plot the confusion matrix and the model metrics for both the test and train set
def model_metrics(y_train, y_test, y_train_pred, y_test_pred):
    # Confusion matrix calculation for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]

    # Confusion matrix calculation for train set
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(cm_test, annot=True, fmt=".3f", cmap="Blues", ax=axes[0],
                xticklabels=['Predicted Fraudulent Transaction', 'Predicted Valid Transaction'],
                yticklabels=['Actual Fraudulent Transaction', 'Actual Valid Transaction'])
    axes[0].set_title("Confusion Matrix (Test Set)", fontsize=20, fontweight='bold')

    sns.heatmap(cm_train, annot=True, fmt=".3f", cmap="Greens", ax=axes[1],
                xticklabels=['Predicted Fraudulent Transaction', 'Predicted Valid Transaction'],
                yticklabels=['Actual Fraudulent Transaction', 'Actual Valid Transaction'])
    axes[1].set_title("Confusion Matrix (Train Set)", fontsize=20, fontweight='bold')

    plt.show()

    # Metrics calculation for test set
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    # Metrics calculation for train set
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    # Deltas calculation
    delta_accuracy = abs(accuracy_train - accuracy_test)
    delta_precision = abs(precision_train - precision_test)
    delta_recall = abs(recall_train - recall_test)
    delta_f1 = abs(f1_train - f1_test)

    # Metrics PrettyTable creation
    table = PrettyTable()
    table.title = "\033[1mModel Metrics\033[0m"
    table.field_names = ["\033[95mMetric\033[0m", "\033[95mTest Set\033[0m", "\033[95mTrain Set\033[0m", "\033[95mDelta\033[0m"]
    table.add_row(["\033[94mAccuracy\033[0m", f"{accuracy_test:.3f}", f"{accuracy_train:.3f}", f"{delta_accuracy:.3f}"])
    table.add_row(["\033[96mPrecision\033[0m", f"{precision_test:.3f}", f"{precision_train:.3f}", f"{delta_precision:.3f}"])
    table.add_row(["\033[93mRecall\033[0m", f"{recall_test:.3f}", f"{recall_train:.3f}", f"{delta_recall:.3f}"])
    table.add_row(["\033[91mF1 Score\033[0m", f"{f1_test:.3f}", f"{f1_train:.3f}", f"{delta_f1:.3f}"])

    # Print the table
    print(table)

# ----------------------------------------------------------------------------------------------------------------------------------------

# This function will plot the the Feature Importances for the models that have this attribute otherwise it will print the coefficients
def feature_importances(feature_importances, model_name=None):
    # List of features names
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    # Creation of a DataFrame with the feature importances
    df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    # Order the variables by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Plot the feature importances
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    
    # Set the title and labels accorting to the model
    if model_name == 'Logistic Regression' or model_name == 'SVC':
        plt.xlabel('Coefficients', fontsize=15)
        plt.title('Feature Coefficients', fontsize=20, fontweight='bold')
    else:
        plt.xlabel('Importance', fontsize=15)
        plt.title('Feature Importances', fontsize=20, fontweight='bold')
    
    plt.ylabel('Feature', fontsize=15)
    
    # Global settings for a better visualization
    plt.tight_layout()
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    sns.despine()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

def MLP_plot(metrics_history, n_epoch):
    epoch_range = range(0, n_epoch)  

    # Plot global settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    # Convert the single metrics in a compatible format for the plots
    train_ce_loss = jnp.array(metrics_history['Train Loss'])
    test_ce_loss = jnp.array(metrics_history['Test Loss'])
    train_accuracy = jnp.array(metrics_history['Train Accuracy'])
    test_accuracy = jnp.array(metrics_history['Test Accuracy'])

    # ================
    # CE Loss - Plot
    # ================
    sns.lineplot(ax=axes[0], x=epoch_range, y=train_ce_loss, label='Train Loss')
    sns.lineplot(ax=axes[0], x=epoch_range, y=test_ce_loss, label='Test Loss')
    axes[0].set_title('Binary Cross Entropy Loss over Epochs', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend(prop={'size': 12}, edgecolor='white', facecolor='white', framealpha=0)
    axes[0].grid(True, linestyle='--', color='gray', alpha=0.15)

    # =====================
    # Accuracy - Plot
    # =====================
    sns.lineplot(ax=axes[1], x=epoch_range, y=train_accuracy, label=r'Train Accuracy score')
    sns.lineplot(ax=axes[1], x=epoch_range, y=test_accuracy, label=r'Test Accuracy score')
    axes[1].set_title(r'Accuracy over Epochs', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel(r'Accuracy', fontsize=14)
    axes[1].legend(prop={'size': 12}, edgecolor='white', facecolor='white', framealpha=0)
    axes[1].grid(True, linestyle='--', color='gray', alpha=0.15)
    axes[1].set_ylim([0.45, 1])

    # Additional settings for better visualization
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

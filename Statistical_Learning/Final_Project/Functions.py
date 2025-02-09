import pandas as pd
import numpy as np
from sklearn import datasets
import gudhi as gd
from tqdm import tqdm
import ast

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium

from prince import PCA
from sklearn.preprocessing import StandardScaler

import networkx as nx
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

import warnings
warnings.filterwarnings("ignore")

def plot_graph_with_labels(G):  
    # Draw the graph
    plt.figure(figsize=(15, 10))

    # Use a spring layout for the nodes (this spreads them out in a visually pleasing way)
    pos = nx.spring_layout(G, seed=47)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')

    # Draw the labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    # Remove the axis for clarity
    plt.axis('off')

    # Display the graph
    plt.show()

def cluster_cardinality(data):
    labels = data['Cluster']

    # Identifing the subdataset of the noise points
    noise_mask = labels == -1
    noise_points = data[noise_mask]

    # Identifing the subdataset of the non-noise points
    non_noise_mask = labels != -1
    non_noise_points = data[non_noise_mask]

    # Counting unique labels and their respective sizes (for non-noise points)
    unique_labels, cluster_counts = np.unique(labels[non_noise_mask], return_counts=True)
    
    # Creating a summary DataFrame
    summary_data = {
        "Cluster Label": unique_labels,
        "Number of Points": cluster_counts
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Adding a row for the noise points
    noise_row = pd.DataFrame({"Cluster Label": ["Noise"], "Number of Points": [len(noise_points)]})
    summary_df = pd.concat([noise_row, summary_df], ignore_index=True)

    return summary_df

def pca_3d_scatterplot(df, plot_name):  
    # Extract the principal components
    x = df.iloc[:, 1]
    y = df.iloc[:, 2]
    z = df.iloc[:, 3]
    clusters = df['Cluster']

    #color_map = {0: 'rgb(31, 119, 180)', 1: 'rgb(255, 127, 14)', 2: 'rgb(44, 160, 44)'}

    # Create a trace for each cluster with distinct colors
    data = []
    for cluster_id in clusters.unique():
        cluster_data = df[clusters == cluster_id]
        trace = go.Scatter3d(
            x=cluster_data.iloc[:, 1],
            y=cluster_data.iloc[:, 2],
            z=cluster_data.iloc[:, 3],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.7,
                symbol='circle'
            ),
            name=f'Cluster {cluster_id}'
        )
        data.append(trace)

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='PC 1'),
            yaxis=dict(title='Pc 2'),
            zaxis=dict(title='Pc 3')
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(title='Cluster')
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the plot in the notebook
    pyo.plot(fig, filename=plot_name, auto_open = False);


# Function for a grid search for the optimal parameters (based on the silhouette score) that returns the number of clusters into a valid range
def find_best_hdbscan_params(data, min_cluster_size, min_samples, max_clusters):
    
    result = dict()
    
    # Grid search 
    for min_cluster_size in range(5, min_cluster_size+1, 10):
        for min_samples in range(5, min_samples+1 ,10):

            #print(f'min cluster size={min_cluster_size}, min samples={min_samples}')
            model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = model.fit_predict(data)
            
            # Number of clusters excluding the outilers cluster (label = -1)
            num_clusters = max(model.labels_) + 1
            #print(f'we got {num_clusters} clusters')

            # If the number of clusters is in the range of the desired clusters
            if num_clusters in range(3, max_clusters + 1):
                # Calculus of the silhouette score 
                score = silhouette_score(data, labels)

                # Save the results in the dictionary
                if num_clusters not in result:
                    result[num_clusters] = [(min_cluster_size, min_samples), score]
                else:
                    # Update if we find a better score
                    if score > result[num_clusters][1]:
                        result[num_clusters] = [(min_cluster_size, min_samples), score]
    return result
    
    return best_params, best_score

def find_best_spectral_params(data, max_clusters):
    best_n_cluster = None
    s_scores = list()
    best_score = -1
    x_values = range(2, max_clusters+1)

    # Grid search 
    for n_clusters in x_values:

        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=12)
        labels = model.fit_predict(data)
        
        # Number of clusters excluding the outilers cluster (label = -1)
        num_clusters = max(model.labels_)+1

        # Calculus of the silhouette score 
        score = silhouette_score(data, labels) 
        s_scores.append(score)
        
        # Update the touple with the best parameters if the current parameters got the best score calculated until this point 
        if score > best_score:
            best_score = score
            best_n_cluster = n_clusters
        
    # Create a DataFrame to store x_values and s_scores
    data = pd.DataFrame({
        'Number of Clusters': x_values,
        'Score': s_scores
    })
    
    # Plot
    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    sns.lineplot(data=data, x='Number of Clusters', y='Score', marker='o', color='royalblue')
    plt.axvline(x=best_n_cluster, color='darkgreen', linestyle='--')

    plt.title('Optimal numbers of clusters for Spectral clustering', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Scores')

    plt.grid(axis='x', visible=False) 
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    sns.despine()

    plt.tight_layout()
    plt.show()
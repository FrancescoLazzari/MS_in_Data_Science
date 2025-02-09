
def mse(pr_scores1, pr_scores2):
    """
    Calculate the Mean Squared Error (MSE) between two PageRank score dictionaries

    Arguments of the function:
    pr_scores1 (dict) -> A dictionary of PageRank scores
    pr_scores2 (dict) -> A dictionary of PageRank scores 

    Returns of the function:
    mse (float) -> The calculated MSE score
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    
    import numpy as np
    # Verify that both inputs are dictionaries
    if not isinstance(pr_scores1, dict) or not isinstance(pr_scores2, dict):
        # If one or both inputs has the wrong type rise a ValueError
        raise ValueError("Both inputs must be dictionaries.")
    # Initialize lists to store the scores from each dictionary
    scores1 = []
    scores2 = []
    # Iterate over the nodes in the first dictionary
    for node in pr_scores1:
        # Append the score from the first dictionary and the corresponding score from the second dictionary
        scores1.append(pr_scores1.get(node, 0))
        scores2.append(pr_scores2.get(node, 0))
    # Convert lists to numpy arrays for more efficient numerical operations
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    # Calculate the MSE score
    mse = np.mean((scores1 - scores2) ** 2)

    return mse

#-------------------------------------------------------------------------------------------------------------------------

def score_check(pr_scores1, pr_scores2, epsilon=0.0001):
    """
    Count the number of nodes where the difference in PageRank scores 
    between two dictionaries is less than the specified epsilon.

    Arguments of the function:
    pr_scores1 (dict) -> A dictionary of PageRank scores
    pr_scores2 (dict) -> A dictionary of PageRank scores 
    epsilon(int) -> The maximum difference that the two score can have in absolute value
                    by default is set to 0.0001

    Returns of the function:
    count (int) -> The percentage of nodes wich have a score differences less than epsilon
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    # Verify that both inputs are dictionaries
    if not isinstance(pr_scores1, dict) or not isinstance(pr_scores2, dict):
        # If one or both inputs has the wrong type rise a ValueError
        raise ValueError("Both inputs must be dictionaries.")
    # Initialize a counter
    count = 0  
    # Iterate over each key (node) in the first PageRank scores dictionary
    for key in pr_scores1:
        # Check if the node is present in the second dictionary
        if key in pr_scores2:
            # Compare the scores for the same node between dictionaries
            if abs(pr_scores1[key] - pr_scores2[key]) <= epsilon:
                # Increment the counter if the absolute value of the difference is less or equal than epsilon
                count += 1  
    return f'{(count/len(pr_scores1))*100} %'


#-------------------------------------------------------------------------------------------------------------------------

def extract_graph_data(graph, graph_name):
    '''
    Extract Graph Data

    Arguments of the function:
    graph (nx.Graph) -> A networkx graph 
    graph_name (str) -> A string representing the name of the graph

    Output:
    A dictionary containing all the requested data of the graph
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    # Get the number of nodes in the graph
    num_nodes = len(graph.nodes())
    # Get the number of edges in the graph
    num_edges = len(graph.edges())
    # Calculate the density of the graph
    density = nx.density(graph)

    # Define an internal function to compute the degree distribution in bins
    def compute_degree_distribution(degrees):
        # Define the bins for degree distribution, with intervals of 25
        bins = range(0, max(degrees) + 25, 25)
        # Calculate the histogram of degrees
        hist, bin_edges = np.histogram(degrees, bins=bins)
        # Create a dictionary to map each degree range to its frequency
        distribution = {f"{int(bin_edges[i])}-{int(bin_edges[i+1])-1}": hist[i] for i in range(len(hist))}
        # Return the created sub-dictionary
        return distribution

    # Check if the graph is directed
    if graph.is_directed():
        # Collect the in-degrees for each node
        in_degrees = [d for n, d in graph.in_degree()]
        # Collect the out-degrees for each node
        out_degrees = [d for n, d in graph.out_degree()]
        # Combine in-degrees and out-degrees
        degrees = in_degrees + out_degrees
        # Compute the distribution of in-degrees
        in_degree_distribution = compute_degree_distribution(in_degrees)
        # Compute the distribution of out-degrees
        out_degree_distribution = compute_degree_distribution(out_degrees)
    else:
        # Collect the degrees for each node for an undirected graph
        degrees = [d for n, d in graph.degree()]
        # Compute the degree distribution
        degree_distribution = compute_degree_distribution(degrees)

    # Calculate the average degree
    avg_degree = np.mean(degrees)
    # Define the threshold for identifying hubs (95th percentile)
    hubs_threshold = np.percentile(degrees, 95)
    # Identify hubs as nodes with degrees above the threshold
    hubs = [n for n, d in graph.degree() if d > hubs_threshold]

    # Create a dictionary with the calculated data to return
    data = {
        "Graph Name": graph_name,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Graph Density": density,
        "Graph Type": "dense" if density > 0.5 else "sparse",
        "Average Degree": avg_degree,
        "Graph Hubs": hubs
    }

    # Add the degree distribution to the dictionary, based on whether the graph is directed or not
    if graph.is_directed():
        data["In-Degree Distribution"] = in_degree_distribution
        data["Out-Degree Distribution"] = out_degree_distribution
    else:
        data["Degree Distribution"] = degree_distribution

    # Return the dictionary with the graph data
    return data


#-------------------------------------------------------------------------------------------------------------------------



def node_contribution(graph, node, graph_name):
    '''
    Node Contribution Analysis

    Arguments:
    graph (nx.Graph) -> A networkx graph 
    node (int) -> The node for which we will make the analysis
    graph_name (str) -> The name of the graph.

    Output:
    A dictionary containing the centrality measures for 
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    # Calculate the betweenness centrality 
    betweenness_centrality = nx.centrality.betweenness_centrality(graph)[node]
    # Calculate the PageRank value 
    pagerank = nx.pagerank(graph)[node]
    # Calculate the closeness centrality
    closeness_centrality = nx.centrality.closeness_centrality(graph, u=node)
    # Calculate the degree centrality 
    degree_centrality = nx.centrality.degree_centrality(graph)[node]

    # Return a dictionary with the centrality measures for the node
    return {
        "Node": node,
        "Graph": graph_name,
        "Betweenness Centrality": betweenness_centrality,
        "PageRank": pagerank,
        "Closeness Centrality": closeness_centrality,
        "Degree Centrality": degree_centrality,
    }


#-------------------------------------------------------------------------------------------------------------------------

def shortest_ordered_walk(graph, authors_a, a_1, a_n, N):
    '''
    This function finds the shortest ordered walk in a subgraph

    Argumnets of the function:
    graph (nx.Graph) -> A NetworkX graph
    authors_a (list of str) -> A sequence of authors that the path must traverse
    a_1 (str) -> Starting author
    a_n (str) -> Ending author
    N (int) -> The number of top papers to consider

    Return: 
    A dictionary with the shortest walk and papers crossed
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    # Extract the top N papers in the graph based on degree centrality
    top_n_papers = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]

    # Create a subgraph that includes only these top N papers
    subgraph = graph.subgraph(top_n_papers).copy()

    # Check if both the start author and the end author are present in the subgraph
    if a_1 not in subgraph or a_n not in subgraph:
        return "One or more authors are not present in the graph."

    # Initialize a the list to store the sequence of authors in the shortest walk
    ordered_nodes = [a_1] + authors_a + [a_n]

    # Initialize a lists to store the shortest walk and the papers crossed
    shortest_walk = [] 
    papers_crossed = []

    def bfs_shortest_walk(graph, start, end):
        '''
        This function performs a breadth-first search to find the shortest path between two authors in the graph

        Argumnets of the function:
        graph (nx.DiGraph) -> A NetworkX graph
        start (str) -> Starting author
        end (str) -> Ending author
        
        Return: 
        A list containing the shortest path and the papers encountered
        '''
        # Initialize a queue for the BFS, starting with the start author
        queue = [(start, [start], [])]
        # Set to keep track of visited authors to avoid loops
        visited = set()

        # Loop to explore the graph using breadth-first search
        while queue:
            # Extract the current author, path so far, and papers encountered
            current, walk, papers = queue.pop(0)

            # If the current author is the end author, return the path and papers
            if current == end:
                return walk, papers

            # If the current author hasn't been visited, explore their connections
            if current not in visited:
                # Mark the current author as visited
                visited.add(current)
                # Iterate through each neighbor of the current author
                for neighbor in graph[current]:
                    # Extract paper information from the edge attributes
                    edge_attrs = graph[current][neighbor]
                    # Add the neighbor to the queue for further exploration
                    queue.append((neighbor, walk + [neighbor], papers + edge_attrs.get("titles", [])))

        # If no path is found return empty lists
        return [], []

    # Iterate through each pair of consecutive authors to find the shortest walk between them
    for i in range(len(ordered_nodes) - 1):
        # Find the shortest walk between the current pair of authors
        walk, papers = bfs_shortest_walk(subgraph, ordered_nodes[i], ordered_nodes[i + 1])

        # If no path exists between a pair, return a message
        if not walk:
            return "There is no such path."

        # Add the found path and papers to the respective lists
        shortest_walk.extend(walk[:-1])  # Exclude the last author as it will be included in the next pair
        papers_crossed.extend(papers)

    # Add the final author to complete the walk
    shortest_walk.append(a_n)

    # Return the shortest walk and the papers crossed in the walk
    return {"Shortest Walk": shortest_walk, "Papers Crossed": papers_crossed}


#-------------------------------------------------------------------------------------------------------------------------

def disconnecting_graphs(graph, authorA, authorB, N):
    '''
    Disconnecting Graphs

    Arguments of the function:
    graph (nx.Graph) -> A networkx graph object
    authorA, authorB (str) -> The nodes (authors) to disconnect
    N (int) -> Number of top nodes to consider based on degree centrality

    Output:
    Returns the initial subgraph, the components containing authorA and authorB after disconnection, and the number of edges removed
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    # Select the top N authors based on degree centrality
    top_n_authors = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]
    # Create a subgraph including only the top N authors
    subgraph = graph.subgraph(top_n_authors).copy()
    # Store a copy of the initial subgraph for later comparison
    initial_subgraph = subgraph.copy()
    
    # Check if both authorA and authorB are present in the subgraph
    if authorA not in subgraph or authorB not in subgraph:
        print(f"One or both authors ({authorA}, {authorB}) not present in the graph.")
        return None, None, None, None

    def dfs(graph, visited, start):
        # Perform a depth-first search from a starting node
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                # Add neighbors of the node to the stack, excluding already visited nodes
                stack.extend(set(graph.neighbors(node)) - visited)
        return visited

    def min_edge_cut_between_subgraphs(graph, nodes_A, nodes_B):
        # Calculate the minimum edge cut required to separate two sets of nodes
        min_edge_cut = []
        # Perform DFS to find all nodes connected to nodes_A
        visited_A = dfs(graph, set(), next(iter(nodes_A)))
        for node_A in visited_A:
            for neighbor_B in nodes_B:
                if graph.has_edge(node_A, neighbor_B):
                    # Add the edge to the min-edge cut if it connects nodes_A to nodes_B
                    min_edge_cut.append((node_A, neighbor_B))
        return min_edge_cut

    # Calculate the min-edge cut needed to disconnect authorA from authorB
    min_edge_cut = min_edge_cut_between_subgraphs(subgraph, [authorA], [authorB])
    # Remove the edges from the subgraph to disconnect the authors
    subgraph.remove_edges_from(min_edge_cut)
    # Find connected components in the modified subgraph
    components = list(nx.connected_components(subgraph))
    # Identify the components containing authorA and authorB
    G_a = next((comp for comp in components if authorA in comp), None)
    G_b = next((comp for comp in components if authorB in comp), None)

    # Return the initial subgraph, components containing authorA and authorB, and the number of edges removed
    return initial_subgraph, G_a, G_b, len(min_edge_cut)


#-------------------------------------------------------------------------------------------------------------------------

def extract_communities(graph, N, paper_1, paper_2):
    """
    Extracts communities from a graph using Girvan-Newman algorithm
    
    Arguments of the function:
    graph (nx.Graph) -> The graph from which to extract communities
    N (int) -> Number of top nodes to consider based on degree centrality
    paper_1 (str) -> The first paper
    paper_2 (str) -> The second paper

    Returns:
    A tuple containing the subgraph with top N nodes, number of edges removed, 
    list of communities, and a boolean indicating if paper_1 and paper_2 are in the same community
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    def edge_to_remove_directed(graph):
        # Initialize the minimum weight to infinity and minimum edge to None
        min_weight = float('inf')
        min_edge = None
        # Iterate through all edges in the graph
        for edge in graph.edges(data=True):
            # Check if the weight of the current edge is less than the minimum weight found so far
            if edge[2]['weight'] < min_weight:
                # Update minimum weight and the edge associated with it
                min_weight = edge[2]['weight']
                min_edge = edge[:2]
        # Return the edge with the minimum weight
        return tuple(min_edge)

    def edge_to_remove_undirected(graph):
        # Find the edge with the highest betweenness centrality in the undirected graph
        edge_betweenness = calculate_edge_betweenness(graph)
        return max(edge_betweenness, key=edge_betweenness.get)

    def calculate_edge_betweenness(graph):
        # Initialize an empty dictionary for edge betweenness centrality
        edge_betweenness = {}
        # Iterate through all edges in the graph
        for edge in graph.edges():
            # Calculate betweenness centrality for each edge
            edge_betweenness[edge] = calculate_edge_betweenness_centrality(graph, edge)
        # Return the dictionary containing edge betweenness centrality for all edges
        return edge_betweenness

    def calculate_edge_betweenness_centrality(graph, edge):
        # Create a mapping from each node to its neighbors excluding the other node in the edge
        node_to_neighbors = {v: set(graph.neighbors(v)) - {u} for u, v in graph.edges()}
        # Initialize a dictionary to track the shortest paths starting from each node
        node_to_shortest_paths = {node: {node} for node in graph.nodes()}

        # Perform Breadth-First Search to find shortest paths
        # Initialize a queue with the target node of the edge and a visited set
        queue = [edge[1]]
        visited = set()
        visited.add(edge[1])
        # Process nodes in the queue
        while queue:
            current_node = queue.pop(0)
            # Iterate through the neighbors of the current node
            for neighbor in node_to_neighbors.get(current_node, []):
                if neighbor not in visited:
                    # Mark the neighbor as visited and add it to the queue
                    visited.add(neighbor)
                    queue.append(neighbor)
                    # Update the shortest paths to include paths through the current node
                    node_to_shortest_paths[neighbor] = set.union(node_to_shortest_paths[neighbor], node_to_shortest_paths[current_node], {neighbor})

        # Calculate betweenness centrality for the given edge
        betweenness_centrality = 0
        # Iterate through all nodes in the graph
        for node in graph.nodes():
            # Exclude the nodes that are part of the edge
            if node != edge[0] and node != edge[1]:
                # Check if the edge is part of the shortest paths of the node
                for path in node_to_shortest_paths.get(node, []):
                    if edge[0] in path and edge[1] in path:
                        # Increment the betweenness centrality for each shortest path the edge is part of
                        betweenness_centrality += 1 / len(node_to_shortest_paths[node])

        # Return the calculated betweenness centrality for the edge
        return betweenness_centrality

    def girvan_newman_directed(graph):
        # Implement Girvan-Newman algorithm for community detection in a directed graph
        while nx.number_weakly_connected_components(graph) == 1:
            edge_to_remove_value = edge_to_remove_directed(graph)
            graph.remove_edge(*edge_to_remove_value)
        return list(nx.weakly_connected_components(graph))

    def girvan_newman_undirected(graph):
        # Implement Girvan-Newman algorithm for community detection in an undirected graph
        while nx.number_connected_components(graph) == 1:
            edge_to_remove_value = edge_to_remove_undirected(graph)
            graph.remove_edge(*edge_to_remove_value)
        return list(nx.connected_components(graph))

    # Select the top N papers based on degree centrality
    top_n_papers = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]
    
    # Create a subgraph with only the top N papers
    subgraph = graph.subgraph(top_n_papers).copy()
    initial_subgraph = subgraph.copy()

    # Check if both papers are present in the subgraph
    if paper_1 not in subgraph or paper_2 not in subgraph:
        print(f"One or both papers ({paper_1}, {paper_2}) not present in the graph.")

    # Perform Girvan-Newman community detection
    if graph.is_directed():
        communities = girvan_newman_directed(subgraph)
    else:
        communities = girvan_newman_undirected(subgraph)
        
    # Check if Paper_1 and Paper_2 belong to the same community
    same_community = any([paper_1 in community and paper_2 in community for community in communities])

    # Calculate the minimum number of edges to be removed
    num_edges_to_remove = len(initial_subgraph.edges()) - len(subgraph.to_undirected().edges())
    
    return subgraph, num_edges_to_remove, communities, same_community

#-------------------------------------------------------------------------------------------------------------------------

def visualize_graph_data(graph_data):
    '''
    Visualization of Graph Data

    Argument of the function:
    graph_data (dict) -> A dictionary containing all the requested data of the graph

    Output:
    Displays histograms of degree distribution, prints a table of key metrics and returns a pandas DataFrame of graph hubs
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    # Create histograms for degree distribution
    plt.figure(figsize=(15, 8))

    def plot_distribution(distribution_dict, title, subplot_index):
        # Convert the sub-dictionary into a DataFrame
        distribution_df = pd.DataFrame(list(distribution_dict.items()), columns=['Degree Range', 'Count'])
        # Create a subplot for the histogram
        plt.subplot(1, 2 if 'In-Degree Distribution' in graph_data and 'Out-Degree Distribution' in graph_data else 1, subplot_index)
        # Plot the histogram 
        sns.barplot(x='Degree Range', y='Count', data=distribution_df)
        # Set the title for the histogram
        plt.title(title)
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

    # Check if both in-degree and out-degree distributions exist for directed graphs
    if 'In-Degree Distribution' in graph_data and 'Out-Degree Distribution' in graph_data:
        # Plot two histograms for directed graphs (in-degree and out-degree)
        plot_distribution(graph_data['In-Degree Distribution'], 'In-Degree Distribution', 1)
        plot_distribution(graph_data['Out-Degree Distribution'], 'Out-Degree Distribution', 2)
    else:
        # Plot a single histogram for undirected graphs
        plot_distribution(graph_data['Degree Distribution'], 'Degree Distribution', 1)

    # Remove the top and right border
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Create a table with the key characteristics of the graph (excluding hubs, distributions, and Graph Name)
    metrics = {k: v for k, v in graph_data.items() if k not in ["Graph Hubs", "In-Degree Distribution", "Out-Degree Distribution", "Degree Distribution", "Graph Name"]}
    # Initialize a PrettyTable
    table = PrettyTable()
    # Set the title of the table to the graph's name
    table.title = f"{graph_data['Graph Name']}"
    # Set the column names of the table to the keys of the metrics
    table.field_names = metrics.keys()
    # Add the metrics values as a row in the table
    table.add_row(metrics.values())

    print(table)

    # Create a pandas DataFrame for the graph hubs
    hubs_df = pd.DataFrame(graph_data["Graph Hubs"], columns=["Graph Hubs"])

    return hubs_df


#-------------------------------------------------------------------------------------------------------------------------

def visualize_node_contribution(centrality_measures, node, graph_name):
    '''
    Visualize The Node Contribution

    Arguments of the function:
    centrality_measures (dict) -> A dictionary containing centrality measures of a node
    node (int) -> The node for which centrality measures are to be visualized
    graph_name (str) -> The name of the graph.

    Output:
    Prints a table displaying the centrality measures for the specified node.
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    # Initialize a PrettyTable for displaying centrality measures
    table = PrettyTable()
    # Set the title of the table using the node and the graph name
    table.title = f"Contribution of node {node} in {graph_name}"
    # Define the column names for the table
    table.field_names = ["Node", "Betweenness Centrality", "PageRank", "Closeness Centrality", "Degree Centrality"]
    # Format the centrality values to 4 decimal places and prepare them for adding to the table
    value = ["{:.4f}".format(centrality_measures[name]) for name in table.field_names[1:]]
    # Add a row to the table with the node and its centrality values
    table.add_row([node] + value)


    print(table)

#-------------------------------------------------------------------------------------------------------------------------

def visualize_shortest_path(graph, result, N):
    '''
    Visualization of the Shortest Path in a Graph

    Argumnets of the function:
    graph (nx.DiGraph) -> A NetworkX graph
    result (dict) -> A dictionary containing the 'Shortest Walk' and 'Papers Crossed' as keys
    N (int) -> The number of top nodes to consider based on degree centrality
    
    Output:
    A plot showing the subgraph with the highlighted shortest path and a legend indicating start and end nodes
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    # Extract the shortest path and the papers crossed from the result variable
    path, papers = result['Shortest Walk'], result['Papers Crossed']
    # Identify the start and the end node
    start_node, end_node = path[0], path[-1]  

    # Extract the top N nodes based on degree centrality
    top_n_nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:N]
    # Create the subgraph
    subgraph = graph.subgraph(top_n_nodes)  

    # Set the plot dimension
    fig, ax = plt.subplots(figsize=(20, 15))  # Creating a matplotlib figure and axis

    # Color the nodes 
    # start node in green, end node in red, others nodes in light blue
    node_colors = ['green' if node == start_node else 'red' if node == end_node else 'lightblue' for node in subgraph.nodes()]
    
    # Color the edges 
    # edges in the path in blue, others in light grey
    path_edges = set(zip(path, path[1:])) | set(zip(path[1:], path))
    edge_colors = ['blue' if (u, v) in path_edges else 'lightgrey' for u, v in subgraph.edges()]

    # Give a number to the edges along the shortest path
    edge_labels = {(u, v): str(i) for i, (u, v) in enumerate(zip(path, path[1:])) if u in subgraph and v in subgraph}

    # Plot the subgraph in a circular layout
    layout = nx.circular_layout(subgraph)  
    # Plot the nodes
    nx.draw(subgraph, layout, node_color=node_colors, with_labels=True, ax=ax, font_size=10, node_size=700) 
    # Plot the edges
    nx.draw_networkx_edges(subgraph, layout, edgelist=subgraph.edges(), edge_color=edge_colors, width=2, ax=ax)
    # Add a label to the edges
    nx.draw_networkx_edge_labels(subgraph, layout, edge_labels=edge_labels, font_color='black', ax=ax)  

    # Add a legend to the plot
    start_patch = mpatches.Patch(color='green', label='Start Node')  
    end_patch = mpatches.Patch(color='red', label='End Node')  
    # Show the legend
    plt.legend(handles=[start_patch, end_patch], loc='upper left')  
    # Sett the title of the plot
    ax.set_title("Shortest Path", fontsize=24)  
    # Show the plot
    plt.show()  

    path_df = pd.DataFrame({'List of Crossed Papers': papers })
    return path_df


#-------------------------------------------------------------------------------------------------------------------------


def visualize_disconnected_graph(original_graph, G_a, G_b, num_edges_to_disconnect):
    '''
    Arguments:
    original_graph (nx.Graph) -> A networkx graph
    G_a (set) -> A set of nodes representing first subgraphs after disconnection
    G_b (set) -> A set of nodes representing the other subgraph after disconnection
    num_edges_to_disconnect (int) -> The number of edges removed to disconnect the graph.
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count


    # Print the number of links to be disconnected
    print("Number of links to be disconnected:", num_edges_to_disconnect)
    # Generate positions for nodes for consistent layout between graphs
    pos = nx.spring_layout(original_graph)
    # Set up the plot for the original graph
    plt.figure(figsize=(15, 8))

    # Draw the original graph with specified settings
    nx.draw(original_graph, pos, with_labels=True, font_size=8, node_size=200, font_color="black")
    # Title for the original graph
    plt.title("Original Graph")
    # Display the plot for the original graph
    plt.show()

    # Create subgraphs from the original graph based on G_a and G_b
    G_a_subgraph = original_graph.subgraph(G_a)
    G_b_subgraph = original_graph.subgraph(G_b)

    # Set up the plot for the disconnected graph
    plt.figure(figsize=(15, 8))

    # Draw subgraph G_a with specific color and settings
    nx.draw(G_a_subgraph, pos, with_labels=True, font_size=8, node_size=200, font_color="black", node_color="skyblue")
    # Draw subgraph G_b with different color and settings
    nx.draw(G_b_subgraph, pos, with_labels=True, font_size=8, node_size=200, font_color="black", node_color="lightcoral")

    # Title for the disconnected graph
    plt.title("Disconnected Graph")
    # Add legend to distinguish the two subgraphs
    plt.legend(["Subgraph G_a", "Subgraph G_b"])
    # Display the plot for the disconnected graph
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------

def plot_communities(graph, communities, paper_1, paper_2):
    """
    Plots the graph highlighting the identified communities 

    Arguments of the function:
    graph (nx.Graph) -> A NetworkX graph object representing the original graph
    communities (list) -> A list of sets, where each set contains nodes belonging to a community
    paper_1 (str) -> Identifier for the first paper of interest
    paper_2 (str) -> Identifier for the second paper of interest

    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count
    

    # Compute node positions for consistent layout across all plots
    pos = nx.spring_layout(graph)
    # Plot the original graph
    plt.figure(figsize=(12, 8))
    # Plot nodes
    nx.draw_networkx_nodes(graph, pos, node_size=200)  
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)     
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")  
    
    plt.title("Original Graph")
    plt.show()

    # Plot the graph showing the communities
    plt.figure(figsize=(12, 8))
    for i, community in enumerate(communities, 1):
        # Draw nodes of each community in a different color
        nx.draw_networkx_nodes(graph, pos, nodelist=list(community), node_size=200, node_color=f"C{i}")

    # Draw edges    
    nx.draw_networkx_edges(graph, pos, alpha=0.5)  
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")  
    plt.title("Graph with Communities")
    plt.show()

    # Plot the final graph and highlight the communities of Paper_1 and Paper_2
    plt.figure(figsize=(12, 8))
    for i, community in enumerate(communities, 1):
        # Highlight communities containing Paper_1 or Paper_2, others in grey
        color = f"C{i}" if paper_1 in community or paper_2 in community else "lightgrey"
        # Increase node size for Paper_1 and Paper_2
        size = [300 if node in [paper_1, paper_2] else 100 for node in community]
        nx.draw_networkx_nodes(graph, pos, nodelist=list(community), node_size=size, node_color=color)

    # Draw the edges    
    nx.draw_networkx_edges(graph, pos, alpha=0.5)  
     # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black") 

    plt.title("Graph with Paper_1 and Paper_2 Communities Highlighted")
    plt.show()

def print_community_table(communities):
    '''
    Prints a table listing the papers in each community
    '''
    import networkx as nx
    import pandas as pd
    import numpy as np
    from collections import Counter
    from typing import List, Dict, Tuple
    import heapq
    from timeit import default_timer as timer
    
    import ijson
    import time
    import csv
    import matplotlib.pyplot as plt
    from itertools import zip_longest
    import seaborn as sns
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches
    from itertools import count

    # Print the header for community table
    print("Community\tPapers")
    print("-" * 30)
    # Iterate through each community and print its papers
    for i, community in enumerate(communities, 1):
        print(f"Community {i}:\t{', '.join(map(str, community))}")


#-------------------------------------------------------------------------------------------------------------------------

# Define a function to control the system
def control_system(Citation_G, Collaboration_G):
    """
    Control the system using a set of visualization functions and interactive widgets.
    
    The function defines and utilizes dropdowns for selecting visualization functions, graph types,
    and specific nodes or papers. It also includes functions for handling different visualization
    scenarios such as graph features, nodes contribution, shortest ordered route, disconnected graph,
    and communities. The user can interactively select options and observe the visualizations and outputs.
    """
    import time
    import ipywidgets as widgets
    from ipywidgets import interact, interact_manual, Text, Layout
    import IPython.display
    from IPython.display import display, clear_output
    from prettytable import PrettyTable
    import matplotlib.patches as mpatches

    ##### CREATING DROPDOWN WIDGESTS

    # Create a dropdown for selecting visualization functions
    dropdown_funcs = widgets.Dropdown(options=['Graph_features', 'Nodes_contribution',
                                              'Shortest_ordered_route', 'Disconnected_graph', 'Communities'],
                                     value=None, description='Visualization', layout=Layout(width='40%', background_color='lightgray'))

    # Get the list of nodes for citation and collaboration graphs and sort them so that the user can find an ID easily
    citation_nodes = sorted(list(Citation_G.nodes()))
    collaboration_nodes = sorted(list(Collaboration_G.nodes()))

    # Create dropdowns for selecting graph types (citation or collaboration)
    dropdown_graph = widgets.Dropdown(options=['Collaboration_G', 'Citation_G'], value=None, description='Graph Type', layout=Layout(width='30%', background_color='lightgray'))
    dropdown_Co_graph = widgets.Dropdown(options=['Collaboration_G'], value=None, description='Graph Type', layout=Layout(width='30%', background_color='lightgray'))

    # Create dropdowns for selecting paper IDs from citation graph
    dropdown_node_citation = widgets.Dropdown(options=citation_nodes, value=None, description='Papers ID', layout=Layout(width='20%', background_color='lightgray'))
    dropdown_node_paper_1 = widgets.Dropdown(options=citation_nodes, value=None, description='Papers ID', layout=Layout(width='20%', background_color='lightgray'))
    dropdown_node_paper_2 = widgets.Dropdown(options=citation_nodes, value=None, description='Papers ID', layout=Layout(width='20%', background_color='lightgray'))

    # Create dropdowns for selecting author IDs from collaboration graph
    dropdown_node_collaboration = widgets.Dropdown(options=collaboration_nodes, value=None, description='Authors ID', layout=Layout(width='20%', background_color='lightgray'))
    dropdown_node_collaboration_start = widgets.Dropdown(options=collaboration_nodes, value=None,
                                                          description='Authors ID', layout=Layout(width='20%', background_color='lightgray'))
    dropdown_node_collaboration_end = widgets.Dropdown(options=collaboration_nodes, value=None,
                                                        description='Authors ID', layout=Layout(width='20%', background_color='lightgray'))


    ##### CREATING 5 FUNCTIONS EACH OF WHICH HANDLES ONE VISUALIZATION TASK

    #### Visualization 1 - Visualize graph features

    # Define a function for handling graph features visualization
    def g_features(change):
        """
        Handle the visualization of graph features.
        
        This function is triggered when the user selects the 'Graph_features' option from the dropdown.
        It displays the degree distribution for the selected graph and outputs relevant graph features.
        """

        clear_output(wait=True)  # Clear existing output
        # Grab the new value of the changed dropdown using the .new attribute
        graph_name = change.new

        # Grab the user choice of the graph type and saved it as a variable to pass it to the visualization function
        if graph_name == 'Collaboration_G':
            graph_obj = Collaboration_G
        elif graph_name == 'Citation_G':
            graph_obj = Citation_G

        display(widgets.HBox([dropdown_funcs]))
        display(dropdown_graph)

        graph_data = extract_graph_data(graph_obj, graph_name)

        # Plot the visualization
        display(visualize_graph_data(graph_data))
        

    #### Visualization 2 - Visualize the node's contribution

    # Define a function for handling nodes contribution visualization
    def n_contribution(change):
        """
        Handle the visualization of nodes contribution.
        
        This function is triggered when the user selects the 'Nodes_contribution' option from the dropdown.
        It allows the user to select a specific node and displays its contribution metrics in a table.
        """

        clear_output(wait=True)  # Clear existing output
        graph_name = dropdown_graph.value  # Get the current value of the graph dropdown

        # Grab the user choice of the graph type and saved it as a variable to pass it to the visualization function
        if graph_name == 'Collaboration_G':
            graph_obj = Collaboration_G
            node_dropdown = dropdown_node_collaboration
        elif graph_name == 'Citation_G':
            graph_obj = Citation_G
            node_dropdown = dropdown_node_citation
        
        display(dropdown_funcs)
        display(dropdown_graph)
        print("Please select a node:")
        display(node_dropdown)

        def handle_nodes_func2(node_change):
            '''
            This function is triggered if the user changes the node option (paper/author) from the corresponding dropdown
            '''
            clear_output(wait=True)  # Clear existing output
            
            # Grab the new value of the changed dropdown using the .new attribute
            node = node_change.new if node_change.new else node_change.value
            
            # Run the visualization function
            output_data = node_contribution(graph_obj, node, graph_name)
            
            display(dropdown_funcs)
            display(dropdown_graph)
            display(node_dropdown)

            visualize_node_contribution(output_data, node, graph_name)

        # Observe the node_dropdown and trigger handle_nodes_func2 in case of changes
        node_dropdown.observe(handle_nodes_func2, names='value')
    
    # Define a list to store selected authors
    selected_nodes = [None, None]

    #### Visualization 3 - Visualize the shortest-ordered route

    # Define a function for handling shortest ordered route visualization
    def shortest_walk(change):
        """
        Handle the visualization of the shortest ordered route.
        
        This function is triggered when the user selects the 'Shortest_ordered_route' option from the dropdown.
        It prompts the user to define parameters, including the top N papers and the initial and end nodes.
        """

        clear_output(wait=True)  # Clear existing output
        display(dropdown_funcs)
        display(dropdown_Co_graph)       

        # Use a Text widget for user input
        print("Please define the top N number of papers to be considered: ")
        input_text = Text(description="N Value", value="")
        display(input_text)        

        def handle_input_text_func_3(change):
            '''
            This function is triggered if the user changes the N option (top N authors) from the corresponding dropdown
            '''
            time.sleep(3)
            clear_output(wait=True)  # Clear existing output
            # Grab the new value of the changed dropdown using the .new attribute
            if change.new:
                N = int(change.new)
            else:
                N = int(input_text.value)

            display(dropdown_funcs)
            display(dropdown_Co_graph) 
            print("Please select the initial node (author 1) and the end node (author 2), respectively:")
            display(widgets.HBox([dropdown_node_collaboration_start, dropdown_node_collaboration_end]))
            
            def handle_nodes_func3_a1(change_a1):
                '''
                This function is triggered if the user changes the node option (initial node a_1) from the corresponding dropdown
                '''
                clear_output(wait=True)  # Clear existing output
                # Grab the new value of the changed dropdown using the .new attribute
                selected_nodes[0] = change_a1.new if change_a1.new else dropdown_node_collaboration_start.value

            def handle_nodes_func3_a2(change_a2):
                '''
                This function is triggered if the user changes the node option (end node a_n) from the corresponding dropdown
                '''
                # Grab the new value of the changed dropdown using the .new attribute
                selected_nodes[1] = change_a2.new if change_a2.new else dropdown_node_collaboration_end.value

                
                display(dropdown_funcs)
                display(dropdown_Co_graph)
                display(input_text)
                print("Please select the initial node (author 1) and the end node (author 2), respectively:")
                display(widgets.HBox([dropdown_node_collaboration_start, dropdown_node_collaboration_end]))
                
                result = shortest_ordered_walk(Collaboration_G, [], selected_nodes[0], selected_nodes[1], N)
                display(result)
                visualize_shortest_path(Collaboration_G, result, N) 

            # Observe the dropdowns and trigger the corresponding functions in case of changes
            dropdown_node_collaboration_start.observe(handle_nodes_func3_a1, names='value')
            dropdown_node_collaboration_end.observe(handle_nodes_func3_a2, names='value')
        
        # Observe the dropdown and trigger the corresponding function in case of changes
        input_text.observe(handle_input_text_func_3, names='value')

    # Define a list to store selected authors
    selected_authors = [None, None]

    #### Visualization 4 - Visualize the disconnected graph

    # Define a function for handling disconnected graph visualization
    def disconnected_graph(change):
        """
        Handle the visualization of the disconnected graph.
        
        This function is triggered when the user selects the 'Disconnected_graph' option from the dropdown.
        It prompts the user to define parameters, including the top N papers and two authors for sub-graphs.
        """

        clear_output(wait=True)  # Clear existing output
        graph_name = dropdown_graph.value  # Get the current value of the graph dropdown
        
        # Grab the user choice of the graph type and saved it as a variable to pass it to the visualization function
        if graph_name == 'Collaboration_G':
            graph_obj = Collaboration_G
        elif graph_name == 'Citation_G':
            graph_obj = Citation_G        
        
        display(dropdown_funcs)
        display(dropdown_graph)

        # Use a Text widget for user input
        print("Please define the top N number of papers to be considered: ")
        input_text = Text(description="N Value", value="")
        display(input_text)

        def handle_input_text_func_4(change):
            '''
            This function is triggered if the user changes the N option (top N authors) from the corresponding dropdown
            '''
            time.sleep(3)
            clear_output(wait=True)  # Clear existing output
            # Grab the new value of the changed dropdown using the .new attribute
            if change.new:
                N = int(change.new)
            else:
                N = int(input_text.value)
            
            display(dropdown_funcs)
            display(dropdown_graph)
            print("Please select Author A (sub-graph G_a) and Author B (sub-graph G_b), respectively:")
            display(widgets.HBox([dropdown_node_collaboration_start, dropdown_node_collaboration_end]))

            def handle_nodes_func4_a1(change_a1):
                '''
                This function is triggered if the user changes the node option (authorB) from the corresponding dropdown
                '''
                clear_output(wait=True)  # Clear existing output
                # Grab the new value of the changed dropdown using the .new attribute
                selected_authors[0] = change_a1.new if change_a1.new else dropdown_node_collaboration_start.value

            def handle_nodes_func4_a2(change_a2):
                '''
                This function is triggered if the user changes the node option (authorB) from the corresponding dropdown
                '''
                # Grab the new value of the changed dropdown using the .new attribute
                selected_authors[1] = change_a2.new if change_a2.new else dropdown_node_collaboration_end.value

                # Run the visualization function
                subgraph, G_a, G_b, num_edges_in_min_edge_cut = disconnecting_graphs(graph_obj, selected_authors[0], selected_authors[1], N)
                visualize_disconnected_graph(subgraph, G_a, G_b, num_edges_in_min_edge_cut)
            
            # Observe the dropdowns and trigger the corresponding functions in case of changes
            dropdown_node_collaboration_start.observe(handle_nodes_func4_a1, names='value')
            dropdown_node_collaboration_end.observe(handle_nodes_func4_a2, names='value')
        
        # Observe the dropdown and trigger the corresponding function in case of changes
        input_text.observe(handle_input_text_func_4, names='value')

    # Define a list to store selected papers
    selected_papers = [None, None]

    #### Visualization 5 - Visualize the communities

    # Define a function for handling communities visualization
    def communities(change):
        """
        Handle the visualization of communities.
        
        This function is triggered when the user selects the 'Communities' option from the dropdown.
        It prompts the user to define parameters, including the top N papers and two papers for community analysis.
        """

        clear_output(wait=True)  # Clear existing output
        graph_name = dropdown_graph.value  # Get the current value of the graph dropdown
        
        # Grab the user choice of the graph type and saved it as a variable to pass it to the visualization function
        if graph_name == 'Collaboration_G':
            graph_obj = Collaboration_G
        elif graph_name == 'Citation_G':
            graph_obj = Citation_G 
        
        display(dropdown_funcs)
        display(dropdown_graph)

        # Use a Text widget for user input
        print("Please define the top N number of papers to be considered: ")
        input_text = Text(description="N Value", value="")
        display(input_text)

        def handle_input_text_func_5(change):
            '''
            This function is triggered if the user changes the N option (top N papers) from the corresponding dropdown
            '''
            time.sleep(3)
            clear_output(wait=True)  # Clear existing output
            # Grab the new value of the changed dropdown using the .new attribute
            if change.new:
                N = int(change.new)
            else:
                N = int(input_text.value)
            
            display(dropdown_funcs)
            display(dropdown_graph)
            print("Please select Paper 1 and Paper 2, respectively:")
            display(widgets.HBox([dropdown_node_paper_1, dropdown_node_paper_2]))

            def handle_nodes_func5_p1(change_p1):
                '''
                This function is triggered if the user changes the node option (paper_1) from the corresponding dropdown
                '''
                clear_output(wait=True)  # Clear existing output
                # Grab the new value of the changed dropdown using the .new attribute
                selected_papers[0] = change_p1.new if change_p1.new else dropdown_node_paper_1.value

            def handle_nodes_func5_p2(change_p2):
                '''
                This function is triggered if the user changes the node option (paper_2) from the corresponding dropdown
                '''
                clear_output(wait=True)  # Clear existing output
                # Grab the new value of the changed dropdown using the .new attribute
                selected_papers[1] = change_p2.new if change_p2.new else dropdown_node_paper_2.value

                # Run the visualization function
                subgraph, num_edges_to_remove, extracted_communities, same_community = extract_communities(graph_obj, N, selected_papers[0], selected_papers[1])
                display(dropdown_funcs)
                display(dropdown_graph)
                print("Please select Paper 1 and Paper 2, respectively:")
                display(widgets.HBox([dropdown_node_paper_1, dropdown_node_paper_2]))
                # Print a table showing  communities and the papers that belong to each community
                print_community_table(extracted_communities)
                # Plot the original graph, the graph showing the communities, and the final graph with Paper_1 and Paper_2 communities highlighted
                plot_communities(subgraph, extracted_communities, selected_papers[0], selected_papers[1])

            # Observe the dropdowns and trigger the corresponding functions in case of changes
            dropdown_node_paper_1.observe(handle_nodes_func5_p1, names='value')
            dropdown_node_paper_2.observe(handle_nodes_func5_p2, names='value')

        # Observe the dropdown and trigger the corresponding function in case of changes
        input_text.observe(handle_input_text_func_5, names='value')

    #### Visualization selection

    # Define a function for handling changes in the selected visualization function dropdown
    def func_change(change):
        """
        Handle changes in the selected visualization function.
        
        This function is triggered when the user selects a visualization function from the main dropdown.
        It dynamically updates the display based on the selected option.
        """

        clear_output(wait=True)  # Clear existing output
        # Grab the new value of the changed dropdown using the .new attribute
        visualization_choice = change.new

        if visualization_choice == 'Graph_features':
            # Display the dropdown
            display(dropdown_funcs)
            display(dropdown_graph)
            # Observe the dropdown and trigger the corresponding function in case of changes
            dropdown_graph.observe(g_features, names='value')

        elif visualization_choice == 'Nodes_contribution':
            # Display the dropdown
            display(dropdown_funcs)
            display(dropdown_graph)
            # Observe the dropdown and trigger the corresponding function in case of changes
            dropdown_graph.observe(n_contribution, names='value')

        elif visualization_choice == 'Shortest_ordered_route':
            # Display the dropdown
            display(dropdown_funcs)
            display(dropdown_Co_graph)
            # Observe the dropdown and trigger the corresponding function in case of changes
            dropdown_Co_graph.observe(shortest_walk, names='value')

        elif visualization_choice == 'Disconnected_graph':
            # Display the dropdown
            display(dropdown_funcs)
            display(dropdown_graph)
            # Observe the dropdown and trigger the corresponding function in case of changes
            dropdown_graph.observe(disconnected_graph, names='value')

        elif visualization_choice == 'Communities':
            # Display the dropdown
            display(dropdown_funcs)
            display(dropdown_graph)
            # Observe the dropdown and trigger the corresponding function in case of changes
            dropdown_graph.observe(communities, names='value')

    # Display the initial dropdown
    display(dropdown_funcs)
    # Observe the dropdown and trigger the corresponding function in case of changes
    dropdown_funcs.observe(func_change, names='value')

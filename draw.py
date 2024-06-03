import networkx as nx
from ipywidgets import IntProgress
from IPython.display import display
from matplotlib import pyplot as plt

def draw_graph(A):
    """
    Draw the graph A.
    """
    G = nx.Graph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i, j)
    nx.draw(G, with_labels=True)
    plt.show()

def draw_markov_chain(A):
    """
    Draw the Markov chain A.
    """
    G = nx.DiGraph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] > 0:
                G.add_edge(i, j, weight=A[i][j])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def draw_graph_and_path(A, path):
    """
    Draw the graph A with the path.
    """
    G = nx.Graph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i, j)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)


    # Draw in green the first city
    first_city = path.index(True)
    nx.draw_networkx_nodes(G, pos, nodelist=[first_city], node_color='g')

    for i in range(len(path) - 1):
        if path[i]:
            nx.draw_networkx_edges(G, pos, edgelist=[(i, i + 1)], edge_color='r')
    plt.show()

def plot_graph(A, weights=None):
    """
    Plot the graph A.
    """
    plt.figure()
    plt.imshow(A, cmap='gray')
    plt.axis('off')
    plt.show()
    if weights is not None:
        print(weights)

def init_loading_bar(size):
    """
    Initialize the loading bar.
    """
    f = IntProgress(min=0, max=size)
    display(f)
    return f

def update_loading_bar(f, value):
    """
    Update the loading bar.
    """
    f.value = value

def draw_graph_and_solution(A, solution):
    """
    Draw the graph A with the solution.
    """
    G = nx.Graph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i, j)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)

    for i, j in solution:
        nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], edge_color='r')
    plt.show()


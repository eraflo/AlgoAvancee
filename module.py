import numpy as np
import random as rand
from matplotlib import pyplot as plt
import networkx as nx

def generate_random_symetrical_graph(n, min, max):
    """
    Generate a random graph with n nodes (tuple of tuples)
    """
    A = np.random.randint(min, max, (n, n))
    A = np.triu(A, 1)
    A = A + A.T
    A_tuple = ()
    for i in range(n):
        A_tuple += (tuple(A[i]),)
    return A_tuple

def generate_random_symetrical_graph_for_periods(n, min, max, periods):
    """
    Generate a random graph with n nodes (tuple of tuples) for each period.
    """
    A = []
    for i in range(len(periods)):
        A.append(generate_random_symetrical_graph(n, min, max))
    return A

def generate_random_weights(A, max_weight):
    """
    Generate a random list of weights for a graph.
    """
    weights = []
    for t in range(len(A)):
        weights.append([])
        for i in range(len(A[t])):
            weights[t].append([])
            for j in range(len(A[t][i])):
                if A[t][i][j] == 1:
                    weights[t][i].append(rand.randint(2, max_weight))
                else:
                    weights[t][i].append(float('inf'))
    return weights


def generate_random_collect_points(cities):
    """
    Generate a random list of cities where the vehicle will collect the packages.
    """
    collect_points = [False] * cities
    for i in range(cities):
        collect_points[i] = rand.choice([True, False])
    return collect_points

def generate_random_delivery_requests(cities, collect_points):
    """
    Generate a random list of cities where the vehicle will deliver the packages.
    """
    drop_points = []
    for i in range(len(collect_points)):
        number_of_drop_points = rand.randint(1, cities - 1)

        for j in range(number_of_drop_points):
            drop_point = rand.randint(0, cities - 1)
            drop_points.append((i, drop_point))

    return drop_points

def generate_random_periods(range_period):
    """
    Generate a random list of periods.
    """
    periods = []
    i = 0
    while i < range_period:
        start = i
        next_start = start + 1
        end = rand.randint(next_start, range_period)
        periods.append((start, end))

        i = end + 1
    return periods

def voisinage(A, period, city):
    """
    Return the neighbors of a city in a period.
    """
    voisinage = []
    for i in range(len(A[period])):
        if A[period][city][i] == 1:
            voisinage.append(i)
    return voisinage

def generate_random_solution(A, C, P, R, V, s0, periods):
    """
    Generate a random solution.
    """
    solution = [[s0] for _ in range(V)]
    t = periods[0][0]

    for v in range(V):
        for period in periods:
            while t < period[1]:
                city = solution[v][-1]
                voisinage = voisinage(A, periods.index(period), city)
                next_city = rand.choice(voisinage)
                solution[v].append(next_city)
                t += C[periods.index(period)][city][next_city]

        
        



    
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

    
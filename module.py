import numpy as np
import random as rand
from matplotlib import pyplot as plt
import math as m
from collections import deque
import networkx as nx

def remove_from_list_tuple(l, i):
    """
    Remove the tuple with i as first element from the list of tuples l.
    """
    for item in l:
        if item[0] == i:
            l.remove(item)
            break  
    return l

def generate_random_symetrical_boolean_graph(n):
    """
    Generate a random symetrical graph with n nodes. There is at least one edge between two nodes.
    """
    A = np.random.randint(0, 2, (n, n))
    A = np.triu(A, 1)
    A = A + A.T

    if(not connexity(A)):
        A = generate_random_symetrical_boolean_graph(n)

    A_tuple = ()
    for i in range(n):
        A_tuple += (tuple(A[i]),)
    return A_tuple


# TODO : erase
def generate_random_symetrical_weighted_graph(n, min, max):
    """
    Generate a random symetrical weighted graph with n nodes.
    """
    A = np.random.uniform(min, max, (n, n))
    A = np.triu(A, 1)
    A = A + A.T
    A_tuple = ()
    for i in range(n):
        A_tuple += (tuple(A[i]),)
    return A_tuple



def generate_empty_graph(n):
    """
    Generate an empty graph with n nodes.
    """
    A = np.zeros((n, n))
    A_tuple = ()
    for i in range(n):
        A_tuple += (tuple(A[i]),)
    return A_tuple


def connexity(A):
    n = len(A)
    visited = np.zeros(n, dtype=bool)
    stack = [0] 
    visited[0] = True

    while stack:
        node = stack.pop()
        for i in range(n):
            if A[node][i] and not visited[i]:
                stack.append(i)
                visited[i] = True

    return np.all(visited)



def generate_random_collect_points(cities):
    """
    Generate a random list of cities where the vehicle will collect the packages.
    """
    collect_points = np.random.randint(0, 2, cities)
    return collect_points



def generate_random_delivery_requests(cities, collect_points):
    """
    Generate a random list of cities where the vehicle will deliver the packages.
    """
    drop_points = []
    for i in range(len(collect_points)):
        if collect_points[i]:
            number_of_drop_points = rand.randint(0, cities - 1)

            for j in range(number_of_drop_points):
                drop_point = rand.randint(0, cities - 1)
                if((i, drop_point) not in drop_points):
                    drop_points.append((i, drop_point))
    return drop_points




def generate_random_delivery_requests_v2(cities, collect_points):

    # Création d'un masque pour les points collectés
    mask = collect_points.astype(bool)

    # Nombre de points de chute pour chaque point collecté
    number_of_drop_points = np.random.randint(0, cities - 1, size=np.sum(mask))

    # Générer les coordonnées des points de chute
    drop_points_indices = np.random.randint(0, cities - 1, size=(np.sum(number_of_drop_points), 2))

    # Filtrer les doublons
    drop_points = drop_points_indices[~np.isin(drop_points_indices[:, 0], np.where(mask)[0])]

    # Créer un array de tuples
    drop_points = np.array([(point, drop) for point, drop in drop_points])
    return drop_points




def C(A, phi, Temp, i, j, t, amplitude, offset, frequency):
    """
    Cost function between the cities i and j at time t.

    @param A: Graph of the cities.
    @param phi: Initial phases
    @param Temp: Initial values to simulate randomness.
    @param i: City of departure.
    @param j: City of arrival.
    @param t: Time.
    @param amplitude: Amplitude of the cost function.
    @param offset: Offset of the cost function.
    @param frequency: Frequency of the cost function.
    """
    return max(0, round((amplitude * m.sin(frequency * t + phi[i][j]) + offset) * Temp[i][j]), 4) if A[i][j] == 1 else float('inf')



def neighbors(A, i):
    """
    Return the neighbors of the city i in the graph A.
    """
    return [j for j in range(len(A[i])) if A[i][j] == 1]



def pass_through(A, i, j):
    """"
    Mark the edge between the cities i and j as passed through.
    """
    A[i][j] = 1
    A[j][i] = 1



def collect(P, r):
    """
    Mark the object m from the request r as collected.
    """
    P[r[0]][r[1]] = 1



def deliver(D, r):
    """
    Mark the object m from the request r as delivered.
    """
    D[r[0]][r[1]] = 1



def get_city_passed_through(X, t):
    """
    Get the city passed through at time t.
    """
    for i in range(len(X[t])):
        for j in range(len(X[t][i])):
            if X[t][i][j]:
                return i, j



def AStar(A, start, end, t, phi, Temp, amplitude, offset, frequency):
    """
    A* algorithm to find the shortest path between the cities start and end at time t.
    """
    n = len(A)
    open_list = deque([(start, 0)])
    closed_list = []
    g = np.full(n, float('inf'))
    g[start] = 0
    f = np.full(n, float('inf'))
    f[start] = 0
    parent = np.full(n, None)

    while open_list:
        i = min(open_list, key=lambda x: x[1])[0]

        open_list = remove_from_list_tuple(open_list, i)
        closed_list.append(i)

        if i == end:
            path = [end]
            while parent[path[0]] is not None:
                path.insert(0, parent[path[0]])
            return path

        for j in neighbors(A, i):
            if j not in closed_list:
                if g[j] > g[i] + C(A, phi, Temp, i, j, t, amplitude, offset, frequency):
                    g[j] = g[i] + C(A, phi, Temp, i, j, t, amplitude, offset, frequency)
                    f[j] = g[j] + C(A, phi, Temp, j, end, t, amplitude, offset, frequency)
                    parent[j] = i
                    t += 1
                    if (j, g[j] + f[j]) not in open_list:
                        open_list.append((j, g[j] + f[j]))

    return None

    
    



def construct_path(solution):
    """
    Construct the path from the solution (list of tuples (i, j) where i is the city of departure and j the city of arrival).
    """
    path = [solution[0][0]]
    for i, j in solution:
        path.append(j)
    return path

def check_delivery_done(D, R):
    """
    Check if all the deliveries have been done.
    """
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] == 1 and (i, j) in R:
                return False
    return True

def generate_random_solution(A, R):
    """
    Generate a random solution for the problem.
    """
    X = []
    p = []
    d = []

    s0 = rand.randint(0, len(A) - 1)
    X.append(generate_empty_graph(len(A)))
    
    

    
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

    
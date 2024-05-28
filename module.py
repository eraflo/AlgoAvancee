import numpy as np
import random as rand
from matplotlib import pyplot as plt
import networkx as nx

def generate_random_symetrical_graph(n):
    """
    Generate a random graph with n nodes (tuple of tuples)
    """
    A = np.random.randint(0, 2, (n, n))
    A = np.triu(A, 1)
    A = A + A.T
    A_tuple = ()
    for i in range(n):
        A_tuple += (tuple(A[i]),)
    return A_tuple


def generate_random_weights(nb_weights, max_weight):
    """
    Generate a tuple of random weights between nodes.
    """
    weights = np.random.randint(1, max_weight, (nb_weights, nb_weights))
    weights = np.triu(weights, 1)
    weights = weights + weights.T
    weights_tuple = ()
    for i in range(nb_weights):
        weights_tuple += (tuple(weights[i]),)
    return weights_tuple

def generate_weights_over_time(nb_weights, max_weight, nb_time_steps, step=1):
    """
    Generate a tuple of random weights over time.
    """
    weights = ()
    for i in range(0, nb_time_steps, step):
        weights += (generate_random_weights(nb_weights, max_weight),)
    return weights

def generate_random_depot(nb_cities, nb_objects):
    """
    Generate tuples of random numbers of depots in random cities for nb_objects.
    """
    depots = ()
    for i in range(nb_objects):
        nb_depots_for_object = rand.randint(1, nb_cities)
        depots_for_object = ()
        for j in range(nb_depots_for_object):
            depot = rand.randint(0, nb_cities - 1)
            if depot not in depots_for_object:
                depots_for_object += (depot,)
        
        # Order the cities
        depots_for_object = tuple(sorted(depots_for_object))
        depots += (depots_for_object,)
    return depots

def generate_random_delivery(nb_cities, nb_objects):
    """
    Generate dictionaries with cities to deliver for each object.
    """
    deliveries = {}
    for i in range(nb_objects):
        nb_deliveries_for_object = rand.randint(1, nb_cities)
        cities_to_deliver = []
        for j in range(nb_deliveries_for_object):
            city = rand.randint(0, nb_cities - 1)
            if city not in cities_to_deliver:
                cities_to_deliver.append(city)
        deliveries[i] = cities_to_deliver
        
    return deliveries

def voisinage(cities_paths, city):
    """
    Return the neighbors of a city.
    """
    cities = [c for c in range(len(cities_paths[city])) if cities_paths[city][c] == 1]
    return cities

def has_object(path, depots, k):
    """"
    Return True if has passed through a depot for object k.
    """ 
    return any([path[city] for city in depots[k] if city < len(path)])

def has_delivered(path, depots, deliveries, k):
    """
    Return True if has delivered object k.
    """
    return any([has_object(path, depots, k) and path[city] for city in deliveries[k] if city < len(path)])

def get_object(objects_in_truck, k, step):
    """"
    Get the object from depot k.
    """
    if(not objects_in_truck[step][k]):
        objects_in_truck[step][k] = True
        return True
    return False

def deliver_object(deliveries_done, city, k):
    """
    Deliver the object k in city.
    """
    if(not deliveries_done[city][k]):
        deliveries_done[city][k] = True
        return True
    return False

def is_depots(city, depots):
    """
    Return True if city is a depot.
    """
    return any([city in depots[k] for k in range(len(depots))])



def random_solution(cities_paths, depots, deliveries, period):
    """
    Generate a random solution for the problem. 0 : I don't pass through the city, 1 : I pass through the city.
    """
    solution = (([False for i in range(len(cities_paths))] for j in range(len(cities_paths))) for k in range(period))

    hasObjects = [False for obj in range(len(depots))]
    hasDelivered = ([False for city in range(len(deliveries))] for obj in range(len(depots)))

    start_city = rand.randint(0, len(cities_paths) - 1)

    if(is_depots(start_city, depots)):
        get_object(hasObjects, depots.index([start_city]), start_city)
    
    step = 0
    while not all(hasDelivered):
        neighbors = voisinage(cities_paths, start_city)
        end_city = rand.choice(neighbors)

        if not hasDelivered[end_city] and hasObjects[end_city] and cities_paths[end_city]:
            solution[step][start_city][end_city] = True
            deliver_object(hasDelivered, end_city, hasObjects[end_city].index(True))
        
        if(is_depots(end_city, depots) and not hasObjects[end_city]):
            get_object(hasObjects, depots.index([end_city]), end_city)

        start_city = end_city
        step += 1
        

    return solution

def get_path(solution):
    """
    Get the path from the solution.
    """
    path = []
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            for k in range(len(solution[i][j])):
                if solution[i][j][k]:
                    path.append(j)
    return path

    
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

def draw_graph_and_weights(A, weights):
    """
    Draw the graph A with weights.
    """
    G = nx.Graph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i, j, weight=weights[i][j])
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

def draw_whole_graph(A, weights, depots, deliveries, path):
    """
    Draw the graph A with weights, depots, deliveries and path.
    """
    G = nx.Graph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G.add_edge(i, j, weight=weights[i][j])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Draw depots
    for i in range(len(depots)):
        nx.draw_networkx_nodes(G, pos, nodelist=depots[i], node_color='b')

    # Draw deliveries
    for i in range(len(deliveries)):
        nx.draw_networkx_nodes(G, pos, nodelist=deliveries[i], node_color='g')

    # Draw path
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

    
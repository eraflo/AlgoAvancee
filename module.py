import numpy as np
import random as rand



import utilities as u
import actions as a


def generate_random_symetrical_boolean_graph(n, isTuple = True):
    """
    Generate a random symetrical graph with n nodes. There is at least one edge between two nodes.
    """
    A = np.random.randint(0, 2, (n, n))
    A = np.triu(A, 1)
    A += A.T  

    if(not u.connexity(A)):
        A = generate_random_symetrical_boolean_graph(n)

    if isTuple:
        A = tuple(map(tuple, A))
    return A

def tuple_to_graph(A):
    """
    Convert a tuple to a graph.
    """
    return np.array(A)

def generate_random_symetrical_weighted_graph(n, min, max):
    """
    Generate a random symetrical weighted graph with n nodes.
    """
    A = np.random.uniform(min, max, (n, n))
    A = np.triu(A, 1)
    A += A.T  
    A_tuple = tuple(map(tuple, A))
    return A_tuple


def generate_empty_graph(n, m, isTuple = True):
    """
    Generate an empty graph with n nodes.
    """
    A = np.zeros((n, m))
    if isTuple:
        A = tuple(map(tuple, A))
    return A

def generate_empty_line(n):
    """
    Generate an empty line with n elements.
    """
    A = np.zeros((n, ))
    return A


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
    drop_points = set()
    for i, collect in enumerate(collect_points):
        if collect:
            number_of_drop_points = rand.randint(0, cities - 1)
            drop_points.update((i, rand.randint(0, cities - 1)) for _ in range(number_of_drop_points))
    return sorted(drop_points)

        

def verify_solution(R, solution):
    """
    Verify if the solution is valid.
    """
    deliveries_done = set()
    pickups_done = set()

    for cur, next_city in solution:
        for i, j in R:
            if cur == i and i not in pickups_done:
                pickups_done.add(i)
            if cur == j and i in pickups_done:
                deliveries_done.add((i, j))

    return len(deliveries_done) == len(R) and solution[0][0] == solution[-1][1]
    
    
    
    


    
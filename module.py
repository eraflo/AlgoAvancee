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



def construct_path(solution):
    """
    Construct the path from the solution.
    """
    path = []
    for i, j in solution:
        path.append(i)
    return path


def generate_random_solution(A, R):
    """
    Generate a random solution for the problem.
    """
    n = len(A)
    X = []
    p = [0] * n

    s0 = rand.randint(0, n - 1)


    neighbors = {}

    def neighbors_cache(A, i):
        if(i not in neighbors):
            neighbors[i] = a.neighbors(A, i)
        return neighbors[i]

    cur = s0
    next_city = None

    deliveries_done = set() 

    while len(deliveries_done) < len(R) and next_city != s0:
        neighbors_cur = neighbors_cache(A, cur)

        next_city = rand.choice(neighbors_cur)


        for i, j in R:
            has_pickup = p[i] == 1
            if i == cur and not has_pickup:
                p[cur] = 1
            if j == cur and has_pickup:
                deliveries_done.add(j)
                #R.remove((i, j))
            
            if j in neighbors_cur and j not in deliveries_done and has_pickup:
                next_city = j
                
            if i in neighbors_cur and not has_pickup:
                next_city = i

        
        X.append((cur, next_city))

        cur = next_city
    
    return X, p, deliveries_done, s0
        



    
    


    
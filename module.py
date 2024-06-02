import numpy as np
import random as rand
import math as m
from collections import deque
import heapq

import utilities as u


def generate_random_symetrical_boolean_graph(n):
    """
    Generate a random symetrical graph with n nodes. There is at least one edge between two nodes.
    """
    A = np.random.randint(0, 2, (n, n))
    A = np.triu(A, 1)
    A += A.T  

    if(not u.connexity(A)):
        A = generate_random_symetrical_boolean_graph(n)

    A_tuple = tuple(map(tuple, A))
    return A_tuple

def generate_random_symetrical_weighted_graph(n, min, max):
    """
    Generate a random symetrical weighted graph with n nodes.
    """
    A = np.random.uniform(min, max, (n, n))
    A = np.triu(A, 1)
    A += A.T  
    A_tuple = tuple(map(tuple, A))
    return A_tuple



def generate_empty_graph(n):
    """
    Generate an empty graph with n nodes.
    """
    A = np.zeros((n, n))
    A_tuple = tuple(map(tuple, A))
    return A_tuple



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
    Construct the path from the solution (list of tuples (i, j) where i is the city of departure and j the city of arrival).
    """
    path = [solution[0][0]]
    for i, j in solution:
        path.append(j)
    return path

def generate_random_solution(A, R):
    """
    Generate a random solution for the problem.
    """
    X = []
    p = []
    d = []

    s0 = rand.randint(0, len(A) - 1)
    X.append(generate_empty_graph(len(A)))
    
    


    
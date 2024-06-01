import numpy as np
import random as rand
import math as m
from collections import deque
import heapq
import threading as th
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import actions as a


def AStar(A, start, end, t, phi, Temp, amplitude, offset, frequency):
    """
    A* algorithm to find the shortest path between two nodes in a graph
    """
    n = len(A)
    open_list = []
    heapq.heappush(open_list, (0, start))
    closed_list = set()
    g = np.full(n, float('inf'))
    g[start] = 0
    f = np.full(n, float('inf'))
    f[start] = 0

    parent = np.full(n, None)

    cost_cache = {}

    # Cache the cost function
    def cost(i, j, t):
        if (i, j, t) not in cost_cache:
            cost_cache[(i, j, t)] = a.C(A, phi, Temp, i, j, t, amplitude, offset, frequency)
        return cost_cache[(i, j, t)]
    
    while open_list:
        _, i = heapq.heappop(open_list)

        if i in closed_list:
            continue
        closed_list.add(i)

        if i == end:
            path = [end]
            while parent[path[0]] is not None:
                path.insert(0, parent[path[0]])
            return path
        
        for j in a.neighbors(A, i):
            if j not in closed_list:
                try_g = g[i] + cost(i, j, t)
                if try_g < g[j]:
                    g[j] = try_g
                    f[j] = g[j] + cost(j, end, t)
                    parent[j] = i
                    t += 1
                    heapq.heappush(open_list, (f[j], j))
    return None


def construct_path(solutions, solutions_lock, max_iter_per_trial, A, R, pheromones, pheromone_lock, phi, Temp, amplitude, offset, frequency, alpha, beta, gamma, neighbors_cache, cost):
    """
    Generate a random solution for the problem.
    """
    n = len(A)
    X = []
    p = np.zeros(n)
    len_R = len(R)

    s0 = rand.randint(0, n - 1)

    cur = s0
    next_city = None

    deliveries_done = set() 
    
    t = 0
    trial = 0
    while (len(deliveries_done) < len_R or next_city != s0) and trial < max_iter_per_trial:
        
        neighbors_cur = neighbors_cache(A, cur)
        
        with pheromone_lock:
            pheromone_values = np.array([pheromones[cur][neighbor] for neighbor in neighbors_cur])
        cost_values = np.array([cost(cur, neighbor, t) for neighbor in neighbors_cur])

        probabilities = (pheromone_values ** alpha) * ((1 / cost_values) ** beta)
        total_probabilities = probabilities.sum()

        if total_probabilities > 0:
            probabilities = (gamma + probabilities) / total_probabilities
            next_city = rand.choices(neighbors_cur, probabilities)[0]
        else:
            next_city = rand.choice(neighbors_cur)

        for i, j in R:
            has_pickup = p[i] == 1
            if i == cur and not has_pickup:
                p[cur] = 1
            if j == cur and has_pickup:
                deliveries_done.add((i, j))
        
        X.append((cur, next_city))

        cur = next_city
        trial += 1
    
    with solutions_lock:
        solutions.append((X, s0))
        
def ants_colony(A, R, fourmis, phi, Temp, amplitude, offset, frequency, alpha, beta, gamma, rho, Q, iterations, max_iter_per_trial):
    """
    Ants colony algorithm to solve the problem.
    """
    n = len(A)
    
    best_solution = None
    
    
    solutions_lock = th.Lock()
    pheromones_lock = th.Lock()
    neighbors_lock = th.Lock()
    cost_lock = th.Lock()
    
    
    pheromones = np.ones((n, n))


    neighbors = {}
    cost_cache = {}

    def neighbors_cache(A, i):
        if(i not in neighbors):
            neighbors[i] = a.neighbors(A, i)
        return neighbors[i]
    
    def cost(i, j, t):
        if (i, j, t) not in cost_cache:
            cost_cache[(i, j, t)] = a.C(A, phi, Temp, i, j, t, amplitude, offset, frequency)
        return cost_cache[(i, j, t)]

    for _ in range(iterations):
        solutions_temp = []
        with ThreadPoolExecutor(max_workers=fourmis) as executor:
            futures = [
                executor.submit(
                    construct_path,
                    solutions_temp,
                    solutions_lock,
                    max_iter_per_trial,
                    A,
                    R,
                    pheromones,
                    pheromones_lock,
                    phi,
                    Temp,
                    amplitude,
                    offset,
                    frequency,
                    alpha,
                    beta,
                    gamma,
                    neighbors_cache,
                    cost
                ) for _ in range(fourmis)
            ]
        
        for future in futures:
            future.result()
        
        for X, s0 in solutions_temp:
            cost_solution = 0
            for i, j in X:
                cost_solution += cost(i, j, cost_solution)
            pheromone_deposit = Q / cost_solution
            with pheromones_lock:
                for i, j in X:
                    pheromones[i][j] += pheromone_deposit
                    pheromones[j][i] += pheromone_deposit
                pheromones *= (1 - rho)
            
            if best_solution is None or cost_solution < best_solution[1]:
                best_solution = (X, cost_solution)
    
    return best_solution
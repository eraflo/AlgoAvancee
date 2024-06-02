import numpy as np
import random as rand
import math as m
from collections import deque
import heapq
import threading as th
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pulp

import actions as a
from pulp import LpSolverDefault


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

    # Precompute neighbors for each city
    neighbors_dict = {i: neighbors_cache(A, i) for i in range(n)}

    while (len(deliveries_done) < len_R or next_city != s0) and trial < max_iter_per_trial:
        
        neighbors_cur = neighbors_dict[cur]
        

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
            if i == cur and p[cur] == 0:
                p[cur] = 1
            if j == cur and p[i] == 1:
                deliveries_done.add((i, j))
        
        X.append((cur, next_city))

        cur = next_city
        t += cost_values[neighbors_cur.index(next_city)]
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

def linear_programming(A, R, phi, Temp, amplitude, offset, frequency):
    N = range(len(A))  # Ensemble des villes (indices des villes)
    s0 = 0  # Nœud de départ du véhicule (ville1)

    # Définition du problème
    prob = pulp.LpProblem("VehicleRoutingProblem", pulp.LpMinimize)

    # Variables de décision
    T_max = 100
    x = pulp.LpVariable.dicts("x", [(i, j, t) for i in N for j in N for t in range(T_max)], 0, 1, pulp.LpBinary)
    p = pulp.LpVariable.dicts("p", [(i, t) for i in N for t in range(T_max)], 0, 1, pulp.LpBinary)
    d = pulp.LpVariable.dicts("d", [(i, t) for i in N for t in range(T_max)], 0, 1, pulp.LpBinary)
    cost = pulp.LpVariable.dicts("cost", [(i, j, t) for i in N for j in N for t in range(T_max)], 0, None, pulp.LpContinuous)
    y_t = pulp.LpVariable.dicts("y_t", range(T_max), 0, 1, pulp.LpBinary)
    T = pulp.LpVariable("T", 0, T_max, pulp.LpInteger)

    # Fonction objectif : Minimiser la distance totale parcourue
    prob += pulp.lpSum(cost[(i, j, t)] for i in N for j in N for t in range(T_max))

    # Contraintes

    # Mise à jour des coûts
    for t in range(T_max):
        for i in N:
            for j in N:
                calculated_cost = a.C(A, phi, Temp, i, j, t, amplitude, offset, frequency)
                if A[i][j] == 1:
                    prob += cost[(i, j, t)] >= calculated_cost * x[(i, j, t)]

    # Collecte avant livraison
    for m in range(len(R)):
        pickup, delivery = R[m]
        for t in range(T_max):
            prob += pulp.lpSum(p[(pickup, t_)] for t_ in range(t+1)) >= pulp.lpSum(d[(delivery, t_)] for t_ in range(t+1))

    # Lien collecte-livraison
    for m in range(len(R)):
        pickup, delivery = R[m]
        for t in range(T_max):
            prob += d[(delivery, t)] <= pulp.lpSum(p[(pickup, t_)] for t_ in range(t+1))

    # Conservation de flux
    for t in range(T_max-1):
        for i in N:
            prob += pulp.lpSum(x[(i, j, t)] for j in N if j != i) == pulp.lpSum(x[(j, i, t+1)] for j in N if j != i)

    # Départ initial
    prob += pulp.lpSum(x[(s0, j, 0)] for j in N if j != s0) == 1

    # Retour final (assuré par y_t)
    for t in range(T_max):
        prob += pulp.lpSum(x[(i, s0, t)] for i in N if i != s0) == y_t[t]

    prob += pulp.lpSum(y_t[t] for t in range(T_max)) == 1

    # Fin de livraison
    prob += pulp.lpSum(d[(i, t)] for i in N for t in range(T_max)) == len(R)

    # Connectivité des arcs
    for i in N:
        for j in N:
            for t in range(T_max):
                prob += x[(i, j, t)] <= A[i][j]

    # Déterminer la valeur de T dynamiquement
    for t in range(T_max):
        prob += pulp.lpSum(x[(i, j, t)] for i in N for j in N) <= 1
        prob += pulp.lpSum(x[(i, j, t)] for i in N for j in N) >= (T - t) / T_max

    # Résoudre le problème
    prob.solve()

    # Affichage des résultats
    print(f"Status: {pulp.LpStatus[prob.status]}")

    # Affichage des valeurs des variables
    solution = []
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)
            if v.name[0] == 'x':
                i, j, t = v.name[2:-1].split(',')
                solution.append((int(i), int(j), int(t)))

    # Affichage de la valeur de la fonction objectif
    print("Total Distance = ", pulp.value(prob.objective))
    print("Total Time T = ", pulp.value(T))
    
    
    
            


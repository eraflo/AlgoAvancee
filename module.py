import numpy as np
import random as rand
import threading as th


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




def construct_path(solutions, solutions_lock, max_iter_per_trial, A, R, pheromones, pheromone_lock, phi, Temp, amplitude, offset, frequency, alpha, beta, gamma, neighbors_cache, cost, neighbors_lock, cost_lock):
    """
    Generate a random solution for the problem.
    """
    n = len(A)
    X = []
    p = [0] * n

    s0 = rand.randint(0, n - 1)


    cur = s0
    next_city = None

    deliveries_done = set() 
    
    t = 0
    trial = 0
    while (len(deliveries_done) < len(R) or next_city != s0) and trial < max_iter_per_trial:
        
        with neighbors_lock:
            neighbors_cur = neighbors_cache(A, cur)
        
        probabilities = []
        with pheromone_lock, cost_lock:
            for neighbor in neighbors_cur:
                pheromone_value = pheromones[cur][neighbor]
                cost_value = cost(cur, neighbor, t)
                probability = pow(pheromone_value, alpha) * pow(1 / cost_value, beta)
                probabilities.append(probability)
            
        total = sum(probabilities)
        probabilities = [(gamma + prob) / total for prob in probabilities]
        
        next_city = rand.choices(neighbors_cur, probabilities)[0]

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
    
    solutions = []
    cost_solutions = []
    
    
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
        threads = []
        for fourmi in range(fourmis):
            solution = th.Thread(target=construct_path, args=(
                solutions, 
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
                cost,
                neighbors_lock,
                cost_lock)
            )
            threads.append(solution)
            solution.start()
        
        for thread in threads:
            thread.join()
        
        for X, s0 in solutions:
            cost_solution = 0
            for i, j in X:
                cost_solution += cost(i, j, cost_solution)
            pheromone_deposit = Q / cost_solution
            with pheromones_lock:
                for i, j in X:
                    pheromones[i][j] += pheromone_deposit
                    pheromones[j][i] += pheromone_deposit
                pheromones *= (1 - rho)
            
            cost_solutions.append(cost_solution)
    
    mini = min(solutions, key=lambda x: cost_solutions[solutions.index(x)])
    solution = mini[0]
    mini_cost = min(cost_solutions)
    return solution, mini_cost
        

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
    
    
    
    


    
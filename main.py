import module as m
import actions as a
import algorithms as al
import utilities as u
import draw as d
import stats as s

import time
import cProfile
import re
import pstats
import io


def main():
    V = 1 # Number of vehicles
    N = 1000 # Number of cities

    # Random Paramters
    phi = m.generate_random_symetrical_weighted_graph(N, 0.5, 5) # Initial phase 
    Temp = m.generate_random_symetrical_weighted_graph(N, 0, 20) # Initial random values
    amplitude = 0.2
    offset = 0.3
    frequency = 0.5
    
    # ants parameters
    alpha = 1
    beta = 2
    gamma = 0.5
    rho = 0.1
    Q = 100
    
    iterations = 5
    max_iter_per_trial = 100000
    fourmis = 2


    start = time.time()
    A = m.generate_random_symetrical_boolean_graph(N)

    P = m.generate_random_collect_points(N)

    R = m.generate_random_delivery_requests(N, P)

    #path1 = al.AStar(A, 0, 500, 0, phi, Temp, amplitude, offset, frequency)
    
    #s.stats_with_different_size(1000, 5000, 1000, 10, al.AStar, phi, Temp, amplitude, offset, frequency)

    # solution,  pickup, delivery, s0 = m.generate_random_solution(A, R)
    # print("Solution: ", solution)
    
    print("P: ", P)
    print("R: ", R)
    solution, cost = al.ants_colony(A, R, fourmis, phi, Temp, amplitude, offset, frequency, alpha, beta, gamma, rho, Q, iterations, max_iter_per_trial)
    solution_ok = m.verify_solution(R, solution)
    print("Solution: ", solution)
    print("Cost: ", cost)
    print("Solution OK: ", solution_ok)
    
    #al.linear_programming(A, R, phi, Temp, amplitude, offset, frequency)


    
    end = time.time()
    print("Time: ", end - start)

    # solution = m.generate_random_solution(A, C, P, R, V, s0, periods)

    # for i in range(20):
    #     C = a.C(A, phi, Temp, 0, 48, i, amplitude, offset, frequency)
    #     print("Cost: ", C)


    # for i in range(20):
    #     B = m.generate_random_symetrical_boolean_graph(N)
    #     path = m.AStar(B, 0, 48, 0, phi, Temp, amplitude, offset, frequency)
    #     print("Path: ", path)

    #print("Size of R: ", len(R))

    #print("A: ", A)
    #print("Path1: ", path1)


    #print("-------------------")
    # print("Solution: ", solution)

    #d.draw_graph_and_solution(A, solution)
    
if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    my_result = main()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('perf.txt', 'w+') as f:
        f.write(s.getvalue())
    
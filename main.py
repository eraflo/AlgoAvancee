import module as m
import actions as a
import algorithms as al
import utilities as u
import draw as d
import stats as s

import time
import cProfile
import re

#if __name__ == '__main__':

def main():
    V = 1 # Number of vehicles
    N = 2000 # Number of cities

    # Random Paramters
    phi = m.generate_random_symetrical_weighted_graph(N, 0.5, 5) # Initial phase 
    Temp = m.generate_random_symetrical_weighted_graph(N, 0, 20) # Initial random values
    amplitude = 0.2
    offset = 0.3
    frequency = 0.5


    start = time.time()
    A = m.generate_random_symetrical_boolean_graph(N)

    P = m.generate_random_collect_points(N)

    R = m.generate_random_delivery_requests(N, P)

    #path1 = al.AStar(A, 0, 500, 0, phi, Temp, amplitude, offset, frequency)
    
    #s.stats_with_different_size(1000, 5000, 1000, 10, al.AStar, phi, Temp, amplitude, offset, frequency)

    solution,  pickup, delivery, s0 = m.generate_random_solution(A, R)
    print("Solution: ", solution)


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

    #print("P: ", P)
    #print("R: ", R)
    #print("Size of R: ", len(R))

    #print("A: ", A)
    #print("Path1: ", path1)


    #print("-------------------")
    # print("Solution: ", solution)
    

cProfile.run('main()')
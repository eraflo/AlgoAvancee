import module as m
import time

if __name__ == '__main__':

    V = 1 # Number of vehicles
    N = 1000 # Number of cities

    # Random Paramters
    phi = m.generate_random_symetrical_weighted_graph(N, 0.5, 5)
    Temp = m.generate_random_symetrical_weighted_graph(N, 0, 20)
    amplitude = 0.7
    offset = 0.3
    frequency = 0.5


    start = time.time()
    A = m.generate_random_symetrical_boolean_graph(N)

    P = m.generate_random_collect_points(N)

    #R = m.generate_random_delivery_requests_v2(N, P)

    #path = m.AStar(A, 0, 5, 0, phi, Temp, amplitude, offset, frequency)


    end = time.time()
    print("Time: ", end - start)

    # solution = m.generate_random_solution(A, C, P, R, V, s0, periods)

    for i in range(20):
        B = m.generate_random_symetrical_boolean_graph(N)
        path = m.AStar(B, 0, 48, 0, phi, Temp, amplitude, offset, frequency)
        print("Path: ", path)

    # print("P: ", P)
    # print("R: ", R)


    print("-------------------")
    # print("Solution: ", solution)
    
import module as m

if __name__ == '__main__':
    V = 1 # Number of vehicles
    N = 10 # Number of cities

    s0 = 0 # Initial city


    periods = m.generate_random_periods(24)

    A = m.generate_random_symetrical_graph_for_periods(N, 0, 2, periods)

    P = m.generate_random_collect_points(N)

    R = m.generate_random_delivery_requests(N, P)

    C = m.generate_random_weights(A, 10)

    solution = m.generate_random_solution(A, C, P, R, V, s0, periods)

    print("A: ", A)
    print("P: ", P)
    print("R: ", R)
    print("C: ", C)
    print("periods: ", periods)

    print("-------------------")
    print("Solution: ", solution)
    
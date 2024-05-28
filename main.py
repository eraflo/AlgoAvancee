import module as m

if __name__ == '__main__':
    nb_nodes = 10
    nb_depots = 1

    max_weight = 100


    timeperiod = 1000
    step = 10

    cities_paths = m.generate_random_symetrical_graph(nb_nodes)
    weights = m.generate_weights_over_time(nb_nodes, max_weight, timeperiod, step)
    depots = m.generate_random_depot(nb_nodes, nb_depots)
    deliveries = m.generate_random_delivery(nb_nodes, nb_depots)

    solution = m.random_solution(cities_paths, depots, deliveries, timeperiod)
    solution = m.get_path(solution)
    print(depots)
    print(deliveries)
    print(solution)
    m.draw_whole_graph(cities_paths, weights[0], depots, deliveries, solution)
    # m.draw_whole_graph(cities_paths, weights[2], depots, deliveries, solution)
    # m.draw_whole_graph(cities_paths, weights[1], depots, deliveries, solution)
    
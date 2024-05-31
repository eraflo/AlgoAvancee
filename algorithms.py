import numpy as np
import random as rand
import math as m
from collections import deque
import heapq

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


def tabou(A, phi, Temp, amplitude, offset, frequency):
    """
    Tabou algorithm to find the shortest path between two nodes in a graph
    """
    pass
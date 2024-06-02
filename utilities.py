import numpy as np
import random as rand
import math as m
from collections import deque
import heapq


def remove_from_list_tuple(l, i):
    """
    Remove the tuple with i as first element from the list of tuples l.
    """
    for item in l:
        if item[0] == i:
            l.remove(item)
            break  
    return l

def connexity(A):
    n = len(A)
    visited = np.zeros(n, dtype=bool)
    stack = [0] 
    visited[0] = True

    while stack:
        node = stack.pop()
        neighbors = np.nonzero(A[node])[0]
        for i in neighbors:
            if not visited[i]:
                stack.append(i)
                visited[i] = True

    return np.all(visited)

import numpy as np
import random as rand
import math as m
from collections import deque
import heapq

import time

import module as m
import draw as d





def stats_with_different_size(start, end, step, repetitions, algo, phi, Temp, amplitude, offset, frequency):
    """
    Compare the execution time of the algorithm algo with different sizes of the graph A.
    """
    sizes = range(start, end, step)
    average_time = []
    max_time = []
    
    f = d.init_loading_bar(len(sizes))
    for n in sizes:
        timeRecorded = []
        for _ in range(repetitions):
            start = time.time()
            A = m.generate_random_symetrical_boolean_graph(n)

            P = m.generate_random_collect_points(n)

            R = m.generate_random_delivery_requests(n, P)
            algo(A, 0, n-1, 0, phi, Temp, amplitude, offset, frequency)
            timeRecorded.append(time.time() - start)
            d.update_loading_bar(f, f.value + 1)
            
        average_time.append(np.mean(timeRecorded))
        max_time.append(np.max(timeRecorded))
        
        
    return sizes, average_time, max_time


        
        
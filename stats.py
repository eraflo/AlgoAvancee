import numpy as np
import random as rand
import math as m
from collections import deque
import heapq

import time

import module as m



def stats_with_different_size(start, end, step, repetitions, algo, A, phi, Temp, amplitude, offset, frequency):
    """
    Compare the execution time of the algorithm algo with different sizes of the graph A.
    """
    sizes = range(start, end, step)
    average_time = []
    max_time = []
    
    for n in sizes:
        time = []
        for _ in range(repetitions):
            start = time.time()
            algo(A, 0, n-1, 0, phi, Temp, amplitude, offset, frequency)
            time.append(time.time() - start)
        average_time.append(np.mean(time))
        max_time.append(np.max(time))
        
    return sizes, average_time, max_time


        
        
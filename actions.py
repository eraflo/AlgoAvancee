import numpy as np
import random as rand
import math as m
from collections import deque
import heapq
import functools as f



def neighbors(A, i):
    """
    Return the neighbors of the city i in the graph A.
    """
    return [j for j in range(len(A[i])) if A[i][j] == 1]


@f.lru_cache(maxsize=None)
def C(A, phi, Temp, i, j, t, amplitude, offset, frequency):
    """
    Cost function between the cities i and j at time t.

    @param A: Graph of the cities.
    @param phi: Initial phases
    @param Temp: Initial values to simulate randomness.
    @param i: City of departure.
    @param j: City of arrival.
    @param t: Time.
    @param amplitude: Amplitude of the cost function.
    @param offset: Offset of the cost function.
    @param frequency: Frequency of the cost function.
    """
    return max(1, round((amplitude * m.sin(frequency * t + phi[i][j]) + offset) * Temp[i][j], 4)) if A[i][j] == 1 else float('inf')


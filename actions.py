import numpy as np
import random as rand
import math as m
from collections import deque
import heapq



def neighbors(A, i):
    """
    Return the neighbors of the city i in the graph A.
    """
    return [j for j in range(len(A[i])) if A[i][j] == 1]


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
    return max(0, round((amplitude * m.sin(frequency * t + phi[i][j]) + offset) * Temp[i][j], 4)) if A[i][j] == 1 else float('inf')



def pass_through(A, i, j):
    """"
    Mark the edge between the cities i and j as passed through.
    """
    A[i][j] = 1
    A[j][i] = 1



def collect(P, r):
    """
    Mark the object m from the request r as collected.
    """
    P[r[0]][r[1]] = 1



def deliver(D, r):
    """
    Mark the object m from the request r as delivered.
    """
    D[r[0]][r[1]] = 1



def get_city_passed_through(X, t):
    """
    Get the city passed through at time t.
    """
    for i in range(len(X[t])):
        for j in range(len(X[t][i])):
            if X[t][i][j]:
                return i, j
            

def check_delivery_done(D, R):
    """
    Check if all the deliveries have been done.
    """
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] == 1 and (i, j) in R:
                return False
    return True
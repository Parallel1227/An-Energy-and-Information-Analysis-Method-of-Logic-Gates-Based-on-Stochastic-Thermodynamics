import numpy as np
from numpy import random
from operator import itemgetter
from XOR_Functions import Gauss_in00, Gauss_in01, Gauss_in10, Gauss_in11, EC_XOR, MutualInfo_XOR
"""
This file executes GA
"""

# initialization
Size = 80
G = 100
CodeL = 10
Vmax = 15.0
Vmin = 4.0
pmax = 0.999
pmin = 0.001
a = 0.90
time = np.zeros(G)
F = np.zeros(Size)
bfi = np.zeros(G)
tempE = np.zeros((Size, 3 * CodeL))
E = random.randint(0, 2, (Size, 3 * CodeL))
BestJ = np.zeros(G)
maxeffi = 0
best_V = []
best_p1 = []
best_p2 = []

def Efficiency_XOR(V_d, pa, pb, a):
    EC = EC_XOR(V_d, pa, pb)
    I = MutualInfo_XOR(V_d, pa, pb, a)
    return I / EC


# GA
for k in range(G):
    time[k] = k
    for s in range(Size):
        y1 = 0
        y2 = 0
        y3 = 0
        m1 = E[s, :CodeL]
        # coding the variables
        for i in range(CodeL):
            y1 = y1 + m1[i] * pow(2, i)
        x1 = (Vmax - Vmin) * y1 / 1023 + Vmin  # supply voltage

        m2 = E[s, CodeL:CodeL + CodeL]
        for i in range(CodeL):
            y2 = y2 + m2[i] * pow(2, i)
        x2 = (pmax - pmin) * y2 / 1023 + pmin  # pa

        m3 = E[s, (CodeL + CodeL):]
        for i in range(CodeL):
            y3 = y3 + m3[i] * pow(2, i)
        x3 = (pmax - pmin) * y3 / 1023 + pmin  # pb

        F[s] = Efficiency_XOR(x1, x2, x3, 0.90)

    Ji = 1. / F

    BestJ[k] = min(Ji)
    fi = F

    Indexfi, Orderfi = zip(*sorted(enumerate(fi), key=itemgetter(1)))
    Bestfi = Orderfi[Size - 1]  # best fitness
    BestS = E[Indexfi[Size - 1], :]
    bfi[k] = Bestfi

    # Step 2 Calculate the probability that each individual will remain to the next generation
    fi_sum = sum(fi)
    fi_Size = (Orderfi / fi_sum) * Size
    fi_S = np.floor(fi_Size)

    kk = 0
    for i in range(Size):
        for j in range(int(fi_S[i])):
            tempE[kk, :] = E[Indexfi[i], :]
            kk += 1

    # Step 3  cross
    pc = 0.60
    n = int(np.ceil(30 * random.random()))
    for i in range(0, Size, 2):
        temp = random.random()
        if pc > temp:
            for j in range(n - 1, 3 * CodeL):
                tempE[i, j] = E[i + 1, j]
                tempE[i + 1, j] = E[i, j]

    tempE[Size - 1, :] = BestS
    E = tempE

    # Step 4 variation
    pm = 0.002

    for i in range(Size):
        for j in range(3 * CodeL):
            temp = random.random()
            if pm > temp:
                if tempE[i, j] == 0:
                    tempE[i, j] = 1
                else:
                    tempE[i, j] = 0

    tempE[Size - 1, :] = BestS  # keep the best chromesome
    E = tempE

    if (maxeffi < Bestfi):
        maxeffi = Bestfi
        best_V = BestS[:CodeL]
        best_p1 = BestS[CodeL:2 * CodeL]
        best_p2 = BestS[2 * CodeL:]

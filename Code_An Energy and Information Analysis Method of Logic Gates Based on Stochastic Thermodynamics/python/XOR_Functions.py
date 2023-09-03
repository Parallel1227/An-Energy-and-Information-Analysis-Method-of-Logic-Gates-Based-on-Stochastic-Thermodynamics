# -*- coding: utf-8 -*-
from pylab import *
import numpy as np
from scipy.linalg import null_space
from scipy import integrate

"""
This file describes the stochastic process of XOR's computation
and contains some basic functions
"""

def Gauss_in00(x):
    sigma2 = 0.005
    mu = 0.090022
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def Gauss_in01(x, Vd):
    sigma2 = 0.005
    mu = Vd + 0.0159655
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def Gauss_in10(x, Vd):
    sigma2 = 0.005
    mu = Vd + 0.0159655
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def Gauss_in11(x):
    sigma2 = 0.005
    mu = 0.090022
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def Fermi(x):
    return 1.0 / (exp(x) + 1)


def Bose(x):
    if (x > 1e-4):
        return 1.0 / (exp(x) - 1)
    else:
        return 1e5


def Gauss_in0(x, a):
    sigma2 = 0.005
    mu = a
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def NAND_propagation(Vin_A, Vin_B, Vout, V_D):
    A = zeros((16, 16))
    c = zeros(16)

    Gamma_l = 0.2
    Gamma_r = 0.2
    Gamma = 0.2
    Gamma_g = 0.2

    mu_l = 0.0
    mu_r = mu_l - V_D
    kBT = 1.0
    Cg = 200.0

    E_P1 = Vin_A
    E_P2 = Vin_B
    E_N1 = 1.5 * V_D - Vin_A
    E_N2 = 1.5 * V_D - Vin_B

    mu_g = 0.0 - Vout

    k_N2l = Gamma_l * Fermi((E_N2 - mu_l) / kBT)
    k_lN2 = Gamma_l * (1.0 - Fermi((E_N2 - mu_l) / kBT))
    k_P1r = Gamma_r * Fermi((E_P1 - mu_r) / kBT)
    k_rP1 = Gamma_r * (1.0 - Fermi((E_P1 - mu_r) / kBT))
    k_P2r = Gamma_r * Fermi((E_P2 - mu_r) / kBT)
    k_rP2 = Gamma_r * (1.0 - Fermi((E_P2 - mu_r) / kBT))
    k_P1g = Gamma_g * Fermi((E_P1 - mu_g) / kBT)
    k_gP1 = Gamma_g * (1.0 - Fermi((E_P1 - mu_g) / kBT))
    k_P2g = Gamma_g * Fermi((E_P2 - mu_g) / kBT)
    k_gP2 = Gamma_g * (1.0 - Fermi((E_P2 - mu_g) / kBT))
    k_N1g = Gamma_g * Fermi((E_N1 - mu_g) / kBT)
    k_gN1 = Gamma_g * (1.0 - Fermi((E_N1 - mu_g) / kBT))
    if (E_N1 > E_N2):
        k_N1N2 = Gamma * Bose((E_N1 - E_N2) / kBT)
        k_N2N1 = Gamma * (1 + Bose((E_N1 - E_N2) / kBT))
    else:
        k_N2N1 = Gamma * Bose((E_N2 - E_N1) / kBT)
        k_N1N2 = Gamma * (1 + Bose((E_N2 - E_N1) / kBT))
    if (E_N1 > E_P1):
        k_N1P1 = Gamma * Bose((E_N1 - E_P1) / kBT)
        k_P1N1 = Gamma * (1 + Bose((E_N1 - E_P1) / kBT))
    else:
        k_P1N1 = Gamma * Bose((E_P1 - E_N1) / kBT)
        k_N1P1 = Gamma * (1 + Bose((E_P1 - E_N1) / kBT))
    if (E_N1 > E_P2):
        k_N1P2 = Gamma * Bose((E_N1 - E_P2) / kBT)
        k_P2N1 = Gamma * (1 + Bose((E_N1 - E_P2) / kBT))
    else:
        k_P2N1 = Gamma * Bose((E_P2 - E_N1) / kBT)
        k_N1P2 = Gamma * (1 + Bose((E_P2 - E_N1) / kBT))
    if (E_P1 > E_P2):
        k_P1P2 = Gamma * Bose((E_P1 - E_P2) / kBT)
        k_P2P1 = Gamma * (1 + Bose((E_P1 - E_P2) / kBT))
    else:
        k_P2P1 = Gamma * Bose((E_P2 - E_P1) / kBT)
        k_P1P2 = Gamma * (1 + Bose((E_P2 - E_P1) / kBT))

    A[1][0] = k_P1r + k_P1g
    A[2][0] = k_P2r + k_P2g
    A[3][0] = k_N1g
    A[4][0] = k_N2l
    A[0][0] = -1.0 * (A[1][0] + A[2][0] + A[3][0] + A[4][0])
    A[0][1] = k_rP1 + k_gP1
    A[2][1] = k_P2P1
    A[3][1] = k_N1P1
    A[5][1] = k_P2r + k_P2g
    A[6][1] = k_N1g
    A[7][1] = k_N2l
    A[1][1] = -1.0 * (A[0][1] + A[2][1] + A[3][1] + A[5][1] + A[6][1] + A[7][1])
    A[0][2] = k_rP2 + k_gP2
    A[1][2] = k_P1P2
    A[3][2] = k_N1P2
    A[5][2] = k_P1r + k_P1g
    A[8][2] = k_N1g
    A[9][2] = k_N2l
    A[2][2] = -1.0 * (A[0][2] + A[1][2] + A[3][2] + A[5][2] + A[8][2] + A[9][2])
    A[0][3] = k_gN1
    A[1][3] = k_P1N1
    A[2][3] = k_P2N1
    A[4][3] = k_N2N1
    A[6][3] = k_P1r + k_P1g
    A[8][3] = k_P2r + k_P2g
    A[10][3] = k_N2l
    A[3][3] = -1.0 * (A[0][3] + A[1][3] + A[2][3] + A[4][3] + A[6][3] + A[8][3] + A[10][3])
    A[0][4] = k_lN2
    A[3][4] = k_N1N2
    A[7][4] = k_P1r + k_P1g
    A[9][4] = k_P2r + k_P2g
    A[10][4] = k_N1g
    A[4][4] = -1.0 * (A[0][4] + A[3][4] + A[7][4] + A[9][4] + A[10][4])
    A[1][5] = k_rP2 + k_gP2
    A[2][5] = k_rP1 + k_gP1
    A[6][5] = k_N1P2
    A[8][5] = k_N1P1
    A[13][5] = k_N2l
    A[14][5] = k_N1g
    A[5][5] = -1.0 * (A[1][5] + A[2][5] + A[6][5] + A[8][5] + A[13][5] + A[14][5])
    A[1][6] = k_gN1
    A[3][6] = k_rP1 + k_gP1
    A[5][6] = k_P2N1
    A[7][6] = k_N2N1
    A[8][6] = k_P2P1
    A[12][6] = k_N2l
    A[14][6] = k_P2r + k_P2g
    A[6][6] = -1.0 * (A[1][6] + A[3][6] + A[5][6] + A[7][6] + A[8][6] + A[12][6] + A[14][6])
    A[1][7] = k_lN2
    A[4][7] = k_rP1 + k_gP1
    A[6][7] = k_N1N2
    A[9][7] = k_P2P1
    A[10][7] = k_N1P1
    A[12][7] = k_N1g
    A[13][7] = k_P2r + k_P2g
    A[7][7] = -1.0 * (A[1][7] + A[4][7] + A[6][7] + A[9][7] + A[10][7] + A[12][7] + A[13][7])
    A[2][8] = k_gN1
    A[3][8] = k_gP2 + k_rP2
    A[5][8] = k_P1N1
    A[6][8] = k_P1P2
    A[9][8] = k_N2N1
    A[11][8] = k_N2l
    A[14][8] = k_P1r + k_P1g
    A[8][8] = -1.0 * (A[2][8] + A[3][8] + A[5][8] + A[6][8] + A[9][8] + A[11][8] + A[14][8])
    A[2][9] = k_lN2
    A[4][9] = k_rP2 + k_gP2
    A[7][9] = k_P1P2
    A[8][9] = k_N1N2
    A[10][9] = k_N1P2
    A[11][9] = k_N1g
    A[13][9] = k_P1r + k_P1g
    A[9][9] = -1.0 * (A[2][9] + A[4][9] + A[7][9] + A[8][9] + A[10][9] + A[11][9] + A[13][9])
    A[3][10] = k_lN2
    A[4][10] = k_gN1
    A[7][10] = k_P1N1
    A[9][10] = k_P2N1
    A[11][10] = k_P2r + k_P2g
    A[12][10] = k_P1r + k_P1g
    A[10][10] = -1.0 * (A[3][10] + A[4][10] + A[7][10] + A[9][10] + A[11][10] + A[12][10])
    A[8][11] = k_lN2
    A[9][11] = k_gN1
    A[10][11] = k_rP2 + k_gP2
    A[12][11] = k_P1P2
    A[13][11] = k_P1N1
    A[15][11] = k_P1r + k_P1g
    A[11][11] = -1.0 * (A[8][11] + A[9][11] + A[10][11] + A[12][11] + A[13][11] + A[15][11])
    A[6][12] = k_lN2
    A[7][12] = k_gN1
    A[10][12] = k_rP1 + k_gP1
    A[11][12] = k_P2P1
    A[13][12] = k_P2N1
    A[15][12] = k_P2r + k_P2g
    A[12][12] = -1.0 * (A[6][12] + A[7][12] + A[10][12] + A[11][12] + A[13][12] + A[15][12])
    A[5][13] = k_lN2
    A[7][13] = k_rP2 + k_gP2
    A[9][13] = k_rP1 + k_gP1
    A[11][13] = k_N1P1
    A[12][13] = k_N1P2
    A[14][13] = k_N1N2
    A[15][13] = k_N1g
    A[13][13] = -1.0 * (A[5][13] + A[7][13] + A[9][13] + A[11][13] + A[12][13] + A[14][13] + A[15][13])
    A[5][14] = k_gN1
    A[6][14] = k_rP2 + k_gP2
    A[8][14] = k_rP1 + k_gP1
    A[13][14] = k_N2N1
    A[15][14] = k_N2l
    A[14][14] = -1.0 * (A[5][14] + A[6][14] + A[8][14] + A[13][14] + A[15][14])
    A[11][15] = k_rP1 + k_gP1
    A[12][15] = k_rP2 + k_gP2
    A[13][15] = k_gN1
    A[14][15] = k_lN2
    A[15][15] = -1.0 * (A[11][15] + A[12][15] + A[13][15] + A[14][15])

    p = null_space(A)
    total = 0.0
    for j in range(16):
        total += p[j][0]
    for j in range(16):
        p[j][0] = p[j][0] * 1.0 / total
    p_P1 = p[1][0] + p[5][0] + p[6][0] + p[7][0] + p[12][0] + p[13][0] + p[14][0] + p[15][0]
    p_P2 = p[2][0] + p[5][0] + p[8][0] + p[9][0] + p[11][0] + p[13][0] + p[14][0] + p[15][0]
    p_N1 = p[3][0] + p[6][0] + p[8][0] + p[10][0] + p[11][0] + p[12][0] + p[14][0] + p[15][0]
    p_N2 = p[4][0] + p[7][0] + p[9][0] + p[10][0] + p[11][0] + p[12][0] + p[13][0] + p[15][0]

    J1 = k_N2l * (1 - p_N2)
    J2 = k_lN2 * p_N2
    J3 = k_P1r * (1 - p_P1)
    J4 = k_rP1 * p_P1
    J5 = k_P2r * (1 - p_P2)
    J6 = k_rP2 * p_P2
    J7 = k_P1g * (1 - p_P1)
    J8 = k_gP1 * p_P1
    J9 = k_P2g * (1 - p_P2)
    J10 = k_gP2 * p_P2
    J11 = k_N1g * (1 - p_N1)
    J12 = k_gN1 * p_N1

    Jg = J8 - J7 + J12 - J11 + J10 - J9

    Vout -= 1.0 * Jg * tint / Cg
    dissipation = 1.0 * (J1 - J2) * tint * (mu_l - mu_g) + 1.0 * (J3 - J4 + J5 - J6) * tint * (mu_r - mu_g)

    return Vout, dissipation


def XOR_propagation(Vin_A, Vin_B, Vout_NAND0, Vout_NAND1, Vout_NAND2, Vout_NAND3, V_D):
    VinA = np.zeros(4)
    VinB = np.zeros(4)
    Vout = np.array([Vout_NAND0, Vout_NAND1, Vout_NAND2, Vout_NAND3])
    dissipation_NAND = np.zeros(4)
    dissipation_tot = 0.0
    VinA[0] = Vin_A
    VinB[0] = Vin_B
    VinA[1] = Vin_A
    VinB[2] = Vin_B

    VinB[1] = Vout[0]
    VinA[2] = Vout[0]
    VinA[3] = Vout[1]
    VinB[3] = Vout[2]

    for j in range(4):
        Vout[j], dissipation_NAND[j] = NAND_propagation(VinA[j], VinB[j], Vout[j], V_D)
        dissipation_tot += dissipation_NAND[j]
    return Vout[0], Vout[1], Vout[2], Vout[3], dissipation_tot


# EC=Energy Consumption
def EC_XOR(V_d, pa, pb):
    V_d = 5.0
    EC_matrix = [[5791.63417232645, 10240.643798967365, 10287.915793501656, 16661.983055743865],
                 [8779.678559214039, 5682.688984275281, 10284.901383868311, 16061.58449921476],
                 [9151.67063026872, 10083.330612729244, 5311.164890917032, 16296.224377104185],
                 [10523.542275672926, 15456.859048801445, 14972.188200378172, 14520.473805522808]
                 ]
    p00 = pa * pb
    p01 = pa * (1 - pb)
    p10 = pb * (1 - pa)
    p11 = (1 - pa) * (1 - pb)
    EC_XOR = EC_matrix[0][0] * p00 * p00 + EC_matrix[0][1] * p00 * p01 + EC_matrix[0][2] * p00 * p10 + EC_matrix[0][
        3] * p00 * p11 + \
             EC_matrix[1][0] * p01 * p00 + EC_matrix[1][1] * p01 * p01 + EC_matrix[1][2] * p01 * p10 + EC_matrix[1][
                 3] * p01 * p11 + \
             EC_matrix[2][0] * p10 * p00 + EC_matrix[2][1] * p10 * p01 + EC_matrix[2][2] * p10 * p10 + EC_matrix[2][
                 3] * p10 * p11 + \
             EC_matrix[3][0] * p11 * p00 + EC_matrix[3][1] * p11 * p01 + EC_matrix[3][2] * p11 * p10 + EC_matrix[3][
                 3] * p11 * p11
    return EC_XOR


def MutualInfo_XOR(V_d, pa, pb, a):

    Vd = V_d
    p_cond = np.zeros(12)
    p_joint = np.zeros(12)

    p_Y = np.zeros(3)

    # y=0
    p_cond[0] = integrate.quad(Gauss_in00, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]  # P(y=0|a=0,b=0)
    p_cond[1] = integrate.quad(Gauss_in01, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]
    p_cond[2] = integrate.quad(Gauss_in10, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]
    p_cond[3] = integrate.quad(Gauss_in11, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]

    # y=1
    p_cond[4] = integrate.quad(Gauss_in00, a * Vd, np.inf, epsabs=1e-28)[0]  # P(y=1|a=0,b=0)
    p_cond[5] = integrate.quad(Gauss_in01, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    p_cond[6] = integrate.quad(Gauss_in10, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    p_cond[7] = integrate.quad(Gauss_in11, a * Vd, np.inf, epsabs=1e-28)[0]

    # y=null
    p_cond[8] = 1.0 - p_cond[0] - p_cond[4]
    p_cond[9] = 1.0 - p_cond[1] - p_cond[5]
    p_cond[10] = 1.0 - p_cond[2] - p_cond[6]
    p_cond[11] = 1.0 - p_cond[3] - p_cond[7]

    for k in range(0, 12):
        if p_cond[k] < 0:
            p_cond[k] = 0
        if p_cond[k] > 1:
            p_cond[k] = 1

    p_00 = pa * pb
    p_01 = pa * (1.0 - pb)
    p_10 = (1.0 - pa) * pb
    p_11 = (1.0 - pa) * (1.0 - pb)

    for i in range(0, 12, 4):
        p_joint[i] = p_cond[i] * p_00

    for i in range(1, 12, 4):
        p_joint[i] = p_cond[i] * p_01

    for i in range(2, 12, 4):
        p_joint[i] = p_cond[i] * p_10

    for i in range(3, 12, 4):
        p_joint[i] = p_cond[i] * p_11

    p_Y[0] = p_joint[0] + p_joint[1] + p_joint[2] + p_joint[3]
    p_Y[1] = p_joint[4] + p_joint[5] + p_joint[6] + p_joint[7]
    p_Y[2] = p_joint[8] + p_joint[9] + p_joint[10] + p_joint[11]

    I_mutual = 0.0  # traversing all 12 cases
    if p_joint[0] > 0:
        I_mutual += p_joint[0] * np.log2(p_cond[0] / p_Y[0])  # P(a=0,b=0, y=0)*log(P(y=0|a=0, b=0)/P(y=0))
    if p_joint[1] > 0:
        I_mutual += p_joint[1] * np.log2(p_cond[1] / p_Y[0])  # P(a=0,b=1, y=0)*log(P(y=0|a=0, b=1)/P(y=0))
    if p_joint[2] > 0:
        I_mutual += p_joint[2] * np.log2(p_cond[2] / p_Y[0])  # P(a=1,b=0, y=0)*log(P(y=0|a=1, b=0)/P(y=0))
    I_mutual += p_joint[3] * np.log2(p_cond[3] / p_Y[0])  # P(a=1,b=1, y=0)*log(P(y=0|a=1, b=1)/P(y=0))

    if p_joint[4] > 0:
        I_mutual += p_joint[4] * np.log2(p_cond[4] / p_Y[1])  # P(a=0,b=0, y=1)*log(P(y=1|a=0, b=0)/P(y=1))
    if p_joint[5] > 0:
        I_mutual += p_joint[5] * np.log2(p_cond[5] / p_Y[1])  # P(a=0,b=1, y=1)*log(P(y=1|a=0, b=1)/P(y=1))
    if p_joint[6] > 0:
        I_mutual += p_joint[6] * np.log2(p_cond[6] / p_Y[1])  # P(a=1,b=0, y=1)*log(P(y=1|a=1, b=0)/P(y=1))
    if p_joint[7] > 0:
        I_mutual += p_joint[7] * np.log2(p_cond[7] / p_Y[1])  # P(a=1,b=1, y=1)*log(P(y=1|a=1, b=1)/P(y=1))

    if p_joint[8] > 0:
        I_mutual += p_joint[8] * np.log2(p_cond[8] / p_Y[2])  # P(a=0,b=0, y=null)*log(P(y=null|a=0, b=0)/P(y=null))
    if p_joint[9] > 0:
        I_mutual += p_joint[9] * np.log2(p_cond[9] / p_Y[2])  # P(a=0,b=1, y=null)*log(P(y=null|a=0, b=1)/P(y=null))
    if p_joint[10] > 0:
        I_mutual += p_joint[10] * np.log2(p_cond[10] / p_Y[2])  # P(a=1,b=0, y=null)*log(P(y=null|a=1, b=0)/P(y=null))
    if p_joint[11] > 0:
        I_mutual += p_joint[11] * np.log2(p_cond[11] / p_Y[2])  # P(a=1,b=1, y=null)*log(P(y=null|a=1, b=1)/P(y=null))

    return I_mutual


# initialization
tint = 1000
T = 10000000
a = 0.90
Vout_XOR1 = 0.0
dissipation_XOR1 = 0.0
Vout_XOR2 = 0.0
dissipation_XOR2 = 0.0
Vout_XOR3 = 0.0
dissipation_XOR3 = 0.0
Vout_XOR4 = 0.0
dissipation_XOR4 = 0.0
Ntot = int(T / tint)
output1 = np.zeros(Ntot)
error1 = np.zeros(Ntot)
diss1 = np.zeros(Ntot)
output2 = np.zeros(Ntot)
error2 = np.zeros(Ntot)
diss2 = np.zeros(Ntot)
output3 = np.zeros(Ntot)
error3 = np.zeros(Ntot)
diss3 = np.zeros(Ntot)
output4 = np.zeros(Ntot)
error4 = np.zeros(Ntot)
diss4 = np.zeros(Ntot)
time = np.zeros(Ntot)
diss_arr = []
Vout_NAND0 = np.array([5.0, 5.0, 5.0, 5.0])
Vout_NAND1 = np.array([5.0, 5.0, 5.0, 5.0])
Vout_NAND2 = np.array([5.0, 5.0, 5.0, 5.0])
Vout_NAND3 = np.array([0.0, 0.0, 0.0, 0.0])

for i in range(Ntot):
    Vout_NAND0[0], Vout_NAND1[0], Vout_NAND2[0], Vout_NAND3[0], dissipation_XOR1 = XOR_propagation(5.0, 5.0,
                                                                                                   Vout_NAND0[0],
                                                                                                   Vout_NAND1[0],
                                                                                                   Vout_NAND2[0],
                                                                                                   Vout_NAND3[0],
                                                                                                   5.0)

    output1[i] = Vout_NAND3[0]
    diss1[i] = diss1[i - 1] + dissipation_XOR1
    error1[i] = 1 - integrate.quad(Gauss_in0, -1 * np.inf, (1 - a) * 5.0, args=(output1[i]))[0]
    # A=1,B=0
    Vout_NAND0[1], Vout_NAND1[1], Vout_NAND2[1], Vout_NAND3[1], dissipation_XOR2 = XOR_propagation(5.0, 0.0,
                                                                                                   Vout_NAND0[1],
                                                                                                   Vout_NAND1[1],
                                                                                                   Vout_NAND2[1],
                                                                                                   Vout_NAND3[1],
                                                                                                   5.0)

    # A=1,B=0
    output2[i] = Vout_NAND3[1]
    diss2[i] = diss2[i - 1] + dissipation_XOR2
    error2[i] = 1 - integrate.quad(Gauss_in0, a * 5.0, np.inf, args=(output2[i]))[0]

    # A=0, B=1
    Vout_NAND0[2], Vout_NAND1[2], Vout_NAND2[2], Vout_NAND3[2], dissipation_XOR3 = XOR_propagation(0.0, 5.0,
                                                                                                   Vout_NAND0[2],
                                                                                                   Vout_NAND1[2],
                                                                                                   Vout_NAND2[2],
                                                                                                   Vout_NAND3[2],
                                                                                                   5.0)
    output3[i] = Vout_NAND3[2]
    diss3[i] = diss3[i - 1] + dissipation_XOR3
    error3[i] = 1 - integrate.quad(Gauss_in0, a * 5.0, np.inf, args=(output3[i]))[0]

    # A=0, B=0
    Vout_NAND0[3], Vout_NAND1[3], Vout_NAND2[3], Vout_NAND3[3], dissipation_XOR4 = XOR_propagation(0.0, 0.0,
                                                                                                   Vout_NAND0[3],
                                                                                                   Vout_NAND1[3],
                                                                                                   Vout_NAND2[3],
                                                                                                   Vout_NAND3[3],
                                                                                                   5.0)
    output4[i] = Vout_NAND3[3]
    diss4[i] = diss4[i - 1] + dissipation_XOR4
    error4[i] = 1 - integrate.quad(Gauss_in0, -1 * np.inf, (1 - a) * 5.0, args=(output4[i]))[0]
    if error1[i] < 0.01 and error2[i] < 0.01 and error3[i] < 0.01 and error4[i] < 0.01:
        diss_arr.append(diss4[i])
        diss_arr.append(diss3[i])
        diss_arr.append(diss2[i])
        diss_arr.append(diss1[i])
        break

    time[i] = i * tint










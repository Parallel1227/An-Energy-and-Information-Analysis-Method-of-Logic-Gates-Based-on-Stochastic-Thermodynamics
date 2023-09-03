from pylab import *
import numpy as np
from scipy import integrate
from XOR_Functions import EC_XOR

"""
This file contains the energy efficiency of parity check circuits
"""


def Gauss_out0(x):
    sigma2 = 0.005
    mu = 0.090022
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def Gauss_out1(x, Vd):
    sigma2 = 0.005
    mu = Vd + 0.0159655
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def MutualInfo_parity(V_d, a, p1, p2, p3):
    Vd = V_d
    p_cond = np.zeros(24)
    p_joint = np.zeros(24)

    p_Y = np.zeros(3)

    p_cond[0] = integrate.quad(Gauss_out0, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]  # P(y=0|a=0,b=0,c=0)
    p_cond[1] = integrate.quad(Gauss_out1, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]
    p_cond[2] = integrate.quad(Gauss_out1, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]
    p_cond[3] = integrate.quad(Gauss_out0, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]
    p_cond[4] = integrate.quad(Gauss_out1, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]
    p_cond[5] = integrate.quad(Gauss_out0, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]
    p_cond[6] = integrate.quad(Gauss_out0, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28)[0]
    p_cond[7] = integrate.quad(Gauss_out1, -1 * np.inf, (1 - a) * Vd, epsabs=1e-28, args=Vd)[0]

    p_cond[8] = integrate.quad(Gauss_out0, a * Vd, np.inf, epsabs=1e-28)[0]  # P(y=1|a=0,b=0,c=0)
    p_cond[9] = integrate.quad(Gauss_out1, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    p_cond[10] = integrate.quad(Gauss_out1, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    p_cond[11] = integrate.quad(Gauss_out0, a * Vd, np.inf, epsabs=1e-28)[0]
    p_cond[12] = integrate.quad(Gauss_out1, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    p_cond[13] = integrate.quad(Gauss_out0, a * Vd, np.inf, epsabs=1e-28)[0]
    p_cond[14] = integrate.quad(Gauss_out0, a * Vd, np.inf, epsabs=1e-28)[0]
    p_cond[15] = integrate.quad(Gauss_out1, a * Vd, np.inf, epsabs=1e-28, args=Vd)[0]
    # y=null
    p_cond[16] = 1.0 - p_cond[0] - p_cond[8]  # P(y=null|a=0,b=0,c=0)
    p_cond[17] = 1.0 - p_cond[1] - p_cond[9]
    p_cond[18] = 1.0 - p_cond[2] - p_cond[10]
    p_cond[19] = 1.0 - p_cond[3] - p_cond[11]
    p_cond[20] = 1.0 - p_cond[4] - p_cond[12]
    p_cond[21] = 1.0 - p_cond[5] - p_cond[13]
    p_cond[22] = 1.0 - p_cond[6] - p_cond[14]
    p_cond[23] = 1.0 - p_cond[7] - p_cond[15]

    for k in range(0, 24):
        if p_cond[k] < 0:
            p_cond[k] = 0
        if p_cond[k] > 1:
            p_cond[k] = 1

    p_000 = p1 * p2 * p3
    p_001 = p1 * p2 * (1.0 - p3)
    p_010 = p1 * (1.0 - p2) * p3
    p_011 = p1 * (1.0 - p2) * (1.0 - p3)
    p_100 = (1.0 - p1) * p2 * p3
    p_101 = (1.0 - p1) * p2 * (1.0 - p3)
    p_110 = (1.0 - p1) * (1.0 - p2) * p3
    p_111 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    for i in range(0, 24, 8):
        p_joint[i] = p_cond[i] * p_000

    for i in range(1, 24, 8):
        p_joint[i] = p_cond[i] * p_001

    for i in range(2, 24, 8):
        p_joint[i] = p_cond[i] * p_010

    for i in range(3, 24, 8):
        p_joint[i] = p_cond[i] * p_011

    for i in range(4, 24, 8):
        p_joint[i] = p_cond[i] * p_100

    for i in range(5, 24, 8):
        p_joint[i] = p_cond[i] * p_101

    for i in range(6, 24, 8):
        p_joint[i] = p_cond[i] * p_110

    for i in range(7, 24, 8):
        p_joint[i] = p_cond[i] * p_111

    p_Y[0] = p_joint[0] + p_joint[1] + p_joint[2] + p_joint[3] + p_joint[4] + p_joint[5] + p_joint[6] + p_joint[7]
    p_Y[1] = p_joint[8] + p_joint[9] + p_joint[10] + p_joint[11] + p_joint[12] + p_joint[13] + p_joint[14] + p_joint[15]
    p_Y[2] = p_joint[16] + p_joint[17] + p_joint[18] + p_joint[19] + p_joint[20] + p_joint[21] + p_joint[22] + p_joint[
        23]

    I_mutual = 0.0

    if p_joint[0] > 0:
        I_mutual += p_joint[0] * np.log2(p_cond[0] / p_Y[0])
    if p_joint[1] > 0:
        I_mutual += p_joint[1] * np.log2(p_cond[1] / p_Y[0])
    if p_joint[2] > 0:
        I_mutual += p_joint[2] * np.log2(p_cond[2] / p_Y[0])
    I_mutual += p_joint[3] * np.log2(p_cond[3] / p_Y[0])
    if p_joint[4] > 0:
        I_mutual += p_joint[4] * np.log2(p_cond[4] / p_Y[0])
    if p_joint[5] > 0:
        I_mutual += p_joint[5] * np.log2(p_cond[5] / p_Y[0])
    if p_joint[6] > 0:
        I_mutual += p_joint[6] * np.log2(p_cond[6] / p_Y[0])
    if p_joint[7] > 0:
        I_mutual += p_joint[7] * np.log2(p_cond[7] / p_Y[0])

    if p_joint[8] > 0:
        I_mutual += p_joint[8] * np.log2(p_cond[8] / p_Y[1])
    if p_joint[9] > 0:
        I_mutual += p_joint[9] * np.log2(p_cond[9] / p_Y[1])
    if p_joint[10] > 0:
        I_mutual += p_joint[10] * np.log2(p_cond[10] / p_Y[1])
    if p_joint[11] > 0:
        I_mutual += p_joint[11] * np.log2(p_cond[11] / p_Y[1])
    if p_joint[12] > 0:
        I_mutual += p_joint[12] * np.log2(p_cond[12] / p_Y[1])
    if p_joint[13] > 0:
        I_mutual += p_joint[13] * np.log2(p_cond[13] / p_Y[1])
    if p_joint[14] > 0:
        I_mutual += p_joint[14] * np.log2(p_cond[14] / p_Y[1])
    if p_joint[15] > 0:
        I_mutual += p_joint[15] * np.log2(p_cond[15] / p_Y[1])

    if p_joint[16] > 0:
        I_mutual += p_joint[16] * np.log2(p_cond[16] / p_Y[2])
    if p_joint[17] > 0:
        I_mutual += p_joint[17] * np.log2(p_cond[17] / p_Y[2])
    if p_joint[18] > 0:
        I_mutual += p_joint[18] * np.log2(p_cond[18] / p_Y[2])
    if p_joint[19] > 0:
        I_mutual += p_joint[19] * np.log2(p_cond[19] / p_Y[2])
    if p_joint[20] > 0:
        I_mutual += p_joint[20] * np.log2(p_cond[20] / p_Y[2])
    if p_joint[21] > 0:
        I_mutual += p_joint[21] * np.log2(p_cond[21] / p_Y[2])
    if p_joint[22] > 0:
        I_mutual += p_joint[22] * np.log2(p_cond[22] / p_Y[2])
    if p_joint[23] > 0:
        I_mutual += p_joint[23] * np.log2(p_cond[23] / p_Y[2])

    return I_mutual


# EC=Energy Consumption
def EC_parity(V_d, pa, pb, pc):
    EC_XOR1 = EC_XOR(V_d, pa, pb)
    EC_XOR2 = EC_XOR(V_d, pa * pb + (1 - pa) * (1 - pb), pc)
    return EC_XOR1 + EC_XOR2


def get_parity_efficiency():
    V_d = 5.0
    pa = pb = pc = 0.5
    ParityCheckCircuits_dissipation = EC_parity(V_d, pa, pb, pc)
    ParityCheckCircuits_information = MutualInfo_parity(V_d, 0.90, pa, pb, pc)
    ParityCheckCircuits_efficiency = ParityCheckCircuits_information / ParityCheckCircuits_dissipation
    return ParityCheckCircuits_efficiency


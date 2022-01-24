# -*- coding: utf-8 -*-
"""
Code for "Team MMSE Precoding with Applications to Cell-free Massive MIMO"
Outupt: Comparison among different CSIT sharing patterns (Fig. 2)
Author: Lorenzo Miretti
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log2, log10, diag, eye
from numpy.linalg import norm, inv, pinv

# Parameters
N_sim = 100          # Monte Carlo iterations for performance evaluation
N_cdf = 100          # Number of points for CDF 
L = 30               # number of TXs 
N = 2                # number of antennas per TX
K = 7                # number of single-antenna RXs                             
P = 100/K            # TX power [mW]
r_lim = (60,50)      # radius of service area [m] and radius of user distribution [m]

def main():
    R_LTMMSE = []   
    R_UTMMSE = []   
    R_MMSE = []    

    for n in range(N_cdf):
        # Draw random scenario (user positions) and compute path losses
        G = generate_scenario(plot = False)
        
        # Draw set of channel realizations for a given scenario
        H_list = get_channel_realizations(G)
        
        # Local TMMSE
        C_list = get_parameters_LTMMSE(H_list)
        R = simulate_scenario(H_list,LTMMSE,C_list)
        R_LTMMSE = R_LTMMSE + R    

        # Unidirectional TMMSE
        Pi = get_parameters_UTMMSE(H_list)
        R = simulate_scenario(H_list,UTMMSE,Pi)
        R_UTMMSE = R_UTMMSE + R                         

        # Centralized MMSE
        R = simulate_scenario(H_list,MMSE,None)
        R_MMSE = R_MMSE + R
        
        print("Progress: %.1f%%" % (100 * (n+1)/N_cdf)) 

    # Plot CDF of achievable rates 
    fontsize = 16
    msize = 12
    lwidth = 4
    N_markers = 10
    marker_sep = round(K*N_cdf/N_markers)
    y_axis = np.arange(1,K*N_cdf+1)/(K*N_cdf)
    R_LTMMSE.sort()
    plt.plot(R_LTMMSE, y_axis,'-o', lw = lwidth, ms = msize, markevery=marker_sep,label='Local Team MMSE')
    R_UTMMSE.sort()
    plt.plot(R_UTMMSE, y_axis,'-s', lw = lwidth, ms = msize, markevery=marker_sep, label='Unidirectional Team MMSE')
    R_MMSE.sort()
    plt.plot(R_MMSE, y_axis,'->', lw = lwidth, ms = msize, markevery=marker_sep, label='Centralized MMSE')
    plt.xlim(left=0)
    plt.ylim((0,1))
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xlabel('Rate [b/s/Hz]', fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)
    plt.show()

def UTMMSE(H,Pi):
    """ Unidirectional TMMSE precoding, wrapper function for recursive computation
    """
    # List of precoders
    T = [np.zeros((N,K),dtype=complex)] * L
    # Compute precoders recursively
    unidirectionalTMMSE(H,Pi,T,0,eye(K,dtype=complex))
    # Convert list into matrix
    T = np.vstack(T)
    return T

def unidirectionalTMMSE(H,Pi,T,l,S_bar):
    """ Recursive routine for unidirectional TMMSE precoding, 
        involving an information matrix V_bar sequentially updated and forwarded from TX 1 to TX L,
        and statistical information given by Pi
    """
    # Compute precoder using local channel measurements and S_bar
    Hl = H[:,l*N:(l+1)*N]
    Fl = pinv(herm(Hl)@Hl + eye(N)/P) @ herm(Hl)
    Pl = Hl @ Fl
    Vl = pinv(eye(K)-Pi[l] @ Pl) @ (eye(K)-Pi[l])
    T[l] = Fl @ Vl @ S_bar
    if l < L-1:  # recursive call, forward updated S_bar
        Vl_bar = eye(K) - Pl @ Vl
        unidirectionalTMMSE(H,Pi,T,l+1,Vl_bar @ S_bar)

def get_parameters_UTMMSE(H_list):
    """ Monte Carlo estimation of statistical parameters for unidirectional TMMSE precoding
        Iterative implementation, could have been implemented recursively similarly to the unidirectionalTMMSE routine
    """
    # List of matrices Pi[l] to be computed using statistical information 
    Pi = [np.zeros((K,K),dtype=complex)] * L
    # Iterative computation from TX L to TX 1
    for l in range(L-1,0,-1):
        # Estimate auxiliary statistical quantities
        E_PS = np.zeros((K,K),dtype=complex)
        E_Sbar = np.zeros((K,K),dtype = complex)
        for n in range(N_sim): 
            H = H_list[n] 
            Hl = H[:,l*N:(l+1)*N]
            Pl = Hl @ pinv(herm(Hl)@Hl + eye(N)/P) @ herm(Hl)
            Vl = pinv(eye(K)-Pi[l] @ Pl) @ (eye(K)-Pi[l])
            Vl_bar = pinv(eye(K)- Pl @ Pi[l]) @ (eye(K)-Pl)
            E_PS += Pl @ Vl / N_sim
            E_Sbar += Vl_bar / N_sim
        # Update Pi[l-1] using Pi[l] and auxiliary statistical quantities
        Pi[l-1] = E_PS + Pi[l] @ E_Sbar
    return Pi

def LTMMSE(H,C_list):
    """ Local TMMSE precoding
    """
    T = np.zeros((L*N,K),dtype = complex)
    for l in range(L):
        Hl = H[:,l*N:(l+1)*N]
        Fl = pinv(herm(Hl)@Hl + eye(N)/P) @ herm(Hl)
        T[l*N:(l+1)*N,:] = Fl @ C_list[l]   
    return T

def get_parameters_LTMMSE(H_list):
    """ Monte Carlo estimation of statistical parameters for local TMMSE precoding
    """
    # List of matrices Pi[l] to be computed using statistical information
    Pi = [np.zeros((K,K),dtype=complex)] * L
    for l in range(L):
        Pi_l = np.zeros((K,K),dtype=complex)
        for n in range(N_sim): 
            H = H_list[n] 
            Hl = H[:,l*N:(l+1)*N]
            Pi_l += Hl @ pinv(herm(Hl)@Hl + eye(N)/P) @ herm(Hl)/N_sim
        Pi[l] = Pi_l
    # Build linear system of equations using Pi
    A = np.zeros((K*L,K*L),dtype=complex)
    for l in range(L):
        for j in range(L):
            if j == l:
                A[K*l:K*(l+1),K*j:K*(j+1)] = np.eye(K,dtype=complex)
            else:
                A[K*l:K*(l+1),K*j:K*(j+1)] = Pi[j]  
    I = np.zeros((K*L,K), dtype = complex)
    for l in range(L):
        I[K*l:K*(l+1),:] = np.eye(K,dtype=complex)
    # Solve system and find optimal coefficients
    C = inv(A) @ I
    # Convert into list of matrices
    C_list = []
    for l in range(L):
        C_list.append(C[K*l:K*(l+1),:])             
    return C_list

def MMSE(H,parameters=None):
    """ Centralized MMSE precoding
    """
    T = pinv(herm(H) @ H + np.eye(N*L)/P) @ herm(H)
    return T

def simulate_scenario(H_list,precoder,parameters):
    ''' Numerically evaluate the performance of a given precoding scheme, using the MSE lower bound
        Since the schemes are optimal, the lower bound coincides with the UatF bound (with unitary UL TX power), 
        or with the hardening bound after applying the UL-DL duality principle.
    ''' 
    # Estimate MSE
    MSE = [0] * K
    for n in range(N_sim):
        H = H_list[n]
        # Compute precoders
        T = precoder(H,parameters)
        # Update MSE estimate for all users
        for k in range(K):     
            e =  np.zeros(K)  
            e[k] = 1
            MSE[k] += (norm(e-H@T[:,k])**2 + norm(T[:,k])**2/P)/N_sim
    # Compute achievable rates
    R = [-log2(MSE[k]) for k in range(K)]
    return R   

def generate_scenario(plot=False):
    """ Circular service area of radius r1.
        The stripe is wrapped around the perimeter.
        Users are drawn in a concentric circle of radius r2
    """
    # Bandwidth
    B = 20*10**6
    # Noise figure (dB)
    noiseFigure = 7
    # Noise power (dBm)
    N0 = -174 + 10*log10(B) + noiseFigure
    # Pathloss exponent
    PL_exp = 3.67
    # Frequency (GHz)
    f_c = 2
    # Average channel gain in dB at a reference distance of 1 meter. 
    G_const = -22.7-26*log10(f_c)

    # Positions of TXs and RXs (in meters, polar coordinates)
    r1 = r_lim[0]
    r2 = r_lim[1]
    r2_RX = np.random.rand(K)*r2**2
    theta_RX = np.random.rand(K)*2*np.pi
    polar_RX = [np.array([sqrt(r2_RX[k]),theta_RX[k]]) for k in range(K)]          
    theta_TX = np.linspace(0,2*np.pi,L+1)
    polar_TX = [np.array([r1,theta]) for theta in theta_TX[:L] ]                  

    # Transform to cartesian coordinates
    x_RX = [polar_RX[k][0]*np.cos(polar_RX[k][1]) for k in range(K)]
    y_RX = [polar_RX[k][0]*np.sin(polar_RX[k][1]) for k in range(K)]
    cart_RX = np.array([x_RX,y_RX])
    x_TX = [polar_TX[l][0]*np.cos(polar_TX[l][1]) for l in range(L)]
    y_TX = [polar_TX[l][0]*np.sin(polar_TX[l][1]) for l in range(L)]
    cart_TX = np.array([x_TX,y_TX])

    # TX-RX distances (including 10m height difference)
    D = np.zeros((K,L))
    for l in range(L):
        for k in range(K):
            D[k,l]= np.sqrt(norm(cart_RX[:,k]-cart_TX[:,l])**2 + 10**2)
    
    # Channel gain (normalized by noise power)
    GdB = G_const - PL_exp*10*log10(D) - N0
    G = 10**(GdB/10)

    # Plot 
    if plot == True:
        plt.figure()
        plt.plot(x_TX,y_TX,'-o')
        plt.plot(x_RX,y_RX,'s')
        plt.axis('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.show()
    return G

def get_channel_realizations(G):
    H_list = []
    for _ in range(N_sim):
        H = np.zeros((K,L*N),dtype=complex)
        for l in range(L):
            H_iid = complex_normal(K,N)
            H[:,l*N:(l+1)*N] = diag(sqrt(G[:,l])) @ H_iid
        H_list.append(H)
    return H_list

def complex_normal(Ni,Nj):
    return 1/sqrt(2)*(np.random.standard_normal((Ni,Nj))+1j*np.random.standard_normal((Ni,Nj))) 

def herm(x):
    return x.conj().T

main()
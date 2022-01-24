# -*- coding: utf-8 -*-
"""
Code for "Team MMSE Precoding with Applications to Cell-free Massive MIMO"
Outupt: Comparison among local precoding schemes (Fig. 3)
Author: Lorenzo Miretti
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log2, log10, exp, absolute, diag, eye
from numpy.linalg import norm, inv, pinv

# Parameters
N_sim = 200          # Monte Carlo iterations for performance evaluation
N_cdf = 100          # Number of points for CDF 
L = 30               # number of TXs 
N = 1                # number of antennas per TX (do not change, Ricean model is not good for N>1)
K = 7                # number of single-antenna RXs                             
P = 100/K            # TX power [mW]
r_lim = (60,50)      # radius of service area [m] and radius of user distribution [m]
kappa =  0           # Ricean factor: h = sqrt(kappa/(1+kappa))*h_mean + sqrt(1/(1+kappa)))*h_ssf

def main():
    R_LTMMSE = []   
    R_LMMSE = [] 
    R_MRT = [] 
    R_OBE = []

    for n in range(N_cdf):
        # Draw random scenario (user positions) and compute path losses
        G = generate_scenario(plot = False)
        
        # Draw set of channel realizations for a given scenario
        H_list = get_channel_realizations(G)

        # Local TMMSE
        C_list = get_parameters_LTMMSE(H_list)
        _,R = simulate_scenario(H_list,LTMMSE,C_list)
        R_LTMMSE = R_LTMMSE + R 

        # Local MMSE with optimal large-scale fading coefficients
        C_list = get_parameters_LMMSE(H_list)
        _,R = simulate_scenario(H_list,LTMMSE,C_list)
        R_LMMSE = R_LMMSE + R

        # MRT
        _,R = simulate_scenario(H_list,MRT,None)
        R_MRT = R_MRT + R 

        # OBE
        C_list = get_parameters_OBE(H_list)
        _,R = simulate_scenario(H_list,OBE,C_list)
        R_OBE = R_OBE + R 

        print("Progress: %.1f%%" % (100 * (n+1)/N_cdf)) 

    # Plot CDF of achievable rates 
    fontsize = 16
    msize = 12
    lwidth = 4
    N_markers = 10
    marker_sep = round(K*N_cdf/N_markers)
    y_axis = np.arange(1,K*N_cdf+1)/(K*N_cdf)
    R_MRT.sort()
    plt.plot(R_MRT, y_axis,'-o', lw = lwidth, ms = msize, markevery=marker_sep, label='MRT')
    R_OBE.sort()
    plt.plot(R_OBE, y_axis,'-d', lw = lwidth, ms = msize, markevery=marker_sep, label='OBE')
    R_LMMSE.sort()
    plt.plot(R_LMMSE, y_axis,'-s', lw = lwidth, ms = msize, markevery=marker_sep, label='Local MMSE')
    R_LTMMSE.sort()
    plt.plot(R_LTMMSE, y_axis,'->', lw = lwidth, ms = msize, markevery=marker_sep,label='Local Team MMSE')
    plt.xlim(left=0)
    plt.ylim((0,1))
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xlabel('Rate [b/s/Hz]', fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)
    plt.savefig("cdf_local.pdf", bbox_inches = 'tight', pad_inches = 0)

    plt.show()

def MRT(H,parameters=None):
    """ Maximum ratio transmission
    """  
    return herm(H)

def OBE(H,C_list):
    """ Optimal bilinear equalizer
    """
    T = np.zeros((L*N,K),dtype = complex)
    for l in range(L):
        Hl = H[:,l*N:(l+1)*N]
        T[l*N:(l+1)*N,:] = herm(Hl) @ C_list[l]   
    return T

def get_parameters_OBE(H_list):
    """ Computation of the OBE parameters
        The parameters are here equivalently obtained by plugging t_k = H_l^\herm C_l e_k into the MSE objective in (5),
        and by minimizing over the deterministic column vector c_k = [c_{1,k} // ... // c_{L,k}] with c_{l,k} = C_le_k.
        It can by shown (e.g., by similar steps as in the Proof of Th. 1) that this is equivalent 
        to optimize the dual UL UatF bound as done in the original paper.  
    """
    # Estimate useful statistical quantities
    A = np.zeros((K*L,K*L),dtype=complex)
    B = np.zeros((K*L,K),dtype=complex)
    for n in range(N_sim): 
        H = H_list[n]
        Q = herm(H) @ H + eye(L*N)/P 
        H_diag = np.zeros((K*L,N*L),dtype=complex)
        for l in range(L):
            H_diag[l*K:(l+1)*K,l*N:(l+1)*N] = H[:,l*N:(l+1)*N]
        A += H_diag @ Q @ herm(H_diag) 
        B += H_diag @ herm(H)   
    A = A/N_sim
    B = B/N_sim
    # Compute optimal coefficients (each column is a c_k)
    C = inv(A) @ B 
    # Convert into list of statistical precoding stages
    C_list = []
    for l in range(L):
        C_l = C[l*K:(l+1)*K,:]
        C_list.append(C_l)
    return C_list  

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

def get_parameters_LMMSE(H_list):
    """ Computation of optimal large-scale fading coefficients for local MMSE precoding
    """
    # Useful statistical quantities
    A = [np.zeros((L,L), dtype=complex) for _ in range(K)]
    b = [np.zeros((L,1),dtype=complex) for _ in range(K)]
    P_TX = [np.zeros(L) for _ in range(K)]
    for n in range(N_sim):
        H = H_list[n]
        # LMMSE precoders without large-scale fading coefficients
        F = np.zeros((L*N,K),dtype = complex)
        for l in range(L):
            Hl = H[:,l*N:(l+1)*N]
            Fl = pinv(herm(Hl)@Hl + eye(N)/P) @ herm(Hl)
            F[l*N:(l+1)*N,:] = Fl 
        # Update statistical quantities
        for k in range(K):
            for i in range(K):
                h_eq = np.zeros((L,1),dtype=complex)
                for l in range(L):
                    h_eq[l] =  H[i,l*N:(l+1)*N] @ F[l*N:(l+1)*N,k]
                A[k] += h_eq @ herm(h_eq) / N_sim
                if i == k:
                    b[k] += h_eq / N_sim
            for l in range(L):
                P_TX[k][l] += norm(F[l*N:(l+1)*N,k])**2/N_sim
    # Compute optimal large-scale fading coefficients
    C_list = [np.zeros((K,K),dtype=complex) for _ in range(L)]
    for k in range(K):
        c = inv(A[k] + diag(P_TX[k])) @ b[k]
        for l in range(L):
            C_list[l][k,k] = c[l]             
    return C_list

def simulate_scenario(H_list,precoder,parameters):
    ''' Numerically evaluate the performance of a given precoding scheme, 
        using the MSE lower bound and the hardening bound
    ''' 
    # Estimate MSE, UL channel mean, and UL interference plus noise
    MSE = [0] * K
    mean = [0] * K
    interf_plus_noise = [0] * K
    for n in range(N_sim):
        H = H_list[n] 
        # Compute precoders
        T = precoder(H,parameters)
        # Update estimates for all users
        for k in range(K):     
            e =  np.zeros(K)  
            e[k] = 1
            MSE[k] += (norm(e-H@T[:,k])**2 + norm(T[:,k])**2/P)/N_sim
            mean[k] += H[k,:]@T[:,k] / N_sim
            interf_plus_noise[k] += (norm(H@T[:,k])**2 + norm(T[:,k])**2/P)/N_sim
    # Compute achievable rates using MSE lower bound
    R_MSE = [-log2(MSE[k]) for k in range(K)]
    # Compute DL achievable rates using UL UatF bound and UL-DL duality
    SINR_UL = [absolute(mean[k])**2/(interf_plus_noise[k]-absolute(mean[k])**2) for k in range(K)]
    R_hard = [log2(1+SINR_UL[k]) for k in range(K)]
    return R_MSE, R_hard  

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
            H_Rician = sqrt(kappa/(1+kappa)) + sqrt(1/(1+kappa))*H_iid
            H[:,l*N:(l+1)*N] = diag(sqrt(G[:,l])) @ H_Rician
        H_list.append(H)
    return H_list

def complex_normal(Ni,Nj):
    return 1/sqrt(2)*(np.random.standard_normal((Ni,Nj))+1j*np.random.standard_normal((Ni,Nj))) 

def herm(x):
    return x.conj().T

main()
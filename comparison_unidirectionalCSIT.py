# -*- coding: utf-8 -*-
"""
Code for "Team MMSE Precoding with Applications to Cell-free Massive MIMO"
Output: Comparison between unidirectional TMMSE precoding and the SGD scheme (Fig. 4)
Author: Lorenzo Miretti
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log2, log10, absolute, diag, eye
from numpy.linalg import norm, pinv

# Parameters
N_sim = 100          # MonteCarlo iterations for performance evaluation
N_pts = 10           # Number of points in plot
N_linsearch = 30     # Number of points for linear search (SGD algorithm)
L = 30               # number of TXs 
N = 1                # number of antennas per TX (do not change, the SGD algorithm requires N=1)
K = 7                # number of single-antenna RXs    
eps = 0.2            # std. deviation CSIT error  
r_lim = (60,0)       # radius of service area [m] and radius of user distribution [m]
                           
def main():
    R_UTMMSE = []   
    R_SGD = []    
    R_SGD_robust = []

    # Focus on a fixed scenario (position of RXs)
    np.random.seed(0)   

    # Draw scenario 
    G = generate_scenario(plot = False)

    # Compute CSIT estimation error variance (corresponds to the error covariance \Sigma_l in the paper)
    sig2 = eps* np.sum(G,axis=0)

    # Draw set of channel realizations for a given scenario
    H_list = get_channel_realizations((1-eps)*G)                # Channel estimates
    E_list = get_channel_realizations(eps*G)                    # CSIT errors
    H_list_true = [H_hat + E for (H_hat,E) in zip(H_list,E_list)]           # True channel realizations

    # Simulate over N_pts SNR levels at the first RX
    k = 0
    SNRdB = np.linspace(-5,40,N_pts) 
    for n in range(N_pts):
        # Get nominal SNR parameter P
        SNR = 10**(SNRdB[n]/10)  
        P = SNR/np.sum(G[k,:])  

        # Unidirectional TMMSE
        Pi = get_parameters_UTMMSE(P,sig2,H_list)
        _,R_hard = simulate_scenario(P,H_list,H_list_true,UTMMSE,(Pi,sig2))
        R_UTMMSE.append(R_hard[k]) 

        # SGD 
        mu = np.ones(K,dtype=complex)
        _,R_hard = simulate_scenario(P,H_list,H_list_true,SGD,mu)
        R_SGD.append(R_hard[k]) 

        # SGD (with linear search for tuning mu)
        x = np.linspace(0.01,2,N_linsearch)
        R_max = 0
        for i in range(N_linsearch):
            mu = np.ones(K,dtype=complex)*x[i]
            _,R_hard = simulate_scenario(P,H_list,H_list_true,SGD,mu)
            if R_hard[k] > R_max:
                R_max = R_hard[k]
        R_SGD_robust.append(R_max) 

        print("Progress: %.1f%%" % (100 * (n+1)/N_pts))

    # Plot achievable rates vs SNR
    fontsize = 16
    msize = 12
    lwidth = 4
    plt.plot(SNRdB,R_UTMMSE,'-o', lw = lwidth, ms = msize, label='Unidirectional Team MMSE')
    plt.plot(SNRdB,R_SGD,'-s', lw = lwidth, ms = msize, label='SGD')
    plt.plot(SNRdB,R_SGD_robust,'->', lw = lwidth, ms = msize, label='Robust SGD')
    plt.ylim(bottom=0)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.ylabel('Rate [b/s/Hz]', fontsize=fontsize)
    plt.xlabel('SNR [dB]',fontsize=fontsize)
    plt.savefig("comparison_SGD_realistic.pdf", bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def SGD(P,H,mu):
    T = np.zeros((L*N,K),dtype = complex)  # initialization 
    for l in range(L): 
        H_l = H[:,N*l:N*(l+1)]
        T[N*l:N*(l+1),:] = pinv(herm(H_l)@H_l) @ herm(H_l) @ (np.eye(K,dtype=complex)-H @ T) @ diag(mu)
    return T 

def UTMMSE(P,H,params):
    """ Unidirectional TMMSE precoding, wrapper function for recursive computation
    """
    # List of precoders
    T = [np.zeros((N,K),dtype=complex)] * L
    # Compute precoders recursively
    unidirectionalTMMSE(P,H,params,T,0,eye(K,dtype=complex))
    # Transform list in matrix
    T = np.vstack(T)
    return T

def unidirectionalTMMSE(P,H,params,T,l,S_bar):
    """ Recursive routine for unidirectional TMMSE precoding, 
        involving an information matrix V_bar sequentially updated and forwarded from TX 1 to TX L,
        and statistical information given by Pi
    """
    Pi = params[0]
    sig2 = params[1]
    # Compute precoder using local channel measurements and S_bar
    Hl = H[:,l*N:(l+1)*N]
    Fl = pinv(herm(Hl)@Hl + eye(N)/P + sig2[l]) @ herm(Hl)
    Pl = Hl @ Fl
    Vl = pinv(eye(K)-Pi[l] @ Pl) @ (eye(K)-Pi[l])
    T[l] = Fl @ Vl @ S_bar
    if l < L-1:  # recursive call, sending updated S_bar
        Vl_bar = eye(K) - Pl @ Vl
        unidirectionalTMMSE(P,H,(Pi,sig2),T,l+1,Vl_bar @ S_bar)

def get_parameters_UTMMSE(P,sig2,H_list):
    """ Monte Carlo estimation of statistical parameters for unidirectional TMMSE precoding
        Iterative implementation, could have been implemented recursively similarly to the unidirectionalTMMSE routine
    """
    Pi = [np.zeros((K,K),dtype=complex)] * L
    # Estimate auxiliary statistical quantities
    for l in range(L-1,0,-1):
        E_PS = np.zeros((K,K),dtype=complex)
        E_Sbar = np.zeros((K,K),dtype = complex)
        for n in range(N_sim): 
            H = H_list[n] 
            Hl = H[:,l*N:(l+1)*N]
            Pl = Hl @ pinv(herm(Hl)@Hl + eye(N)/P+sig2[l]) @ herm(Hl)
            Vl = pinv(eye(K)-Pi[l] @ Pl) @ (eye(K)-Pi[l])
            Vl_bar = pinv(eye(K)- Pl @ Pi[l]) @ (eye(K)-Pl)
            E_PS += Pl @ Vl / N_sim
            E_Sbar += Vl_bar / N_sim
        Pi[l-1] = E_PS + Pi[l] @ E_Sbar
    return Pi

def simulate_scenario(P,H_list,H_list_true,precoder,parameters):
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
        T = precoder(P,H,parameters)
        # Update estimates for all users
        H = H_list_true[n]
        for k in range(K):     
            e =  np.zeros(K)  
            e[k] = 1
            MSE[k] += (norm(e-H@T[:,k])**2 + norm(T[:,k])**2/P)/N_sim
            mean[k] += H[k,:]@T[:,k] / N_sim
            interf_plus_noise[k] += (norm(H@T[:,k])**2 + norm(T[:,k])**2/P)/N_sim
    # Compute achievable rates using MSE lower bound
    R_MSE = [-log2(MSE[k]) for k in range(K)]
    # Compute DL achievable rates using UL UatF bound and UL/DL duality
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
            H[:,l*N:(l+1)*N] = diag(sqrt(G[:,l])) @ H_iid
        H_list.append(H)
    return H_list

def complex_normal(Ni,Nj):
    return 1/sqrt(2)*(np.random.standard_normal((Ni,Nj))+1j*np.random.standard_normal((Ni,Nj))) 

def herm(x):
    return x.conj().T

main()
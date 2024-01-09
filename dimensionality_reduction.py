import numpy as np
import matplotlib.pyplot as plt

from parameters import N_sim, N, K, L

def pca(H_data: np.ndarray, plot=True, xtick_stagger=10, esp_string='') -> tuple[np.ndarray]:
    #center data and compute covariance
    H_cen =  H_data - np.mean(H_data, axis=0)
    H = H_cen.T@H_cen

    #compute singular values and transform
    _, S, V = np.linalg.svd(H, full_matrices=True)

    plot_title = f'PCA Scree Plot {esp_string}'

    #generate scree plot
    scree =  S**2/np.sum(S**2)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(S) + 1), scree, marker='o', linestyle='-')
    plt.title(plot_title)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(np.arange(1, len(S) + 1, xtick_stagger))
    plt.grid(True)  

    if plot:
        plt.show()
    
    plt.savefig(f'./plots/{plot_title}', dpi=1000)

    return H_cen, V

def point_plot(H_cen: np.ndarray, V: np.ndarray) -> None:
    #plot first 2 dims
    Vl = V[0:2,:] # V = W^H so take first L rows as theyre the eigenvectors I need
    H_2 = H_cen@Vl.T
    print(Vl.shape)
    print(H_2.shape)
    plt.figure(figsize=(8, 6))
    plt.scatter(H_2[:, 0], H_2[:, 1], c='blue', alpha=0.5)
    plt.title('2D Point Cloud of Transformed Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def reformat_measurements_local_csi(i: int, Y_data: np.ndarray) -> np.ndarray:
    Yi = Y_data[:,N*i:N*(i+1)]
    temp =  []
    for n in range(N_sim):
        temp.append(Yi[K*n:K*(n+1),:].flatten(order='C'))
    return np.vstack(temp)

def save_truncated_data(i: int, l: int, X: np.ndarray, V: np.ndarray) -> None:
    Vl = V[0:l,:] 
    Xl = X@Vl.T #X needs to be centered!
    np.savetxt(f'./agents/agent-{i}-data.txt', Xl, delimiter=',')

def main():
    #load data
    Y_data = np.loadtxt('agent-measurement-data.txt', delimiter=',', dtype=complex)
    H_data = np.loadtxt('channel-data.txt', delimiter=',', dtype=complex)

    #PCA of channel
    _, _ = pca(H_data, plot=True, xtick_stagger=10, esp_string="For Channel Measurements")

    #PCA of local measurements
    #looking at scree plots this seems reasonable
    l = 5

    for i in range(L):
        Y = reformat_measurements_local_csi(i, Y_data)
        X, V = pca(Y, plot=False, xtick_stagger=1, esp_string=f'For Measurements of Agent {i}')
        save_truncated_data(i, l, X, V)

main()
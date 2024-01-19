from comparison_CSITsharingpatterns import generate_scenario, get_channel_realizations
from parameters import N_cdf, N_sim, N_sim_t, L, N, K, P, r_lim

import numpy as np

def generate_channel_data(N_sim):
    G = generate_scenario(plot = False)
    return get_channel_realizations(G, N_sim)
    
#for pca
def vectorize_channel_data(H_list, filepath):
    H_vectorized = []
    for h in H_list:
        H_vectorized.append(h.flatten(order='C'))

    H = np.vstack(H_vectorized)
    print("shape of data matrix: ", H.shape)

    np.savetxt(filepath, H, delimiter=',')

def local_information_structure(H_list, filepath):
    Y = np.vstack(H_list)
    np.savetxt(filepath, Y, delimiter=',')


def main():
    H_list_train, H_list_test = generate_channel_data(N_sim), generate_channel_data(N_sim_t)
    vectorize_channel_data(H_list_train, 'channel-data-observation.txt')
    local_information_structure(H_list_train, 'agent-measurement-data-observation.txt')
    vectorize_channel_data(H_list_test, 'channel-data-prediction.txt')
    local_information_structure(H_list_test, 'agent-measurement-data-prediction.txt')
    
main()  

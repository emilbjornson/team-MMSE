# Team MMSE Precoding with Applications to Cell-free Massive MIMO

This is a code package related to the following scientific article:

Lorenzo Miretti, Emil Björnson, David Gesbert, “Team MMSE Precoding with Applications to Cell-free Massive MIMO,” IEEE Transactions on Wireless Communications, to appear, 2022.

The package contains a simulation environment that reproduces the numerical results in the article. _We encourage you to also perform reproducible research!_

## Abstract of the article

This article studies a novel distributed precoding design, coined _team minimum mean-square error_ (TMMSE) precoding, which rigorously generalizes classical centralized MMSE precoding to distributed operations based on transmitter-specific channel state information (CSIT). Building on the so-called _theory of teams_, we derive a set of necessary and sufficient conditions for optimal TMMSE precoding, in the form of an infinite dimensional linear system of equations. These optimality conditions are further specialized to cell-free massive MIMO networks, and explicitly solved for two important examples, i.e., the classical case of local CSIT and the case of  unidirectional CSIT sharing along a serial fronthaul. The latter case is relevant, e.g., for the recently proposed _radio stripe_ concept and the related advances on sequential processing exploiting serial connections. In both cases, our optimal design outperforms the heuristic methods that are known from the previous literature. Duality arguments and numerical simulations validate the effectiveness of the proposed team theoretical approach in terms of ergodic achievable rates under a sum-power constraint. 

## Content of Code Package

The article contains 3 simulation figures, numbered 2-4. Figure 3 is composed by 2 subfigures, labelled a-b; Figure 4 is composed by 3 subfigures, labelled a-c.

Figure 2 is generated by the Python script 
> comparison_CSITsharingpatterns.py

Figure 3 is generated by the Python script 
> comparison_localCSIT.py

by setting: kappa = 0 for Fig. 3a; kappa = 1 for Fig. 3b.

Figure 4 is generated by the Python script
> comparison_unidirectionalCSIT.py

by setting: r_lim = (60,0) and eps = 0 for Fig. 4a; r_min = (60,0) and eps = 0.2 for Fig. 4b; r_lim = (60,50) and eps = 0 for Fig. 4c.

Please refer to each file for additional details. 

Warning: The name of some variables may differ from the notation adopted in the paper. 

## Acknowledgements

This work received partial support from the Huawei funded Chair on Future Wireless Networks at EURECOM, and by the Grant 2019-05068 from the Swedish Research Council.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
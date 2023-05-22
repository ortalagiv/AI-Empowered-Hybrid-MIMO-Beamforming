# AI-Empowered Hybrid MIMO Beamforming

Code repository for the article:

N. Shlezinger, M. Ma, O. Lavi, N. T. Nguyen, Y. C. Eldar, M. Juntti, "AI-Empowered Hybrid MIMO Beamforming," available at https://arxiv.org/abs/2303.01723.

Please cite our paper if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [Manifold Optimization](#manifold-optimization)
  * [Alternating Optimization](#alternating-optimization)
  * [Blackbox CNN](#blackbox-cnn)
  * [ManNet](#mannet)
  * [Unfolded PGA](#unfolded-pga)


# Introduction
In the article, we offer a comprehensive tutorial to AI-aided techniques for hybrid MIMO beamforming. 
Our emphasis is primarily on the signal processing algorithms required for such tasks.
We aim to provide insights without being limited to a particular implementation.

In our discussion, we outline different approaches for designing hybrid beamforming, categorizing them into three primary families: 
* Optimization-based methods that utilize iterative optimizers to determine the beampatterns (Riemannian manifold optimizer of [1] and the alternating optimizer of [2]). 
* DNN-based schemes that employ pre-trained DNNs to map CSI to hybrid configurations (CNN based architecture of [3]). 
* Deep-unfolded designs that leverage deep learning techniques to aid in the iterative optimization process (ManNet model of [4] and the unfolded PGA of [5]).

###### [1] X. Yu, J.-C. Shen, J. Zhang, and K. B. Letaief, “Alternating minimization algorithms for hybrid precoding in millimeter wave MIMO systems,” IEEE J. Sel. Topics Signal Process., vol. 10, no. 3, pp. 485–500, 2016.  
###### [2] F. Sohrabi and W. Yu, “Hybrid digital and analog beamforming design for large-scale antenna arrays,” IEEE J. Sel. Topics Signal Process., vol. 10, no. 3, 2016.  
###### [3] A. M. Elbir and A. K. Papazafeiropoulos, “Hybrid precoding for multiuser millimeter wave massive MIMO systems: A deep learning approach,” IEEE Trans. Veh. Technol., vol. 69, no. 1, pp. 552–563, 2019.  
###### [4] N. T. Nguyen, M. Ma, N. Shlezinger, Y. C. Eldar, A. L. Swindlehurst, and M. Juntti, “Deep unfolding hybrid beamforming designs for THz massive MIMO systems,” arXiv preprint arXiv:2302.12041, 2023.  
###### [5] O. Agiv and N. Shlezinger, “Learn to rapidly optimize hybrid precoding,” in Proc. IEEE SPAWC, 2022.

# Folders Structure
### Manifold Optimization
Implementation of Riemannian manifold optimizer of [1].  
The code is implented in MATLAB. To execute run the 'main.m' in the folder *manifold*.

### Alternating Optimization
Implementation of the alternating optimizer of [2].  
The code is implented in MATLAB. To execute run the 'main.m' in the folder *alternating*.

### Blackbox CNN
Implementation of the CNN based architecture of [3].  
The code is implented in Python. To execute run the 'BB_main.py' in the folder *BB_CNN*.

### ManNet
Implementation of the ManNet model of [4].  
The code is implented in Python. To execute run the 'main.py' in the folder *ManNet*.

### Unfolded PGA
Implementation of the unfolded PGA of [5].  
The code is implented in Python. To execute run the 'main.py' in the folder *unfolded_PGA*.

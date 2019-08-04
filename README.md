# Enhanced Mode Clustering
Python3 Implementation of the Enhanced Mode Clustering

- Paper reference: Chen, Yen-Chi, Christopher R. Genovese, and Larry Wasserman. "[A comprehensive approach to mode clustering](https://projecteuclid.org/euclid.ejs/1455715961)." Electronic Journal of Statistics 10.1 (2016): 210-241.
- We provide a Python3 implementation for all the main functions of the enhanced mode clustering. For the original R implementation, please check [https://github.com/yenchic/EMC](https://github.com/yenchic/EMC).
- **EMC_fun.py**: the script that contains all the main functions for the mean shift algorithm and the enhanced mode clustering (EMC), including a functional implementation of the EMC.
- **EMC_class.py**: the script that contains the implementation of a Python class `EMC`.
- **Examples.py**: the script that provides some examples of applying the enhanced mode clustering.

### Requirements
- Python >= 3.6 (Earlier version might be applicable.)
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/index.html) (Used for capturing the connectivity of points by creating a K-nearest neighbors graph and the subsequent hierarchical clustering in the fast mean shift algorithm. Also, used for MDS.), [SciPy](https://www.scipy.org/) (Only the function [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html) is envoked to compute the pairwise distance between points in the soft mode clustering.), [collections](https://docs.python.org/3.6/library/collections.html) (The `Counter` function is used).

### Descriptions
Our implementation follows the flowchart suggested by Chen et al. (2016)
<p align="center">
<img src="https://github.com/zhangyk8/EMC_Python3/blob/master/figures/flowchart.png" style="zoom:60%" />
 <B>Fig 1. </B>A flowchart for the mode clustering analysis (cited from Chen et al., 2016) 
 </p>
 The section numberings in this figure (FIG 7 in the original paper) are not consistent with the paper content. Mode Clustering is discussed in Section 2 and "5"; Denoising is introduced in Section "6"; Visualization comes at Section "7". Except for these small typos, the structure of the flowchart is clear.

#### Mode Clustering
We implement two versions of the mean shift algorithm to capture the local modes in any point dataset. For the standard version, we follow the derivation in Comaniciu and Meer (2002). However, our implementation generalizes their formulas and allows the diagonal bandwidth matrix to have non-identical entries. Specifically, the density estimator can be rewritten as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{f}_{K}(\mathbf{x})=\frac{c_{K,d}}{n\det(H)}\sum_{i=1}^nk(||H^{-1}(\mathbf{x}-\mathbf{x}_i)||^2)," />

where 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;c_{K,d}" /> is a normalized constant, <img src="https://latex.codecogs.com/svg.latex?\Large&space;H=diag(h_1,...,h_d)" />, and <img src="https://latex.codecogs.com/svg.latex?\Large&space;k" /> is the _profile_ of a radially symmetric kernel, such as <img src="https://latex.codecogs.com/svg.latex?\Large&space;k(x)=\exp(-x/2)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for the multivariate gaussian kernel, <img src="https://latex.codecogs.com/svg.latex?\Large&space;k(x)=(1-x)\cdot\mathbf{1}_{[0,1]}(x)" /> for the Epanechnikov kernel, etc.

The gradient of the density estimator can be obtained as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla\hat{f}_{K}(\mathbf{x})=\frac{2c_{K,d}}{n\det(H)}\sum_{i=1}^nH^{-1}(\mathbf{x}_i-\mathbf{x})g(||H^{-1}(\mathbf{x}-\mathbf{x}_i)||^2)\\=\frac{2c_{K,d}}{n\det(H)}\left[\sum_{i=1}^ng(||H^{-1}(\mathbf{x}-\mathbf{x}_i||^2)\right]\left[\frac{\sum_{i=1}^nH^{-1}\mathbf{x}_ig(||H^{-1}(\mathbf{x}-\mathbf{x}_i||^2)}{\sum_{i=1}^ng(||H^{-1}(\mathbf{x}-\mathbf{x}_i||^2)}-H^{-1}\mathbf{x}\right]," />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)=-k'(x)" />.

Therefore, the mean shift iteration (a special case of the fixed-point iteration) becomes

<img src="https://latex.codecogs.com/svg.latex?\Large&space;y_{j+1}=\frac{\sum_{i=1}^n\mathbf{x}_ig(||H^{-1}(\mathbf{y}_j-\mathbf{x}_i||^2)}{\sum_{i=1}^ng(||H^{-1}(\mathbf{y}_j-\mathbf{x}_i||^2)}\quad\j=1,2,..." />

The implementation of the mean shift algorithm is encapsulated into a Python function called `Mean_Shift`.

`def Mean_Shift(data, query, h=None, kernel="gaussian", max_iter=1000, eps=1e-8, D_kernel=None)`
- Inputs:
  - data: Input data matrix (An m*d array of m data points in an d-dimensional space).
  - query: The mesh points to which we apply the mean shift algorithm (An k*d array of k mesh points in an d-dimensional space).
  - h: Smoothing parameter (Bandwidth for KDE)
  - kernel: The kernel name for KDE. ('gaussian', 'epanechnikov', or others) If others, please define your own kernel function and its first derivative. Specify the derivative function in the parameter 'D_kernel'.
  - max_iter: Maximal number of iteration for mean shift.
  - eps: The tolerance (stopping criterion) for mean shift iteration. 
  - D_kernel: The first derivative of the user-specified kernel function.
- Output:
  - A numpy array for mesh points after mean shift iteration (Same data format as 'query')
  
Usage:
```bash
import numpy as np
import matplotlib.pyplot as plt
from EMC_fun import Mean_Shift

N1 = 200
mean1 = [0, 3]
cov1 = 0.2*np.identity(2)
com1 = np.random.multivariate_normal(mean1, cov1, N1)

N2 = 300
mean2 = [3, 0]
cov2 = 0.3*np.identity(2)
com2 = np.random.multivariate_normal(mean2, cov2, N2)

N3 = 400
mean3 = [-3, -1]
cov3 = 0.5*np.identity(2)
com3 = np.random.multivariate_normal(mean3, cov3, N3)

label = np.repeat([0, 1, 2], [N1, N2, N3])

# Concatenate Gaussian components into the final simulated data set
small_D = np.concatenate((com1, com2, com3), axis=0)

# Mean-shift algorithm
small_m = Mean_Shift(data=small_D, query=small_D, h=None, kernel="gaussian", max_iter=1000, eps=1e-10)
plt.scatter(small_D[:,0], small_D[:,1], c=label, s=10)
plt.plot(small_m[:,0], small_m[:,1], "ro")
```

#### Fast Mean Shift Algorithm


### Examples


### Additional Reference
- Dorin Comaniciu and Peter Meer, "Mean shift: a robust approach toward feature space analysis," in _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol.24, no.5, pp.603-619, May 2002.

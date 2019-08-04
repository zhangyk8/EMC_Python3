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
We implement two versions of the mean shift algorithm to capture the local modes in any point dataset. For the standard version, we follow the derivation in Comaniciu and Meer (2002). However, our implementation generalizes their formulas and allows the diagonal bandwidth matrix to embrace non-identical entries. Specifically, the density estimator can be rewritten as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{f}_{K}(\mathbf{x})=\frac{c_{K,d}}{n\det(H)}\sum_{i=1}^nk(||H^{-1}(\mathbf{x}-\mathbf{x}_i)||^2)," title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

where 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;c_{K,d}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is a normalized constant, <img src="https://latex.codecogs.com/svg.latex?\Large&space;H=diag(h_1,...,h_d)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />, and <img src="https://latex.codecogs.com/svg.latex?\Large&space;k" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is the _profile_ of a radially symmetric kernel, such as <img src="https://latex.codecogs.com/svg.latex?\Large&space;k(x)=\exp(-x/2)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for the multivariate gaussian kernel, <img src="https://latex.codecogs.com/svg.latex?\Large&space;k(x)=(1-x)\cdot\mathbf{1}_{[0,1]}(x)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for the Epanechnikov kernel.

### Additional Reference
- Dorin Comaniciu and Peter Meer, "Mean shift: a robust approach toward feature space analysis," in _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol.24, no.5, pp.603-619, May 2002.

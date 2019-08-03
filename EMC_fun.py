"""
@author: yikun zhang

Last edit: August 3, 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.manifold import MDS
from collections import Counter

'''
A helper function for generating high dimensional five clusters dataset (Example 8.1 in Chen et al., 2016)
@ Parameters:
    N_c -- Sample size for each cluster.
    N_f -- Sample size for each filament.
    dis_c -- Standard deviation for generating each cluster point.
    dis_f -- Standard deviation for generating each filament point.
    d_add -- The added dimensions.
@ Return:
    A numpy array (dataset) with shape (n, 3+d_add), where (n=5*N_c+4*N_f) is the number of data points and (3+d_add) is the dimension.
'''
def Five_Cluster(N_c=200, N_f=100, dis_c=0.01, dis_f=0.005, d_add=7):
    # Setting up the means of clusters
    mean = np.zeros((5, 3+d_add))
    data = np.empty([N_c*5+N_f*4, 3+d_add])
    
    mean[1,0] = 0.1
    mean[2,1] = 0.1
    mean[3,2] = 0.1
    mean[4,1] = 0.1
    mean[4,2] = 0.1
    
    for i in range(5):
        data[(i*N_c):((i+1)*N_c),:] = np.random.multivariate_normal(mean[i,:], (dis_c)**2*np.identity(3+d_add), N_c)
        
    for i in range(1,4):
        data[(5*N_c+(i-1)*N_f):(5*N_c+i*N_f),:] = np.multiply(mean[i,:]-mean[0,:], np.random.rand(N_f, 1)) + np.random.multivariate_normal(np.repeat(0, 3+d_add), (dis_f)**2*np.identity(3+d_add), N_f) + mean[0,:]
    data[(5*N_c+3*N_f):,:] = np.multiply(mean[4,:]-mean[3,:], np.random.rand(N_f, 1)) + np.random.multivariate_normal(np.repeat(0, 3+d_add), (dis_f)**2*np.identity(3+d_add), N_f) + mean[3,:]
    return data

'''
## Usage:
# Generate 5-clusters in 10-Dim
sim_data3 = Five_Cluster(N_c=200, N_f=100, dis_c=0.01, dis_f=0.005, d_add=7)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(sim_data3[:,0], sim_data3[:,1], sim_data3[:,2], s=1)
'''


'''
Mean shift algorithm using Python3
@ Parameter:
    data -- Input data matrix (An m*d array of m data points in an d-dimensional space).
    query -- The mesh points to which we apply the mean shift algorithm (An k*d array of k mesh points in an d-dimensional space).
    h -- Smoothing parameter (Bandwidth for KDE)
    kernel -- The kernel name for KDE. ('gaussian', 'epanechnikov', or others) If others, please define your own kernel function and its first derivative. Specify the derivative function in the parameter 'D_kernel'.
    max_iter -- Maximal number of iteration for mean shift.
    eps -- The tolerance (stopping criterion) for mean shift iteration. 
    D_kernel: The first derivative of the user-specified kernel function.
@ Return:
    A numpy array for mesh points after mean shift iteration (Same data format as 'query')
'''
def Mean_Shift(data, query, h=None, kernel="gaussian", max_iter=1000, eps=1e-8, D_kernel=None):
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth for multivariate KDE 
        # (Only works for Gaussian kernel)
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(data, axis=0)
    print("The current bandwidth is "+ str(h) + ".\n")
    
    # The derivative function for the kernel
    if kernel == "gaussian":
        # Gaussian profile
        def f(x):
            return np.exp(-x/2)
    elif kernel == "epanechnikov":
        # Epanechnikov profile
        def f(x):
            return 1 - (x>1)
    else:
        f = D_kernel
    assert (f is not None), "The kernel function must be pre-defined."
    
    modes = np.copy(query)
    for i in range(query.shape[0]):
        iter_old = np.repeat(100.0, d)
        iter_new = modes[i,:]
        iter_T = 0
        while iter_T < max_iter and np.linalg.norm(iter_old - iter_new) > eps:
            iter_old = np.copy(iter_new)
            norm_sq_k = f(np.sum(((iter_old - data)/h)**2, axis=1))
            # Mean-shift iteration
            iter_new = np.sum(data*np.reshape(norm_sq_k, (len(norm_sq_k), 1)), axis=0)/sum(norm_sq_k)
            iter_T += 1
        
        modes[i,:] = iter_new
    
    return modes

'''
## Usage:
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
'''


'''
Fast mean shift algorithm using hierarchical clustering
@ Parameter:
    data -- Input data matrix (An m*d array of m data points in an d-dimensional space).
    query -- The mesh points to which we apply the mean shift algorithm (An k*d array of k mesh points in an d-dimensional space).
    h -- Smoothing parameter (Bandwidth for KDE).
    cut -- The cut for hierarchical clustering (The distance threshold above which, clusters will not be merged).
    K -- The number of nearest neighbors for creating a graph that captures the connectivity.
    kernel -- The kernel name for KDE. ('gaussian', 'epanechnikov', or others) If others, please define your own kernel function and its first derivative. Specify the derivative function in the parameter 'D_kernel'.
    max_iter -- Maximal number of iteration for mean shift.
    eps -- The tolerance (stopping criterion) for mean shift iteration. 
    D_kernel: The first derivative of the user-specified kernel function.
@ Return:
    A numpy array for mesh points after mean shift iteration (Same data format as 'query')
'''
def Fast_Mean_Shift(data, query, h=None, cut=0.1, K=10, kernel="gaussian", max_iter=100, eps=1e-8, D_kernel=None):
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth for multivariate KDE 
        # (Only works for Gaussian kernel)
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(data, axis=0)
    
    # Run the standard mean-shift algorithm with a few iterations
    modes = Mean_Shift(data=data, query=query, h=h, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
    # Define the structure of the data based on k-nearest-neighbors
    modes_con = kneighbors_graph(modes, n_neighbors=K, include_self=False)
    # Run the hierarchical clustering to merge those modes produced by early-stopped mean-shift algorithm
    h_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=np.mean(h)*cut, connectivity=modes_con, linkage='complete')
    h_cluster.fit_predict(modes)
    labels = h_cluster.labels_
    
    # Replace modes with the centers of their affiliated clusters
    for i in np.unique(labels):
        modes[labels == i,:] = np.mean(modes[labels == i,:], axis=0)
        
    return modes

'''
## Usage:
small2 = Fast_Mean_Shift(data=small_D, query=small_D, h=None, cut=0.1, K=10, kernel="gaussian", max_iter=20, eps=1e-7)
'''


'''
A helper function: Group the output mesh points from mean shift algorithm into distinct modes and output mode clustering labels.
@ Parameter:
    modes -- The output mesh points from Mean_Shift() or Fast_Mean_Shift() (A numpy array)
    tol -- Tolerance for pairwise distances between mesh points (Any pair of mesh points with distance less than this value will be grouped into the same cluster).
@ Return: 
    1) A numpy array with the coordinates of distinct modes; 
    2) A 1-dim array with integer labels and length N for specifying the affiliation of each mesh point (N is the number of rows in 'modes').
'''
def Unique_Modes(modes, tol=1e-4):
    n_modes = modes.shape[0]   ## The number of modes
    modes_ind = [0]   ## Candidate list of unique modes
    labels = np.empty([n_modes, ], dtype=int)
    labels[0] = 0
    curr_lb = 0   ## The current label indicator
    
    for i in range(1, n_modes):
        flag = None   ## Indicate whether index i should be added to the candidate list of unique modes
        for j in modes_ind:
            if np.sqrt(sum((modes[i,:] - modes[j,:])**2)) <= tol:
                flag = labels[j]  # The mode has been existing
        if flag is None:
            curr_lb += 1
            modes_ind.append(i)
            labels[i] = curr_lb
        else:
            labels[i] = flag
    
    return modes[modes_ind,:], labels

'''
## Usage:
uni_small_m, small_ml = Unique_Modes(modes=small_m, tol=0.0001)
'''


'''
Soft mode clustering: Absorbing probability for diffusion process starting at each data point
@ Parameter:
    data -- Input data matrix (An n*d array of n data points in an d-dimensional space).
    modes -- Distinct modes matrix (An m*d array of m distinct modes in an d-dimensional space).
    h -- Smoothing parameter (Bandwidth for KDE).
    kernel -- The kernel name for KDE. ('gaussian', 'epanechnikov', or others) If others, please define your own kernel function. Specify the kernel function in the parameter 'kernel_fun'.
    kernel_fun -- The user-specified kernel function.
@ Return:
    An array about the absorbing probabaility from each data point to each local modes.
'''
def Soft_Mode_Clustering(data, modes, h=None, kernel="gaussian", kernel_fun=None):
    # Concatenate the coordinates of estimated modes and original data points
    X = np.concatenate((modes, data), axis=0)
    d = X.shape[1]   ## Dimension of the data
    n = X.shape[0]   ## The total number of states
    n_modes = modes.shape[0]   ## The number of modes (absorbing states)
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth for multivariate KDE
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(X, axis=0)
    print("The current bandwidth is "+ str(h) + ".\n")
    
    # The derivative function for the kernel
    if kernel == "gaussian":
        # Gaussian profile
        def f(x):
            return np.exp(-x/2)
    elif kernel == "epanechnikov":
        # Epanechnikov profile
        def f(x):
            return (1-x)*(x<=1)
    else:
        f = kernel_fun
    assert (f is not None), "The kernel function must be pre-defined."
    
    kernel_dist = f(squareform(pdist(X/h))**2)   ## Unnormalized pairwise kernel distances
    P_lower = (kernel_dist.T/np.sum(kernel_dist, axis=1)).T
    S = P_lower[n_modes:, :n_modes]
    T = P_lower[n_modes:, n_modes:]
    A_hat = np.dot(np.linalg.inv(np.identity(n-n_modes)-T), S)  ## The absorbing probability matrix
    return A_hat

'''
## Usage:
small_soft = Soft_Mode_Clustering(data=small_D, modes=uni_small_m)
'''


'''
Measuring the connectivity between cluster
@ Parameter:
    soft_modes -- The absorbing probability matrix outputed from Soft_Mode_Clustering().
    modes_label -- A 1-dim array for specifying the affiliation of each mesh point.
@ Return:
    The array for connectivities between clusters
'''
def Cluster_Connectivity(soft_modes, modes_label):
    n_modes = soft_modes.shape[1]   ## The number of unique modes
    n = soft_modes.shape[0]    ## The number of data points
    con_mat = np.empty([n_modes, n_modes])
    # Convert the label array into integer type
    m_labels, unique = pd.factorize(modes_label, sort=True)
    
    for i in range(n_modes):
        con_mat[i,:] = np.dot(np.reshape(m_labels == i, (1, n)), soft_modes)/np.sum(m_labels == i)
    con_mat = (con_mat + con_mat.T)/2
    
    return con_mat

'''
## Usage:
small_con = Cluster_Connectivity(soft_modes=small_soft, modes_label=small_ml)
'''


'''
Visualization helper function for mode clustering
@ Parameter:
    data -- Input data matrix (An n*d array of n data points in an d-dimensional space).
    modes -- Distinct modes matrix (An m*d array of m distinct modes in an d-dimensional space).
    modes_label -- A 1-dim array for specifying the affiliation of each data point.
    rho -- A float number specifying the contrast factor for preventing overlapping clusters in the visualization.
@ Return:
    1) The embedding coordinates for data points;
    2) The embedding coordinates for local modes.
'''
def Vis_Cluster(data, modes, modes_label, rho=None):
    assert data.shape[0] == len(modes_label), "The number of rows in the data array must be the same as the length of the 'modes_label' array!"
    assert modes.shape[0] == len(np.unique(modes_label)), "The number of rows in the mode array must be the same as the number of distinct labels in the 'modes_label' array!"
    d = data.shape[1]   ## The dimension of the data
    
    data_emb = np.empty([data.shape[0], 2])
    m_r0 = np.empty([modes.shape[0], ])   ## For computing the default rho
    ## Stage 1: MDS on modes
    embedding = MDS(n_components=2, n_init=7)
    modes_emb = embedding.fit_transform(modes)
    
    if rho is None:
        # Stage 2: MDS pre-computed
        modes_temp = np.copy(modes_emb)
        for i in range(modes.shape[0]):
            # Gather data points for mode i
            mode_clu = np.concatenate((np.reshape(modes[i,:], (1,d)), data[modes_label == i,:]), axis=0)
            clu_emb = embedding.fit_transform(mode_clu)
            modes_temp[i,:] = clu_emb[0,:]   ## Record the temporary mode embedding
            data_emb[modes_label == i,:] = clu_emb[1:,:]
            m_r0[i] = max(pdist(clu_emb[1:,:]))
        
        # Compute the default contrast parameter for decent visualizations
        rho = 2*max(m_r0)/np.percentile(pdist(modes_emb), 0.05)
        print("The current contrast factor is "+str(rho))
        modes_emb = rho*modes_emb
        # Stage 3: translating the data points based on the embeddings of modes
        for i in range(modes.shape[0]):
            # Compute the translation vector
            trans_vec = modes_emb[i,:] - modes_temp[i,:]
            data_emb[modes_label == i,:] = data_emb[modes_label == i,:] + trans_vec
    else:
        print("The current contrast factor is "+str(rho))
        modes_emb = rho*modes_emb
        ## Stage 2: MDS within each cluster
        for i in range(modes.shape[0]):
            # Gather data points for mode i
            mode_clu = np.concatenate((np.reshape(modes[i,:], (1,d)), data[modes_label == i,:]), axis=0)
            clu_emb = embedding.fit_transform(mode_clu)
            # Compute the translation vector (Stage 3)
            trans_vec = modes_emb[i,:] - clu_emb[0,:]
            data_emb[modes_label == i,:] = clu_emb[1:,:] + trans_vec
    
    return data_emb, modes_emb

'''
## Usage:
sim_data_emb, sim_modes_emb = Vis_Cluster(data=small_D, modes=uni_small_m, modes_label=small_ml, rho=1)
plt.scatter(sim_data_emb[:,0], sim_data_emb[:,1], c=small_ml, s=10)
plt.plot(sim_modes_emb[:,0], sim_modes_emb[:,1], "ro")
'''


'''
Enhanced mode clustering
@ Parameter:
    data -- Input data matrix (An n*d array of n data points in an d-dimensional space).
    h -- Smoothing parameter (Bandwidth for KDE).
    normal_ref -- Boolean, indicating whether the bandwidth is set using a normal reference rule.
    fast_ms -- Boolean, indicating whether the fast mean-shift algorithm using hierarchical clustering is applied.
    kernel -- The kernel name for KDE. ('gaussian', 'epanechnikov', or others) If others, please define your own kernel function and its first derivative. Specify the derivative function in the parameter 'D_kernel'.
    max_iter -- Maximal number of iteration for mean shift.
    eps -- The tolerance (stopping criterion) for mean shift iteration. 
    D_kernel: The first derivative of the user-specified kernel function.
    kernel_fun -- The user-specified kernel function.
    n0 -- The threshold size for tiny clusters.
    rho -- A float number specifying the contrast factor for preventing overlapping clusters in the visualization.
    cut -- The cut for hierarchical clustering in fast mean-shift (The distance threshold above which, clusters will not be merged).
    K -- The number of nearest neighbors for creating a graph that captures the connectivity.
    T_denoise -- Maximal times for denoising (If tiny clusters presence, we will remove them and redo mode clustering).
@ Return:
    1) The embedding coordinates for data points;
    2) The embedding coordinates for local modes;
    3) A 1-dim array for specifying the affiliation of each data point;
    4) The array for connectivities between clusters.
'''
def EMC_old(data, h=None, normal_ref=True, fast_ms=False, kernel="gaussian", max_iter=1000, eps=1e-8, D_kernel=None, kernel_fun=None, n0=None, rho=None, cut=0.1, K=10, T_denoise=5):
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    print("Sample size: " + str(n) + ";\n")
    print("Dimension: " + str(d) + ".\n")
    
    # Setting the bandwidth using a normal reference rule
    if h is None and normal_ref:
        h = np.mean(np.std(data, axis=0))*(4/(d+4))**(1/(d+6))*(1/n)**(1/(d+6))
    
    print("The current bandwidth is " + str(h) + ".\n")
    if h is None:
        print("We will use the Silverman's rule of thumb to select bandwidths for each coordinate.\n")
    
    ## Step 1: Mode Clustering
    print("Step 1: Mode Clustering...\n")
    if fast_ms:
        modes = Fast_Mean_Shift(data=data, query=data, h=h, cut=cut, K=K, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
    else:
        modes = Mean_Shift(data=data, query=data, h=h, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
    uni_modes, m_labels = Unique_Modes(modes, tol=1e-5)
    print("Finished.\n")
    
    if n0 is None:
        n0 = (n*np.log(n)/20)**(d/(d+6))  ## Default cluster size threshold
    ## Step 2: Denoising Small Clusters
    print("Step 2: Denoising Small Clusters...\n")
    uni_modes, m_labels = Unique_Modes(modes, tol=1e-5)
    labels_sig = np.copy(m_labels)
    modes_sig = np.copy(uni_modes)
    # Compute the number of data points in each cluster
    labels_cou = dict(Counter(m_labels))
    tiny_dict = {k: v for k,v in labels_cou.items() if v < n0}
    
    i_count = 1
    while len(tiny_dict) > 0:
        # Remove the data points from tiny clusters from the dataset used for density estimation
        cri_new = [(l not in tiny_dict.keys()) for l in labels_sig]
        # print(tiny_dict)
        re_data = data[cri_new,:]
        assert re_data.shape[0] > 0, "The denoising step is terminated since no clusters are significant. Please increase the value of the current bandwidth!"
        if fast_ms:
            modes_re = Fast_Mean_Shift(data=re_data, query=data, h=h, cut=cut, K=K, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
            modes_sig, labels_sig = Unique_Modes(modes_re, tol=1e-5)
        else:
            modes_re = Mean_Shift(data=re_data, query=data, h=h, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
            modes_sig, labels_sig = Unique_Modes(modes_re, tol=1e-5)
        labels_cou = dict(Counter(labels_sig))
        tiny_dict = {k: v for k,v in labels_cou.items() if v < n0}
        
        print("Iteration to denoise: " + str(i_count) + "\t")
        if i_count == T_denoise:
            print("WARNING: There are still tiny clusters!")
            break
        i_count += 1
    n_modes = modes_sig.shape[0]
    print("Finished.\n")
    print("Cluster size threshold: " + str(n0) + ".\n")
    print("The number of significant clusters is " + str(n_modes) + ".\n")
    
    ## Step 3: Soft Mode Clustering
    print("Step 3: Soft Mode Clustering...\n")
    soft_mat = Soft_Mode_Clustering(data=data, modes=modes_sig, h=h, kernel=kernel, kernel_fun=kernel_fun)
    print("Finished.\n")
    
    ## Step 4: Measuring Connectivity
    print("Step 4: Measuring Connectivity...\n")
    con_mat = Cluster_Connectivity(soft_modes=soft_mat, modes_label=labels_sig)
    print("Finished.\n")
    
    ## Step 5: Visualization
    print("Step 5: Visualization...\n")
    data_emb, modes_emb = Vis_Cluster(data=data, modes=modes_sig, modes_label=labels_sig, rho=rho)
    print("Finished.\n")
    
    return data_emb, modes_emb, labels_sig, con_mat

'''
## Usage:
sim_data_emb, sim_modes_emb, modes_labels, sim_con = EMC_old(sim_data3, h=None, fast_ms=False, rho=2, n0=None, cut=0.1)
'''


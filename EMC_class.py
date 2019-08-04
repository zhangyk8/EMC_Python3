"""
@author: yikun zhang

Last edit: August 3, 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS
from collections import Counter
from EMC_fun import Mean_Shift, Fast_Mean_Shift, Unique_Modes, Soft_Mode_Clustering, Cluster_Connectivity

class EMC:
    def __init__(self, data):
        ## data -- Input data matrix (An n*d array of n data points in an d-dimensional space).
        self.data = data
        self.h = None   ## The bandwidth for enhanced mode clustering
        self.label = None   ## Cluster labels for query points
        self.modes = None   ## Distinct local modes corresponding to each label
        self.con_mat = None   ## The connectivity matrix
        self.data_emb = None   ## The embedding coordinates for data points
        self.modes_emb = None   ## The embedding coordinates for local modes
        self.sc = None   ## The size of ordered clusters before denoising
        self.n0 = None   ## The size threshold for denoising tiny clusters
        self.rho = None   ## The contrast parameter for visualization (Avoid overlapping)
        self.noisy_label = None   ## The cluster labels for query points before denoising
        self.noisy_modes = None   ## The local modes corresponding to each label before denoising

    '''
    Visualization helper function for mode clustering
    @ Parameter:
        modes -- Distinct modes matrix (An m*d array of m distinct modes in an d-dimensional space).
        modes_label -- A 1-dim array for specifying the affiliation of each data point.
        rho -- A float number specifying the contrast factor for preventing overlapping clusters in the visualization.
    @ Return:
        1) The embedding coordinates for data points;
        2) The embedding coordinates for local modes.
    '''
    def Vis_Cluster(self, modes, modes_label, rho=None):
        assert self.data.shape[0] == len(
            modes_label), "The number of rows in the data array must be the same as the length of the 'modes_label' array!"
        assert modes.shape[0] == len(np.unique(
            modes_label)), "The number of rows in the mode array must be the same as the number of distinct labels in the 'modes_label' array!"
        d = self.data.shape[1]  ## The dimension of the data

        data_emb = np.empty([self.data.shape[0], 2])
        m_r0 = np.empty([modes.shape[0], ])  ## For computing the default rho
        ## Stage 1: MDS on modes
        embedding = MDS(n_components=2, n_init=7)
        modes_emb = embedding.fit_transform(modes)

        if rho is None:
            # Stage 2: MDS pre-computed
            modes_temp = np.copy(modes_emb)
            for i in range(modes.shape[0]):
                # Gather data points for mode i
                mode_clu = np.concatenate((np.reshape(modes[i, :], (1, d)), self.data[modes_label == i, :]), axis=0)
                clu_emb = embedding.fit_transform(mode_clu)
                modes_temp[i, :] = clu_emb[0, :]  ## Record the temporary mode embedding
                data_emb[modes_label == i, :] = clu_emb[1:, :]
                m_r0[i] = max(pdist(clu_emb[1:, :]))

            # Compute the default contrast parameter for decent visualizations
            rho = 2 * max(m_r0) / np.percentile(pdist(modes_emb), 0.05)
            print("The current contrast factor is " + str(rho))
            modes_emb = rho * modes_emb
            # Stage 3: translating the data points based on the embeddings of modes
            for i in range(modes.shape[0]):
                # Compute the translation vector
                trans_vec = modes_emb[i, :] - modes_temp[i, :]
                data_emb[modes_label == i, :] = data_emb[modes_label == i, :] + trans_vec
        else:
            print("The current contrast factor is " + str(rho))
            modes_emb = rho * modes_emb
            ## Stage 2: MDS within each cluster
            for i in range(modes.shape[0]):
                # Gather data points for mode i
                mode_clu = np.concatenate((np.reshape(modes[i, :], (1, d)), self.data[modes_label == i, :]), axis=0)
                clu_emb = embedding.fit_transform(mode_clu)
                # Compute the translation vector (Stage 3)
                trans_vec = modes_emb[i, :] - clu_emb[0, :]
                data_emb[modes_label == i, :] = clu_emb[1:, :] + trans_vec

        self.rho = rho
        return data_emb, modes_emb

    '''
    Fitting enhanced mode clustering
    @ Parameter:
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
    def fit(self, h=None, normal_ref=True, fast_ms=False, kernel="gaussian", max_iter=1000, eps=1e-8, D_kernel=None,
                kernel_fun=None, n0=None, rho=None, cut=0.1, K=10, T_denoise=5, noisy=False):
        n = self.data.shape[0]  ## Number of data points
        d = self.data.shape[1]  ## Dimension of the data
        print("Sample size: " + str(n) + ";\n")
        print("Dimension: " + str(d) + ".\n")

        # Setting the bandwidth using a normal reference rule
        if (h is None) and normal_ref:
            h = np.mean(np.std(self.data, axis=0)) * (4 / (d + 4)) ** (1 / (d + 6)) * (1 / n) ** (1 / (d + 6))

        print("The current bandwidth is " + str(h) + ".\n")
        if h is None:
            print("We will use the Silverman's rule of thumb to select bandwidths for each coordinate.\n")
        self.h = h

        ## Step 1: Mode Clustering
        print("Step 1: Mode Clustering...\n")
        if fast_ms:
            modes = Fast_Mean_Shift(data=self.data, query=self.data, h=h, cut=cut, K=K, kernel=kernel, max_iter=max_iter, eps=eps,
                                    D_kernel=D_kernel)
        else:
            modes = Mean_Shift(data=self.data, query=self.data, h=h, kernel=kernel, max_iter=max_iter, eps=eps, D_kernel=D_kernel)
        uni_modes, m_labels = Unique_Modes(modes, tol=1e-5)
        print("Finished.\n")

        if n0 is None:
            n0 = (n * np.log(n) / 20) ** (d / (d + 6))  ## Default cluster size threshold
        self.n0 = n0
        ## Step 2: Denoising Small Clusters
        print("Step 2: Denoising Small Clusters...\n")
        labels_sig = np.copy(m_labels)
        modes_sig = np.copy(uni_modes)
        # Compute the number of data points in each cluster
        labels_cou = dict(Counter(m_labels))
        self.sc = list(reversed(sorted(labels_cou.values())))
        tiny_dict = {k: v for k, v in labels_cou.items() if v < n0}

        i_count = 1
        while len(tiny_dict) > 0:
            # Remove the data points from tiny clusters from the dataset used for density estimation
            cri_new = [(l not in tiny_dict.keys()) for l in labels_sig]
            # print(tiny_dict)
            re_data = self.data[cri_new, :]
            assert re_data.shape[0] > 0, \
                "The denoising step is terminated since no clusters are significant. Please increase the value of the current bandwidth!"
            if fast_ms:
                modes_re = Fast_Mean_Shift(data=re_data, query=self.data, h=h, cut=cut, K=K, kernel=kernel,
                                           max_iter=max_iter, eps=eps, D_kernel=D_kernel)
                modes_sig, labels_sig = Unique_Modes(modes_re, tol=1e-5)
            else:
                modes_re = Mean_Shift(data=re_data, query=self.data, h=h, kernel=kernel, max_iter=max_iter, eps=eps,
                                      D_kernel=D_kernel)
                modes_sig, labels_sig = Unique_Modes(modes_re, tol=1e-5)
            labels_cou = dict(Counter(labels_sig))
            tiny_dict = {k: v for k, v in labels_cou.items() if v < n0}

            print("Iteration to denoise: " + str(i_count) + "\t")
            if i_count == T_denoise:
                print("WARNING: There are still tiny clusters!")
                break
            i_count += 1
        n_modes = modes_sig.shape[0]
        print("Finished.\n")
        print("Cluster size threshold: " + str(n0) + ".\n")
        print("The number of significant clusters is " + str(n_modes) + ".\n")
        self.modes = modes_sig
        self.label = labels_sig

        ## Step 3: Soft Mode Clustering
        print("Step 3: Soft Mode Clustering...\n")
        soft_mat = Soft_Mode_Clustering(data=self.data, modes=modes_sig, h=h, kernel=kernel, kernel_fun=kernel_fun)
        print("Finished.\n")

        ## Step 4: Measuring Connectivity
        print("Step 4: Measuring Connectivity...\n")
        con_mat = Cluster_Connectivity(soft_modes=soft_mat, modes_label=labels_sig)
        self.con_mat = con_mat
        print("Finished.\n")

        ## Step 5: Visualization
        print("Step 5: Visualization...\n")
        data_emb, modes_emb = self.Vis_Cluster(modes=modes_sig, modes_label=labels_sig, rho=rho)
        self.data_emb = data_emb
        self.modes_emb = modes_emb
        print("Finished.\n")

        if noisy:
            self.noisy_modes = uni_modes
            self.noisy_label = m_labels

    '''
    Visualization plot for the size of clusters
    @ Parameter:
        figsize -- The size of the output figure. Format: (width, height) in inches.
        title -- A string that specifies the title of the output figure.
        save_path -- A string that specifies the file path for the yielded figure. (If save_path=None, it only shows the plotting in the console).
    '''
    def SC_plot(self, figsize=(6.4, 4.8), title="SC-plot", save_path=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(list(range(len(self.sc))), self.sc, 'bo')
        plt.axhline(y=self.n0, label='n0', color='purple')
        plt.axvline(x=np.sum(np.array(self.sc) > self.n0)-0.5, color='grey')
        plt.legend()
        plt.xlabel("Index of the Ordered Cluster")
        plt.ylabel("Size of the Cluster")
        plt.title(title)
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path)

    '''
    Visualization plot for the enhanced mode clustering
    @ Parameter:
        omega0 -- The threshold for the connectivity measure between a pair of modes (Above the threshold, we will connect two clusters by a straight line).
        figsize -- The size of the output figure. Format: (width, height) in inches.
        pt_size -- The size of data points in the output figure.
        title -- A string that specifies the title of the output figure.
        save_path -- A string that specifies the file path for the yielded figure. (If save_path=None, it only shows the plotting in the console).
    '''
    def plot(self, omega0=None, figsize=(6.4, 4.8), pt_size=10, title='EMC Visualization', save_path=None):
        fig = plt.figure(figsize=figsize)
        plt.scatter(self.data_emb[:,0], self.data_emb[:,1], c=self.label, s=pt_size)
        plt.plot(self.modes_emb[:,0], self.modes_emb[:,1], 'ro')

        if omega0 is None:
            omega0 = 1/(2*self.con_mat.shape[0])
        for i in range(self.con_mat.shape[0]):
            for j in range(i+1, self.con_mat.shape[0]):
                if self.con_mat[i,j] > omega0:
                    # Reshape the coordinates of pairs of modes that have connectivity above the threshold
                    point_co = np.reshape(np.concatenate((self.modes_emb[i,:], self.modes_emb[j,:])), (2,2))
                    plt.plot(point_co[:,0], point_co[:,1], 'b-')

        plt.title(title)
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path)

    '''
    Print the summary of an EMC object
    '''
    def summary(self):
        print("Clusters size: ")
        print(dict(Counter(self.label)))
        print("\n")

        print("The connectivity matrix: ")
        print(self.con_mat)
        print("\n")

        print("The smoothing bandwidth: "+ str(self.h) + "\n")
        print("The threshold for denoising: " + str(self.n0) + "\n")

    '''
    Print the EMC object
    '''
    def print_EMC(self):
        print("The clustering labels: ")
        print(self.label)
        print("\n")

        print("The local modes: ")
        print(self.modes)
        print("\n")

        print("The connectivity matrix: ")
        print(self.con_mat)
        print("\n")

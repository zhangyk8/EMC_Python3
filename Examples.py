"""
@author: yikun zhang

Last edit: August 3, 2019
"""

from EMC_class import EMC
from EMC_fun import Five_Cluster

if __name__ == "__main__":
    ### Example 8.1 in Chen et al. (2016)
    # Generate 5-clusters in 10-Dim
    sim_data3 = Five_Cluster(N_c=200, N_f=100, dis_c=0.01, dis_f=0.005, d_add=7)
    # Create an EMC object
    EMC_ob = EMC(sim_data3)
    # Fitting the enhanced mode clustering (Set fast_ms=True if you want to use a fast mean shift algorithm via hierarchical clustering)
    EMC_ob.fit(h=None, fast_ms=False, rho=2, n0=None, cut=0.1)
    # Plot the sizes of clusters and Visualize the enhanced mode clustering
    EMC_ob.SC_plot(save_path='./figures/sc_plot.pdf')
    EMC_ob.plot(save_path='./figures/EMC_plot.pdf')
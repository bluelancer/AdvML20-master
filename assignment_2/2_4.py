""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

from Kruskal_v1 import Graph
#from Phylogeny import Phylogeny

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def calculate_responsibilities(samples,theta_list, N_data_samples, N_graphical_models, pi, topology_list):
    r_n_k = np.zeros([N_data_samples, N_graphical_models])
    sum_likelihood_by_k = np.zeros([N_data_samples])
    def find_root(topology_list_k):
        root_index = 0
        for i in range(len(topology_list_k)):
            if np.isnan(topology_list_k[i]):
                root_index = i
        return int(root_index)

    def per_GM_likelihood(xn, pTrans_xn_given_root_k, root_index_per_tree, topology_list_per_tree):
        for node_index in range(len(topology_list_per_tree)):
            parent_sample_value = xn[node_index]
            if node_index == root_index_per_tree:
                print('root_index_per_tree',root_index_per_tree,'node_index',node_index)
                print('parent_sample_value',parent_sample_value)
                print('pTrans_xn_given_root_k', pTrans_xn_given_root_k)
                print ('xn[node_index]',xn[node_index])
                likelihood = pTrans_xn_given_root_k[0][parent_sample_value]
            else:
                child_node_index = int(topology_list_per_tree[node_index])
                child_sample_value = xn[child_node_index]
                likelihood *= pTrans_xn_given_root_k[node_index][child_sample_value][parent_sample_value]

        return likelihood

    log_likelihood_of_mixture = 0
    for n in range(N_data_samples):
        likelihood_of_mixture = 0
        for k in range(N_graphical_models):
            pi_k = pi[k]
            theta_k = theta_list[k]
            xn = samples[n]
            topology_list_k = topology_list[k]

            root_index_k = find_root(topology_list_k)
#            pTrans_xn_given_root_k = theta_k[root_index_k][xn[root_index_k]]

            GM_likelihood_xn_k = per_GM_likelihood(xn, theta_k, root_index_k, topology_list_k)
            r_n_k[n, k] = GM_likelihood_xn_k * pi_k

            likelihood_of_mixture += r_n_k[n, k]
            log_likelihood_of_mixture += math.log(likelihood_of_mixture)

            sum_likelihood_by_k[n] +=  r_n_k[n, k]
        r_n_k[n,:]  =  r_n_k[n,:] /sum_likelihood_by_k[n]
    r_n_k += sys.float_info.epsilon

    return r_n_k,log_likelihood_of_mixture

def qk_Xs_Xt (num_clusters,Xs_len,Xt_len,r_n_k,samples):
    nominator = []
    denominator = np.sum(r_n_k, axis=0)
    qk_Xs_Xt_joint_prob_cluster_k = np.zeros([Xs_len,2,Xt_len,2,num_clusters])

    for Xs in range(Xs_len):
        for Xt in range(Xt_len):
            for a in range(2):
                for b in range(2):
                    for n in range(len(samples)):
                        if samples[n][Xs] == a and samples[n][Xt] == b:
                            nominator.append(r_n_k[n])
                        nominator_sum = np.sum(nominator, axis=0)
                        qk_Xs_Xt_joint_prob_cluster_k[Xs][a][Xt][b][:] = nominator_sum/denominator
                        nominator =[]
    return qk_Xs_Xt_joint_prob_cluster_k

def qk_Xs(num_clusters,Xs_len,r_n_k,samples):
    qk_Xs_prob_cluster_k = np.zeros([Xs_len,2,num_clusters])
    nominator = []
    denominator = np.sum(r_n_k, axis=0)

    for Xs in range(Xs_len):
        for a in range(2):
            for n in range(len(samples)):
                if samples[n][Xs] == a:
                    nominator.append(r_n_k[n])
                nominator_sum = np.sum(nominator, axis=0)
                qk_Xs_prob_cluster_k[Xs][a][:] = nominator_sum / denominator
                nominator = []
    return qk_Xs_prob_cluster_k


def I_qk_Xs_X_t(num_clusters,qk_Xs_Xt_joint_prob_cluster_k,qk_Xs_prob_cluster_k,Xs_len,Xt_len):
    elements = []
    elements_sum = np.zeros([Xs_len,Xt_len,num_clusters])
    for Xs in range(Xs_len):
        for Xt in range(Xt_len):
            for k in range(num_clusters):
                for a in range(2):
                    for b in range(2):
                        if qk_Xs_Xt_joint_prob_cluster_k[Xs][a][Xt][b][k] != 0:
                            element = qk_Xs_Xt_joint_prob_cluster_k[Xs][a][Xt][b][k] * \
                                      math.log(qk_Xs_Xt_joint_prob_cluster_k[Xs][a][Xt][b][k]/
                                                    (qk_Xs_prob_cluster_k[Xs][a][k] * qk_Xs_prob_cluster_k[Xt][b][k]))
                        else:
                            element = 0
                        elements.append(element)
                elements_sum[Xs][Xt][k] = np.sum(elements, axis=0)
                elements = []
    I_qk_Xs_X_t_cluster_k = elements_sum
    return I_qk_Xs_X_t_cluster_k

def generate_G_k(num_clusters,I_qk_Xs_X_t_cluster_k,Xs_len,Xt_len):
    G_k = []
    for k in range(num_clusters):
        g = Graph(Xs_len)
        for Xs in range(Xs_len):
            for Xt in range(Xs+1, Xt_len):
                g.addEdge(Xs,Xt,I_qk_Xs_X_t_cluster_k[Xs][Xt][k])
        G_k.append(g)
    return G_k


def MST_k(num_clusters,G_k,Xs_len):
    T_k = []
    for k in range(num_clusters):
        T = np.zeros([Xs_len])
        edge_list = G_k[k].maximum_spanning_tree()




    pass

def update_theta_list():
    pass



def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seed
    np.random.seed(seed_val)

    # TODO: Implement EM algorithm here.

    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Running EM algorithm...")

    pi = []
    topology_list = []
    theta_list = []
    likelihood = []
    loglikelihood = []


    num_samples = samples.shape[0]
    num_nodes = samples.shape[1]

    from Tree import TreeMixture

    tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    tm.sample_mixtures(num_samples=num_samples, seed_val=seed_val)

    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())
        pi.append (tm.pi[i])

    for iter_ in range(max_num_iter):
        # Stage 1: compute responsibility matrix r_n_k
        r_n_k,log_likelihood_of_mixture = calculate_responsibilities(samples,theta_list,num_samples,num_clusters, pi, topology_list)
        loglikelihood.append(log_likelihood_of_mixture)
        # Stage 2: set pi_prime
        pi = np.sum(r_n_k,axis=0)/num_samples
        # Stage 3:
        qk_Xs_Xt_joint_prob_cluster_k = qk_Xs_Xt(num_clusters,num_nodes,num_nodes,r_n_k,samples)
        qk_Xs_prob_cluster_k = qk_Xs(num_clusters,num_nodes,r_n_k,samples)
        I_qk_Xs_X_t_cluster_k = I_qk_Xs_X_t(num_clusters,qk_Xs_Xt_joint_prob_cluster_k, qk_Xs_prob_cluster_k,num_nodes,num_nodes)
        G_k = generate_G_k(num_clusters, I_qk_Xs_X_t_cluster_k, num_nodes, num_nodes)
        T_k = MST_k(G_k)
        # Stage 4: MST

        # print ('nice')
        # Stage 5:
        theta_list = update_theta_list()


    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    # End: Example Code Segment

    ###

    return loglikelihood, topology_list, theta_list

def main():
    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.4.")

    seed_val = 123

    sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "q2_4_results.txt"
    real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(seed_val, samples, num_clusters=num_clusters)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison

        print("\t4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison


if __name__ == "__main__":
    main()

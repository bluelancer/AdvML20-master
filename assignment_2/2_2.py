""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

import numpy as np
from Tree import Tree
from Tree import Node

dp = np.zeros ([1,1])

def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """
    global dp
    global dp_left
    global dp_right
    global G_topology_list

    # TODO Add your code here
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    # print("Calculating the likelihood...")
    # print ("tree_topology",tree_topology)

    # if this is a new sample, then len(tree_topology) = len(beta) = num_vertices
    if len(tree_topology) == len(beta):
        dp = np.zeros([len(beta),5])
        n = 0
        for i in beta:
            if not np.isnan(i):
                dp[n,int(i)] = 1
                # print (n)
            n += 1
        # print ("dp_init", dp)
        dp_left = np.zeros([len(beta),5])
        dp_right = np.zeros([len(beta),5])
        G_topology_list = tree_topology.tolist()
        tree_topology = tree_topology[1:]

    if len(tree_topology) == 0:
        pass
        # print("find leaves")
        # v_parent = G_topology_list[len(G_topology_list)-1]


    else:
        v_parent = tree_topology[0]
        update_topology = tree_topology[2:]
        print ("compute dp _i = ",v_parent)
        _ = calculate_likelihood(update_topology, theta, beta)

        for j in range(5):
            v_left = G_topology_list.index(v_parent)
            v_right = G_topology_list.index(v_parent,G_topology_list.index(v_parent)+1)
            # 𝑝(𝑋𝑣=𝑗|𝑋𝑢=𝑖)
            # 𝑣 = len(tree_topology) - 1 ; 𝑋𝑜∩↓𝑣 = beta[len(tree_topology) - 1])
            # P(𝑋𝑢=𝑖 |𝑋pa(𝑢)=𝑗 ) = theta[u][i][j]
            theta_left = theta[int(v_left)][:][:]
            # 𝑝(𝑋w=𝑗|𝑋𝑢=𝑖)
            # 𝑤 = len(tree_topology) - 2 ;  𝑋𝑜∩↓𝑤 = beta[len(tree_topology) - 2]
            theta_right = theta[int(v_right)][:][:]

            # 𝑠(𝑣,𝑗) = 𝑝(𝑋𝑜∩↓𝑣|𝑋𝑣=𝑗) = dp[int(v_parent)][j] = calculate_likelihood(update_topology, theta, beta)
            # parent_prob = 𝑝(𝑋𝑜∩↓𝑣|𝑋𝑢=𝑖) != likelihood, yet we need to update dp with this,

            # dp_left[u] = 𝑝(𝑋𝑜∩↓𝑣|𝑋𝑢=𝑖) = =Σ_j 𝑝(𝑋𝑜∩↓𝑢|𝑋𝑢=𝑗) (dp_left [u][j]) 𝑝(𝑋𝑣=𝑗|𝑋𝑢=𝑖) (theta[v][j][i])
            # print ("left_leaf:",G_topology_list.index(v_parent))
            dp_left[int(v_parent)] = np.add(dp_left[int(v_parent)], \
                                     np.multiply(theta_left[j], dp[v_left][j]))
            # print ("right_leaf:",G_topology_list.index(v_parent,G_topology_list.index(v_parent)+1))
            # print ("theta_left[j]",theta_left[j])
            dp_right[int(v_parent)] = np.add(dp_right[int(v_parent)], \
                                                      np.multiply(theta_right[j], dp[v_right][j]))

            # 𝑠(𝑢, 𝑖) = 𝑝(𝑋𝑜∩↓𝑢|𝑋𝑢=𝑖) = 𝑝(𝑋𝑜∩↓𝑣 | 𝑋𝑢 = 𝑖)𝑝(𝑋𝑜∩↓𝑤 | 𝑋𝑢 = 𝑖) =
            # Σ_j 𝑝(𝑋𝑜∩↓𝑣|𝑋𝑣=𝑗)𝑝(𝑋𝑣=𝑗|𝑋𝑢=𝑖) * Σ_k 𝑝(𝑋𝑜∩↓𝑤|𝑋𝑤=k)𝑝(𝑋𝑤=k|𝑋𝑢=𝑖)

        dp_right_left = np.multiply(dp_left[int(v_parent)],dp_right[int(v_parent)])
        dp[int(v_parent)] = dp_right_left

    # print ("dp = ", dp)
    if len(tree_topology) == len(beta) -1:
        likelihood = np.dot(theta[0], dp[0])
        print ("likelihood = ",likelihood)
    return dp
    # End: Example Code Segment

def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename ="data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)

if __name__ == "__main__":
    main()

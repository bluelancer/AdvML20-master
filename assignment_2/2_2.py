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
    global G_topology_list
    global child_descendants
    global descendants
    global t

    def get_descendants(current_node_idx, tree_topology):
        descendants = []
        for i, x in enumerate(tree_topology):
            if x == current_node_idx:
                descendants.append(i)
        return descendants

    def get_child_descendants(tree_topology):
        child_descendants = []
        for i in range(len(tree_topology)):
            descendants = get_descendants(i, tree_topology)
            if len(descendants) == 2:
                child_descendants.append((descendants[0], descendants[1]))
            elif len(descendants) == 1:
                child_descendants.append((descendants[0], np.nan))
            else:
                child_descendants.append((np.nan, np.nan))
        return child_descendants

    def get_likelihood_for_current_node_idx (child_descendants,current_node_idx,theta,dp):
        descendants = child_descendants[current_node_idx]
        s_u_i= np.ones([5])
        s_u_i_part = np.zeros([len(theta),5])
        s_u_i_component = []
        for v in descendants:
            for i in range(5):
                s_u_i_part[v][i] = np.dot(theta[v][:][i], dp[v][:])
            for j in range(5):
                s_u_i[j] *= s_u_i_part[v][j]
        return s_u_i



    # TODO Add your code here
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    # print("Calculating the likelihood...")
    # print ("tree_topology",tree_topology)
    # if this is a new sample, then len(tree_topology) = len(beta) = num_vertices
    if len(tree_topology) == len(beta):
        # init
        # 𝑠(𝑣,𝑗) = 𝑝(𝑋𝑜∩↓𝑣|𝑋𝑣=𝑗) = dp[int(v_parent)][j]
        dp = np.zeros([len(beta), 5])
        dp_left = np.ones([len(beta), 5])
        G_topology_list = tree_topology.tolist()

        # print (tree_topology)
        #t = Tree()
        #t.load_tree_from_direct_arrays(tree_topology, theta)
        #nodes = t.root

        # print('after',tree_topology)
        tree_topology = tree_topology[:-1]
        child_descendants = get_child_descendants(G_topology_list)
        descendants = 0

    if descendants == 0:

        # likelihood  = {𝑝(𝑋𝑜∩↓𝑣|𝑋𝑣=𝑗)𝑝(𝑋𝑣=𝑗) | 𝑣=0} = sum(dp[0]*theta[0])
        # This should be correct :)
        descendants = child_descendants[0]
        temp  = descendants
        dp = calculate_likelihood(tree_topology, theta, beta)
        descendants = temp
        dp[0] = get_likelihood_for_current_node_idx(child_descendants,0,theta,dp)
        likelihood = np.dot(dp[0], theta[0])
        return likelihood

    else:
        # for nodes in the child vertex of a parent vertex
        for descendant in descendants:

            # the child node index
            current_node_idx = int(descendant)
            descendants = child_descendants[current_node_idx]
            v_parent = int(G_topology_list[current_node_idx])

            #print('current_node_idx, ',current_node_idx, 'descendants, ', descendants,'parent',v_parent )

            # probability of per child vertex's cdf, on vertex's observation
            #per = np.zeros([5, 5])

            # current node is not a leaf
            if np.isnan(beta[current_node_idx]):
                #print ('Node is not a leaf', current_node_idx)

                # progress to child's child later
                descendants = child_descendants[descendant]

                # Probability that given a child vertex index, and the observation of child vertex, recurse to it's parent's observation cdf
                # Given child vertex index: 𝑝(𝑋𝑣=𝑗|𝑋𝑢=𝑖) (theta[v][j][i])
                #theta_trans = theta[current_node_idx][:][:]

                # computing the per-child's observation (j) cdf, with different parent observation cdf (dp[v_parent]) within dictionary
                temp = descendants
                dp = calculate_likelihood(tree_topology, theta, beta)
                descendants = temp

                dp[current_node_idx][:] = get_likelihood_for_current_node_idx(child_descendants,current_node_idx,theta,dp)

            else:
                # current node is a leaf
                # {𝑝(𝑋𝑜∩↓𝑣|𝑋𝑣 = 𝑗)𝑝(𝑋𝑣=𝑗)|𝑣=𝑜∩↓𝑣} = 1
                #print('find leaves', current_node_idx)
                node_value = beta[current_node_idx]
                dp[current_node_idx][int(node_value)] = 1

    return dp

# End: Example Code Segment

def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename ="data/q2_2/q2_2_large_tree.pkl"  # "data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
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

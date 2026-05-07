import os
import pickle
import logging
from collections import defaultdict
import numpy as np
import torch
import networkx as nx
from scipy.linalg import fractional_matrix_power as fmp
from uq_methods.dse import *
import scipy

ALPHAS_RANGE = np.arange(0, 1.01, 0.1)
HEAT_T_RANGE = np.arange(0.1, 0.71, 0.1)
MATERN_KAPPA_RANGE = [1.0, 2.0, 3.0]
MATERN_NU_RANGE = [1.0, 2.0, 3.0]
EPS = 1e-12


def find_cliques_directed(G):
    G1 = nx.Graph()
    for u,v in G.edges():
        if u in G[v]:
            G1.add_edge(u,v)
    return nx.find_cliques(G1)


def contract_cliques(G, cliques):
    used = [False] * len(G.nodes)
    for clique in cliques:
        first_node = None
        for node in clique:
            if used[node]:
                continue
            if first_node is None:
                first_node = node
            else:
                G = nx.contracted_nodes(G, first_node, node)
                used[first_node] = True
                used[node] = True
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def normalize_kernel(K):
    diagonal_values = np.sqrt(np.diag(K)) + EPS
    normalized_kernel = K / np.outer(diagonal_values, diagonal_values)
    return normalized_kernel


def scale_entropy(entropy, n_classes):
    max_entropy = -np.log(1.0 / n_classes)  # For a discrete distribution with num_classes
    scaled_entropy = entropy / max_entropy
    return scaled_entropy


def vn_entropy(K, normalize=True, scale=True, jitter=0):
    if normalize:
        K = normalize_kernel(K) / K.shape[0]
    result = 0
    eigvs = np.linalg.eig(K + jitter * np.eye(K.shape[0])).eigenvalues.astype(np.float64)
    for e in eigvs:
        if np.abs(e) > 1e-8:
            result -= e * np.log(e)
    if scale:
        result = scale_entropy(result, K.shape[0])
    return np.float64(result)


def contract_cliques_impl(G, cliques):
    used = [False] * len(G.nodes)
    for clique in cliques:
        first_node = None
        for node in clique:
            if used[node]:
                continue
            if first_node is None:
                first_node = node
            else:
                G = nx.contracted_nodes(G, first_node, node)
                used[first_node] = True
                used[node] = True
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def get_laplacian(G, norm_lapl):
    if isinstance(G, nx.DiGraph):
        L = nx.directed_laplacian_matrix(G)
    elif norm_lapl:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    return L


def heat_kernel(G: nx.Graph, t: float = 0.4, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    return scipy.linalg.expm(-t * L)


def matern_kernel(G: nx.Graph, kappa: float = 1, nu=1, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    I = np.eye(L.shape[0])
    #return fmp(nu * I + L, -alpha / 2) @ fmp(nu * I + L.T, -alpha / 2)
    return fmp((2 * nu / kappa**2) * I + L, -nu)


def get_entailment_graph(strings_list, model, question=None, is_weighted=False, weight_strategy="manual"):
    """
    Get graph of entailment
    """
    def get_edge(text1, text2, is_weighted=False, question=None):
        implication_1, prob_impl1 = model.check_implication(text1, text2, question)
        implication_2, prob_impl2 = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2])
        weight = int(implication_1 == 2) + int(implication_2 == 2) + 0.5 * int(implication_1 == 1) + 0.5 * int(implication_2 == 1)
        if is_weighted:
            if weight_strategy == "manual":
                return weight
            elif weight_strategy == "deberta":
                return prob_impl1 + prob_impl2
            else:
                raise ValueError(f"Unknown weight strategy {weight_strategy}")
        return weight >= 1.5

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    nodes = range(len(strings_list))
    edges = []
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                edge = get_edge(string1, strings_list[j], is_weighted=is_weighted, question=question)
                if is_weighted:
                    if edge:
                        edges.append((i, j, edge))
                else:
                    edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    return G


def get_semantic_ids_graph(strings_list, model, semantic_ids, ordered_ids, strict_entailment=False, question=None):
    """Group list of predictions into semantic meaning."""
    def are_similar(text1, text2):

        implication_1, prob = model.check_implication(text1, text2, question)
        implication_2, prob = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        return (implication_1 == 2) + (implication_1 == 1) * 0.5 +\
               (implication_2 == 2) + (implication_2 == 1) * 0.5

    # Initialise all ids with -1.
    nodes = ordered_ids
    weights = defaultdict(list) # (i, j) -> weight
    for i, string1 in enumerate(strings_list):
        node_i = semantic_ids[i]
        for j in range(i + 1, len(strings_list)):
            node_j = semantic_ids[j]
            edge_weight = are_similar(string1, strings_list[j])
            if edge_weight > 0:
                weights[(node_i, node_j)].append(edge_weight)
    for k, v in weights.items():
        weights[k] = np.sum(v)
    assert -1 not in semantic_ids
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(i, j, w) for (i, j), w in weights.items()])
    return G

def get_kernels(graph):
    kernels = {}
    for t in HEAT_T_RANGE:
        kernels[f"heat_t={t:.2}"] = heat_kernel(graph, t=t)
        kernels[f"heatn_t={t:.2}"] = heat_kernel(graph, t=t, norm_lapl=True)

    for kappa in MATERN_KAPPA_RANGE:
        for nu in MATERN_NU_RANGE:
            kernels[f"matern_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu)
            kernels[f"maternn_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu, norm_lapl=True)

    return kernels


def all_graph_entropies(graph):
    kernels = get_kernels(graph)
    results = []
    for kernel_name, kernel in kernels.items():
        for scale in [True, False]:
            kernel_entropy = vn_entropy(kernel, scale=scale)
            postfix = "_s" if scale else ""
            results.append((f'{kernel_name}_kernel_entropy{postfix}', kernel_entropy))
    return results


def kle(responses, model, question, args):
    """
    official default setting is t=0.3 kappa = 1.0 nu=1.0
    x=1,2,3,4,5,6,7
    {weighted}_{deberta}_heat{n}_t=0.x_kernel_entropy_{s}
    x=1,2,3 
    {weighted}_{deberta}_matern{n}_kappa=x.0_nu=x.0_kernel_entropy_{s}
    s = scale, scale or not have no obvious effect
    n = normalize_laplacian
    weighted = use weighted graph
    deberta = use deberta weight strategy
    """
    result_dict = {}
    result_dict['semantic_ids'] = []

    entropies = defaultdict(float)

    semantic_ids = get_semantic_ids(responses, model, question, strict_entailment=args.strict_entailment)


    result_dict['semantic_ids'].append(semantic_ids)
    entropies['cluster_assignment_entropy']=cluster_assignment_entropy(semantic_ids)
    graph = get_entailment_graph(responses, model, question=question, is_weighted=False, weight_strategy="manual")
    for k, value in all_graph_entropies(graph):
        entropies[k]=value

    weighted_graph = get_entailment_graph(responses, model, question=question, is_weighted=True, weight_strategy="manual")
    for k, value in all_graph_entropies(weighted_graph):
        entropies[f"weighted_{k}"]=value

    weighted_graph_deberta = get_entailment_graph(responses, model, question, is_weighted=True, weight_strategy="deberta")
    for k, value in all_graph_entropies(weighted_graph_deberta):
        entropies[f"weighted_deberta_{k}"]=value

    # Methods that rely on log_likelihood_per_semantic_id require model logits,
    # which are not available in black-box settings.
    # Computing K_se requires log_likelihood_per_semantic_id from model output logs.

    result_dict['entropies'] = entropies
    # result_dict['graphs'] = {
    #     'graph': graph,
    #     'weighted_graph': weighted_graph,
    #     'weighted_graph_deberta': weighted_graph_deberta
    # }
    return result_dict
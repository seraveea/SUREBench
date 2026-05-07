# implementation of methods in paper Generating with Confidence

import numpy as np

def compute_weighted_matrix(pipeline, strings_list, question):
    """
    Compute the symmetric weighted matrix for the given strings_list.
    """
    num_strings = len(strings_list)
    W = [[0] * num_strings for _ in range(num_strings)]
    for i in range(num_strings):
        for j in range(i, num_strings):
            implication_1, _ = pipeline.check_implication(strings_list[i], strings_list[j], question)
            implication_2, _ = pipeline.check_implication(strings_list[j], strings_list[i], question)
            assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

            W[i][j] = (implication_1/2 + implication_2/2)/2
            W[j][i] = W[i][j]

    return W

def compute_laplacian_matrix(W):
    """
    Compute the Laplacian matrix L = I - D^(-0.5)WD^(-0.5), where D is the degree matrix of W.
    """
    W = np.array(W)
    num_strings = len(W)
    # Compute degree matrix D (diagonal matrix with sum of each row of W)
    D_diag = np.sum(W, axis=1)
    # Compute D^(-0.5)
    # Add small epsilon to avoid division by zero
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D_diag + 1e-10))
    
    # Compute I - D^(-0.5)WD^(-0.5)
    I = np.eye(num_strings)
    L = I - D_sqrt_inv @ W @ D_sqrt_inv
    return L

def compute_U_eigv(L):
    """
    Compute the eigenvalues of the Laplacian matrix L.
    Args:
        L: Laplacian matrix
    
    Returns:
        numpy array: sorted eigenvalues in ascending order
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(L)
    
    # Sort eigenvalues in ascending order
    eigenvalues = np.sort(eigenvalues.real)
    U_eigv = 0
    for i in range(len(eigenvalues)):
        if 1-eigenvalues[i] > 0:
            U_eigv += 1-eigenvalues[i]
    
    return U_eigv

def compute_UC_deg(W):
    W = np.array(W)
    num_strings = len(W)
    # Compute degree matrix D (diagonal matrix with sum of each row of W)
    D_diag = np.sum(W, axis=1)
    # Create identity matrix of size m
    I = np.eye(num_strings)
    # Create diagonal matrix D from D_diag
    D = np.diag(D_diag)
    # Compute (mI - D)/m^2
    result = (num_strings * I - D) / (num_strings ** 2)
    # Compute the trace
    U_deg = np.trace(result)
    # Compute D_jj/num_strings for each j
    C_deg = D_diag / num_strings
    
    # This is an array where each element is D_jj/num_strings
    # UC_deg was calculated above as the trace of (mI - D)/m^2
    
    return U_deg, C_deg

def compute_UC_ecc(L, k=5, mask_eigenvalue=0.5):
    """
    Compute uncertainty and confidence measures using eccentricity method.
    
    Args:
        L: Laplacian matrix
        k: Number of smallest eigenvectors to use (default use all)
        mask_eigenvalue: Threshold for eigenvalue masking (default 0.5), this is used in original paper
    
    Returns:
        tuple: (U_ecc, C_ecc)
            - U_ecc: Uncertainty measure based on eccentricity
            - C_ecc: List of confidence scores for each response
    """
    # Compute eigenvalues and eigenvectors of Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(L) 
    # Sort eigenvectors by their magnitudes

    keep_mask = eigenvalues > mask_eigenvalue
    eigenvalues, smallest_eigenvectors = eigenvalues[keep_mask], eigenvectors[:, keep_mask]
    smallest_eigenvectors = smallest_eigenvectors.T
    C_ecc = (-1)*np.asarray(
        [np.linalg.norm(x-x.mean(0),2) for x in smallest_eigenvectors]
    )
    U_ecc = np.linalg.norm(C_ecc, 2)


    return U_ecc, C_ecc


if __name__ == "__main__":
    """
    For Generating with Confidence, there are three categories:
    1. Eigenvalue-based uncertainty: U_eigv
    2. Degree-matrix-based uncertainty: U_deg, C_deg
    3. Eccentricity-based uncertainty: U_ecc, C_ecc

    All are based on a semantic adjacency matrix W, where W_ij indicates
    the semantic relation between response i and response j.
    W_ij = 1 means semantically equivalent; W_ij = 0 means semantically different.
    When most entries approach 1, U_deg -> 0 and U_eigv decreases.
    U_ecc behavior is less clear and may require more experiments.

    """
    # Create a symmetric matrix with values between 0 and 1, diagonal values = 1
    np.random.seed(42)  # For reproducibility
    n = 5
    # Generate random matrix between 0 and 1

    # Create a matrix with most values close to 0.1
    W_test = 0.999 + 0.001 * np.random.random((n, n))
    

    W_test = (W_test + W_test.T) / 2
    np.fill_diagonal(W_test, 1.0)


    # Compute Laplacian
    L_test = compute_laplacian_matrix(W_test)

    eigenvalues = np.sort(np.linalg.eigvals(L_test).real)
    print("Eigenvalues of L_test:", eigenvalues)


    print('U_eigv:')
    print(compute_U_eigv(L_test))


    U_ecc, C_ecc = compute_UC_ecc(L_test, k=n)
    print("\nUncertainty (U_ecc):", U_ecc)
    print("Confidence scores (C_ecc):", C_ecc)



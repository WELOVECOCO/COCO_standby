import numpy as np
def components_equal_up_to_sign(a, b, atol=1e-6):
    """Check if each column in a matches b or -b."""
    if a.shape != b.shape:
        return False
    for i in range(a.shape[1]):
        if not (
            np.allclose(a[:, i], b[:, i], atol=atol) or
            np.allclose(a[:, i], -b[:, i], atol=atol)
        ):
            return False
    return True

def pca_svd(x: np.ndarray, n_components=2):
    n, f = x.shape
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)

    principal_components = x_centered @ Vt[:n_components, :].T

    explained_variance = (S[:n_components] ** 2) / (n - 1)
    total_variance = np.sum(S**2) / (n - 1)
    explained_variance_ratio = explained_variance / total_variance

    return principal_components, explained_variance_ratio



def pca_sklearn(x: np.ndarray, n_components=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)

    explained_variance_ratio = pca.explained_variance_ratio_

    return principal_components, explained_variance_ratio


def pca_eigh(x: np.ndarray, n_components=2):
    n, f = x.shape
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    cov = x_centered.T @ x_centered / (n - 1)  

    eigenvalues, eigenvectors = np.linalg.eigh(cov) 
    idx = np.argsort(eigenvalues)[::-1]               # not inherintly sorted because the internal implementation returns them in ascending order
    eigenvectors = eigenvectors[:, idx]  
    eigenvectors_subset = eigenvectors[:, :n_components]  

    
    principal_components = x_centered @ eigenvectors_subset 

    explained_variance = eigenvalues[idx][:n_components]
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance



    return principal_components, explained_variance_ratio



x = np.random.randn(100, 10)


principal_components, explained_variance_ratio = pca_svd(x)
principal_components_sklearn, explained_variance_ratio_sklearn = pca_sklearn(x)

assert components_equal_up_to_sign(principal_components, principal_components_sklearn)
assert np.allclose(explained_variance_ratio, explained_variance_ratio_sklearn, atol=1e-6)


print("TESTS PASSED")


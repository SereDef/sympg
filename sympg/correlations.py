import numpy as np

def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Compute a nearest symmetric positive semidefinite matrix."""
    B = (A + A.T) / 2
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if _is_pd(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not _is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3


def _is_pd(B: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

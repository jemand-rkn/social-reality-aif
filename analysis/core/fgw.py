from __future__ import annotations

import warnings

import numpy as np
import ot


def compute_fgw_distance_vectorized(
    rdm_all: np.ndarray,
    m_all: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute FGW distance matrix (non-negative distance values).

    Args:
        rdm_all: [A, R, R] internal distance matrices for all agents.
        m_all: [A, A, R, R] feature distance matrices for all agent pairs.
        alpha: feature weight in fused GW.
    """
    num_agents, num_refs, _ = rdm_all.shape
    distance_matrix = np.zeros((num_agents, num_agents), dtype=np.float64)

    c_all = rdm_all
    m_all_np = m_all

    p = ot.unif(num_refs)
    q = ot.unif(num_refs)

    for i in range(num_agents):
        for j in range(i, num_agents):
            c_x = c_all[i]
            c_y = c_all[j]
            m_ij = m_all_np[i, j]

            if i == j:
                fgw_dist = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="invalid value encountered in scalar divide",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="divide by zero encountered in scalar divide",
                        category=RuntimeWarning,
                    )
                    fgw_dist = ot.gromov.fused_gromov_wasserstein2(
                        m_ij,
                        c_x,
                        c_y,
                        p,
                        q,
                        alpha=alpha,
                        log=False,
                    )

            distance = float(fgw_dist)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

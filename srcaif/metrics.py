import numpy as np
import torch
import ot
import networkx as nx
import warnings
from networkx.algorithms import community
from sklearn.metrics import pairwise_distances
import rsatoolbox as rsa
from scipy.spatial import KDTree
from scipy.special import digamma
from typing import Dict, List, Tuple, Optional


def compute_rsa_within_clusters(
    society,
    num_samples_per_agent: int = 50,
) -> Dict[str, any]:
    """
    Cluster the social network and compute Representational Similarity Analysis (RSA)
    within each cluster.
    
    For each cluster:
    1. Sample the same number of observations from each agent's buffer
    2. Infer latent representations for these observations
    3. Compute RSA between latent and observation spaces
    
    Args:
        society: Society object with agents and network graph
        num_samples_per_agent: Number of observations to sample from each agent's buffer
        device: torch device (e.g., 'cpu' or 'cuda')
    
    Returns:
        Dictionary containing:
        - 'clusters': List of clusters (each cluster is a list of agent IDs)
        - 'agent_rsa': [num_all_agents,] RSA values per agent (flattened)
        - 'cluster_rsa': [num_clusters,] RSA values per cluster (mean of agents)
                - 'cluster_agent_rsa': [num_clusters, num_agents] RSA values where
                    entry [cluster_idx, agent_id] is agent_id evaluated on cluster_idx's pool
                - 'cluster_cluster_rsa': [num_clusters, num_clusters] RSA values where
                    entry [cluster_i, cluster_j] is mean RSA of agents in cluster_i on cluster_j's pool
        - 'cluster_agents': Dict mapping cluster index to agent IDs in that cluster
    """
    # 1. Cluster the network using greedy modularity optimization
    try:
        # Use Louvain-like algorithm (greedy modularity)
        communities_list = list(community.greedy_modularity_communities(society.graph))
    except:
        # Fallback to simple connected components if the above fails
        communities_list = list(nx.connected_components(society.graph))
    
    # Convert to list of sorted lists for consistency
    clusters = [sorted(list(c)) for c in communities_list]
    
    num_clusters = len(clusters)
    num_agents = society.num_agents

    agent_rsa = np.full(num_agents, np.nan, dtype=float)
    cluster_rsa = np.full(num_clusters, np.nan, dtype=float)
    cluster_agent_rsa = np.full((num_clusters, num_agents), np.nan, dtype=float)
    cluster_cluster_rsa = np.full((num_clusters, num_clusters), np.nan, dtype=float)

    agent_to_cluster = {}
    for cluster_idx, cluster_agent_ids in enumerate(clusters):
        for agent_id in cluster_agent_ids:
            agent_to_cluster[agent_id] = cluster_idx

    results = {
        'clusters': clusters,
        'agent_rsa': agent_rsa,
        'cluster_rsa': cluster_rsa,
        'cluster_agent_rsa': cluster_agent_rsa,
        'cluster_cluster_rsa': cluster_cluster_rsa,
        'cluster_agents': {},
    }

    # 2. Build pooled observations per cluster
    cluster_pools = [None] * num_clusters
    for cluster_idx, cluster_agent_ids in enumerate(clusters):
        cluster_agents = [society.agents[agent_id] for agent_id in cluster_agent_ids]

        # Find minimum buffer size in this cluster
        min_buffer_size = min(len(agent.buffer) for agent in cluster_agents)

        if min_buffer_size == 0:
            # Skip clusters where any agent has empty buffer
            results['cluster_agents'][cluster_idx] = cluster_agent_ids
            continue

        # Determine actual sample size (don't exceed minimum buffer size)
        actual_samples = min(num_samples_per_agent, min_buffer_size)

        # 3. Sample observations from all agents in the cluster
        all_sampled_observations = []
        for agent in cluster_agents:
            # Sample observations from each agent's buffer
            sampled_obs = agent.buffer.sample(actual_samples)  # [actual_samples, *obs_shape]
            all_sampled_observations.append(sampled_obs)

        # Stack all sampled observations from all agents in the cluster
        # [num_agents * actual_samples, *obs_shape]
        pooled_observations = torch.cat(all_sampled_observations, dim=0)

        pooled_obs_np = pooled_observations.cpu().numpy()
        if pooled_obs_np.ndim != 2:
            raise ValueError("Expected 1D observations; pooled_observations must be 2D [N, D].")

        obs_distances = pairwise_distances(pooled_obs_np, metric='euclidean')
        obs_rdm = rsa.rdm.RDMs(obs_distances[np.newaxis, :, :])  # [1, N, N]

        cluster_pools[cluster_idx] = {
            'pooled_observations': pooled_observations,
            'obs_rdm': obs_rdm,
        }
        results['cluster_agents'][cluster_idx] = cluster_agent_ids

    # 3. Evaluate each agent on every cluster pool
    for cluster_idx, pool in enumerate(cluster_pools):
        if pool is None:
            continue

        pooled_observations = pool['pooled_observations']
        obs_rdm = pool['obs_rdm']

        for agent_id in range(num_agents):
            agent = society.agents[agent_id]
            with torch.no_grad():
                latents_dist = agent.encoder(pooled_observations.to(agent.device))
                latents = latents_dist.mean  # Use mean of distribution

            latents_np = latents.cpu().numpy()
            latent_distances = pairwise_distances(latents_np, metric='euclidean')
            latent_rdm = rsa.rdm.RDMs(latent_distances[np.newaxis, :, :])  # [1, N, N]

            rsa_value = rsa.rdm.compare(latent_rdm, obs_rdm, method='rho-a')[0, 0]
            cluster_agent_rsa[cluster_idx, agent_id] = float(rsa_value)

    # 4. Fill agent_rsa and cluster_rsa for within-cluster evaluation
    for agent_id in range(num_agents):
        cluster_idx = agent_to_cluster.get(agent_id)
        if cluster_idx is None:
            continue
        agent_rsa[agent_id] = cluster_agent_rsa[cluster_idx, agent_id]

    for cluster_idx, cluster_agent_ids in enumerate(clusters):
        if not cluster_agent_ids:
            continue
        cluster_values = cluster_agent_rsa[cluster_idx, cluster_agent_ids]
        if np.all(np.isnan(cluster_values)):
            continue
        cluster_rsa[cluster_idx] = float(np.nanmean(cluster_values))

    # 5. Compute cluster-cluster RSA matrix
    for cluster_i, cluster_agent_ids in enumerate(clusters):
        if not cluster_agent_ids:
            continue
        for cluster_j in range(num_clusters):
            values = cluster_agent_rsa[cluster_j, cluster_agent_ids]
            if np.all(np.isnan(values)):
                continue
            cluster_cluster_rsa[cluster_i, cluster_j] = float(np.nanmean(values))
    
    return results


def compute_fgw_similarity_vectorized(rdm_all: torch.Tensor, M_all: torch.Tensor, alpha=0.5):
    """
    Compute Fused GW similarity with pre-computed matrices
    
    Args:
        rdm_all: [A, R, R] internal distance matrices for all agents
        M_all: [A, A, R, R] feature distance matrices for all agent pairs
        alpha: feature weight (0=structure only, 1=feature only)
    """
    A, R, _ = rdm_all.shape
    similarity_matrix = np.zeros((A, A))
    
    # Convert to numpy once
    C_all = rdm_all.cpu().numpy()  # [A, R, R]
    M_all_np = M_all.cpu().numpy()  # [A, A, R, R]
    
    # Uniform distribution
    p = ot.unif(R)
    q = ot.unif(R)
    
    # Only need to compute upper triangle
    for i in range(A):
        for j in range(i, A):
            C_X = C_all[i]  # [R, R]
            C_Y = C_all[j]  # [R, R]
            M = M_all_np[i, j]  # [R, R]
            
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
                        M, C_X, C_Y, p, q, alpha=alpha, log=False
                    )
            
            # Convert to similarity
            similarity = 1.0 / (1.0 + fgw_dist)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix


def compute_fgw_negative_distance_vectorized(
    rdm_all: torch.Tensor,
    M_all: torch.Tensor,
    alpha: float = 0.5,
):
    """
    Compute negative FGW distance with pre-computed matrices.

    Args:
        rdm_all: [A, R, R] internal distance matrices for all agents
        M_all: [A, A, R, R] feature distance matrices for all agent pairs
        alpha: feature weight (0=structure only, 1=feature only)
    """
    A, R, _ = rdm_all.shape
    distance_matrix = np.zeros((A, A))

    # Convert to numpy once
    C_all = rdm_all.cpu().numpy()  # [A, R, R]
    M_all_np = M_all.cpu().numpy()  # [A, A, R, R]

    # Uniform distribution
    p = ot.unif(R)
    q = ot.unif(R)

    # Only need to compute upper triangle
    for i in range(A):
        for j in range(i, A):
            C_X = C_all[i]  # [R, R]
            C_Y = C_all[j]  # [R, R]
            M = M_all_np[i, j]  # [R, R]

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
                        M, C_X, C_Y, p, q, alpha=alpha, log=False
                    )

            distance = -float(fgw_dist)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


# MARK: - Transfer Entropy (TE) Estimation
def estimate_te_gaussian(source, target, lag=1):
    """
    Estimate Transfer Entropy using Gaussian approximation.
    TE(S -> T) = H(T_curr | T_past) - H(T_curr | T_past, S_past)
    
    Args:
        source (np.ndarray): (n_samples, 2) array.
        target (np.ndarray): (n_samples, 2) array.
        lag (int): The lag to test.
    """
    n = len(source)
    if n <= lag + 1:
        return 0.0

    # Current state of target
    t_curr = target[lag:]
    # Past state of target
    t_past = target[:-lag]
    # Past state of source
    s_past = source[:-lag]

    def joint_entropy_gauss(arrays):
        # Concatenate arrays column-wise to form a joint distribution
        combined = np.hstack(arrays)
        cov = np.cov(combined, rowvar=False)
        # Entropy of multivariate Gaussian: 0.5 * log(det(2*pi*e*Cov))
        # Since we take the difference, constants cancel out. 
        # We use 0.5 * log(det(Cov)) as a proxy.
        sign, logdet = np.linalg.slogdet(cov)
        return 0.5 * logdet if sign > 0 else 0.0

    # H(T_curr | T_past) = H(T_curr, T_past) - H(T_past)
    h_tcurr_tpast = joint_entropy_gauss([t_curr, t_past]) - joint_entropy_gauss([t_past])
    
    # H(T_curr | T_past, S_past) = H(T_curr, T_past, S_past) - H(T_past, S_past)
    h_tcurr_tpast_spast = joint_entropy_gauss([t_curr, t_past, s_past]) - joint_entropy_gauss([t_past, s_past])
    
    # Transfer Entropy
    te = h_tcurr_tpast - h_tcurr_tpast_spast
    return max(0, te)


def estimate_te_ksg(source, target, lag=1, k=3):
    """
    Estimate Transfer Entropy using KSG (KNN-based) estimator.
    TE(S -> T) = I(T_curr ; S_past | T_past)
    """
    n_samples = len(source)
    if n_samples <= lag:
        return 0.0

    # Prepare lagged vectors
    t_curr = target[lag:]      # T_t
    t_past = target[:-lag]     # T_{t-lag}
    s_past = source[:-lag]     # S_{t-lag}
    
    def estimate_cmi(X, Y, Z, k):
        """Estimate Conditional Mutual Information I(X;Y|Z) using KSG algorithm."""
        # Ensure all inputs are 2D for concatenation
        n = X.shape[0]
        xyz = np.hstack([X, Y, Z])
        xz = np.hstack([X, Z])
        yz = np.hstack([Y, Z])
        zz = Z # Z is already 2D (n_samples, 2)
        
        # 1. Find distance to k-th neighbor in the full joint space (XYZ)
        tree_xyz = KDTree(xyz)
        # p=np.inf denotes Chebychev distance
        eps, _ = tree_xyz.query(xyz, k=k+1, p=np.inf)
        eps = eps[:, -1] + 1e-15 # Use distance to the k-th neighbor
        
        # 2. Count points within epsilon in marginal spaces
        def count_within(data, eps):
            tree = KDTree(data)
            # We subtract 1 to exclude the point itself
            # query_ball_point is used to count points strictly within distance eps
            counts = tree.query_ball_point(data, eps - 1e-15, p=np.inf)
            return np.array([len(c) for c in counts])

        n_xz = count_within(xz, eps)
        n_yz = count_within(yz, eps)
        n_z = count_within(zz, eps)
        
        # 3. KSG formula for CMI
        # CMI(X;Y|Z) = digamma(k) + <digamma(n_z) - digamma(n_xz) - digamma(n_yz)>
        # Note: Depending on the variant, it's often digamma(n+1)
        cmi = digamma(k) + np.mean(digamma(n_z) - digamma(n_xz) - digamma(n_yz))
        return max(0, cmi)

    return estimate_cmi(t_curr, s_past, t_past, k)


def analyze_sliding_window_te(X, Y, lag=2, window_size=300, step=20, approach='ksg'):
    """
    XからYへのTransfer Entropyをスライディングウィンドウで計算します.
    """
    n_samples = len(X)
    te_history = []
    time_indices = []

    # ウィンドウをスライドさせながら各区間でTEを推定
    for start in range(0, n_samples - window_size, step):
        end = start + window_size
        X_slice = X[start:end]
        Y_slice = Y[start:end]
        
        if approach == 'gaussian':
            te_val = estimate_te_gaussian(X_slice, Y_slice, lag=lag)
        else:
            te_val = estimate_te_ksg(X_slice, Y_slice, lag=lag, k=4)
        
        te_history.append(te_val)
        # ウィンドウの中心をプロット用の時間軸とする
        time_indices.append(start + window_size // 2)
        
    return np.array(time_indices), np.array(te_history)


# MARK: - CCM (Convergent Cross Mapping) for Causality Analysis
def embed_series(series, embedding_dim, lag):
    """
    時系列を遅延埋め込みにより多次元アトラクタに変換します。
    """
    n = len(series)
    max_start = n - (embedding_dim - 1) * lag
    if max_start <= 0:
        return np.empty((0, embedding_dim))
    X = np.array([
        series[i * lag: max_start + i * lag]
        for i in range(embedding_dim)
    ])
    return X.T

def calculate_ccm(shadow_m, target_series, k=None):
    """
    シャドウ・アトラクタ shadow_m から target_series を類推（Cross Mapping）します。
    """
    n_samples = len(shadow_m)
    if k is None:
        k = shadow_m.shape[1] + 1  # 埋め込み次元 + 1 が一般的 [cite: 100]

    tree = KDTree(shadow_m)
    distances, indices = tree.query(shadow_m, k=k+1)
    
    # 自己参照を避けるため1番近い点（自分）を除外
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # 指数重み付けによる類推
    weights = np.exp(-distances / (np.min(distances, axis=1, keepdims=True) + 1e-10))
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    predicted = np.array([np.sum(target_series[indices[i]] * weights[i]) for i in range(n_samples)])
    
    # [cite_start]実際の値と類推値の相関係数を返す [cite: 106]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.corrcoef(predicted, target_series)[0, 1]

def analyze_sliding_window_ccm(X, Y, window_size=500, step=50, dim=3, lag=1):
    """
    スライディングウィンドウでCCMを実行し、X->Yの因果を解析します。
    """
    times = []
    rho_yx = [] # YでXを類推 (X -> Y の因果)

    for start in range(0, len(X) - window_size, step):
        end = start + window_size
        xs, ys = X[start:end, 0], Y[start:end, 0] # 2次元のうち1次元を使用
        
        # [cite_start]状態空間再構成 [cite: 99]
        my = embed_series(ys, dim, lag)
        
        # ターゲット系列の長さを合わせる
        target_x = xs[(dim-1)*lag:]
        
        rho_yx.append(calculate_ccm(my, target_x))
        
        times.append(start + window_size // 2)
        
    return np.array(times), np.array(rho_yx)

import numpy as np
from numpy.linalg import norm
from typing import Tuple

def similarity_matrix(x_q: np.ndarray, X_set: np.ndarray) -> np.ndarray:
    """计算余弦相似度

    Args:
        xq: A query vector (1d ndarray)
        index: A set of vectors.

    """

    index_norm = norm(X_set, axis=1)
    xq_norm = norm(x_q.T)
    sim = np.dot(X_set, x_q.T) / (index_norm * xq_norm)
    return sim



def top_scores(sim: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    # 获取top k 的得分和索引
    
    
    top_k = min(top_k, sim.shape[0])
    idx = np.argpartition(sim, -top_k)[-top_k:]
    scores = sim[idx]

    return scores, idx
"""
Modulo de clustering de productos por afinidad de compra.

Calcula la afinidad entre productos usando el indice de Jaccard sobre
el historial de pedidos, y aplica K-Medoids para agrupar productos que
suelen pedirse juntos. El calculo esta vectorizado con NumPy/Pandas para
escalar a catalogos de miles de productos.
"""

import numpy as np
import pandas as pd
import kmedoids


def build_affinity_matrix(orders: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
    """
    Calcula la afinidad entre cada par de SKUs usando el indice de
    Jaccard, que normaliza por la frecuencia individual de cada
    producto para evitar que productos muy vendidos parezcan afines a
    todo solo por aparecer en muchos pedidos.
    """
    presence = pd.crosstab(orders["order_id"], orders["sku"])
    presence = presence.reindex(columns=skus, fill_value=0)
    presence_binary = (presence > 0).astype(int)

    co_occurrence = presence_binary.T.dot(presence_binary).astype(float)
    total_orders_per_sku = presence_binary.sum(axis=0)

    union = total_orders_per_sku.values[:, None] + total_orders_per_sku.values[None, :] - co_occurrence.values
    union[union == 0] = 1

    jaccard = co_occurrence.values / union
    np.fill_diagonal(jaccard, 0)

    return pd.DataFrame(jaccard, index=skus, columns=skus)


def affinity_to_distance(affinity: pd.DataFrame) -> np.ndarray:
    """Convierte afinidad (mayor = mas relacionado) en distancia para K-Medoids."""
    max_val = affinity.values.max()
    if max_val == 0:
        max_val = 1
    distance = 1 - (affinity.values / max_val)
    np.fill_diagonal(distance, 0)
    return distance


def cluster_products(distance_matrix: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Aplica K-Medoids y retorna el cluster asignado a cada producto."""
    result = kmedoids.KMedoids(n_clusters=n_clusters, method="fasterpam", random_state=random_state).fit(distance_matrix)
    return result.labels_


def silhouette_by_k(distance_matrix: np.ndarray, k_range: range, random_state: int = 42) -> dict[int, float]:
    """Calcula el silhouette score para distintos valores de k."""
    from sklearn.metrics import silhouette_score
    scores = {}
    for k in k_range:
        labels = cluster_products(distance_matrix, k, random_state)
        scores[k] = silhouette_score(distance_matrix, labels, metric="precomputed")
    return scores
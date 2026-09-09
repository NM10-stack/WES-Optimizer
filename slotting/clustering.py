"""
Modulo de clustering de productos por afinidad de compra.

Construye una matriz de distancia entre productos a partir del historial
de pedidos, y aplica K-Medoids para agrupar productos que suelen pedirse
juntos. Cada cluster resultante representa un grupo de productos candidatos
a ubicarse fisicamente cerca entre si.
"""

import numpy as np
import pandas as pd
import kmedoids


def build_affinity_matrix(orders: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
    """
    Calcula la afinidad entre cada par de SKUs usando el indice de
    Jaccard, que normaliza por la frecuencia individual de cada
    producto. Esto evita que productos muy populares parezcan
    'afines' a todo simplemente por aparecer en muchos pedidos,
    haciendo la metrica comparable entre datasets de distinto tamano
    y distribucion.
    """
    presence = pd.crosstab(orders["order_id"], orders["sku"])
    presence = presence.reindex(columns=skus, fill_value=0)
    presence_binary = (presence > 0).astype(int)

    co_occurrence = presence_binary.T.dot(presence_binary).astype(float)
    order_count = presence_binary.sum(axis=0)

    union = order_count.values[:, None] + order_count.values[None, :] - co_occurrence.values
    union[union == 0] = 1

    jaccard = co_occurrence.values / union
    np.fill_diagonal(jaccard, 0)

    return pd.DataFrame(jaccard, index=skus, columns=skus)


def affinity_to_distance(affinity: pd.DataFrame) -> np.ndarray:
    """
    Convierte la matriz de afinidad (mayor valor = mas relacionados) en
    una matriz de distancia (menor valor = mas cercanos), normalizada
    entre 0 y 1, apta para K-Medoids.
    """
    max_val = affinity.values.max()
    if max_val == 0:
        max_val = 1

    distance = 1 - (affinity.values / max_val)
    np.fill_diagonal(distance, 0)

    return distance


def cluster_products(
    distance_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Aplica K-Medoids sobre la matriz de distancia y retorna un arreglo
    con el indice de cluster asignado a cada producto, en el mismo orden
    en que aparecen en la matriz.
    """
    result = kmedoids.KMedoids(
        n_clusters=n_clusters,
        method="fasterpam",
        random_state=random_state,
    ).fit(distance_matrix)

    return result.labels_


def silhouette_by_k(
    distance_matrix: np.ndarray,
    k_range: range,
    random_state: int = 42,
) -> dict[int, float]:
    """
    Calcula el silhouette score para distintos valores de k, usando
    distancias precomputadas. Sirve para elegir el numero de clusters
    que mejor separa los datos: valores mas cercanos a 1 son mejores,
    cercanos a 0 indican clusters solapados, negativos indican una mala
    asignacion.
    """
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        labels = cluster_products(distance_matrix, k, random_state)
        scores[k] = silhouette_score(distance_matrix, labels, metric="precomputed")

    return scores


if __name__ == "__main__":
    products = pd.read_csv("../data/synthetic/products.csv")
    orders = pd.read_csv("../data/synthetic/orders.csv")

    skus = products["sku"].tolist()

    affinity = build_affinity_matrix(orders, skus)
    distance = affinity_to_distance(affinity)

    scores = silhouette_by_k(distance, range(4, 13))
    best_k = max(scores, key=scores.get)

    print("Silhouette score por numero de clusters:")
    for k, score in scores.items():
        print(f"  k={k}: {score:.4f}")
    print(f"\nMejor k encontrado: {best_k}")

    labels = cluster_products(distance, best_k)
    products["cluster"] = labels

    products.to_csv("../data/processed/products_clustered.csv", index=False)
    print(f"\nGuardado en data/processed/products_clustered.csv")
    print(products["cluster"].value_counts().sort_index())
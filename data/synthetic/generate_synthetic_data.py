"""
Generador de datos sinteticos para desarrollo mientras llega el dataset real de STG.

Genera:
- products.csv: catalogo de SKUs con posicion actual en el almacen
- orders.csv: historial de pedidos (para calcular afinidad de compra)

Uso:
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PRODUCTS = 200
N_ORDERS = 2000
MAX_ITEMS_PER_ORDER = 6
WAREHOUSE_AISLES = 10
WAREHOUSE_SLOTS_PER_AISLE = 20


def generate_products(n_products: int) -> pd.DataFrame:
    """Catalogo de productos con su ubicacion actual (heuristica alfanumerica)."""
    skus = [f"SKU-{i:04d}" for i in range(n_products)]
    aisle = np.random.randint(1, WAREHOUSE_AISLES + 1, size=n_products)
    slot = np.random.randint(1, WAREHOUSE_SLOTS_PER_AISLE + 1, size=n_products)
    weight_kg = np.round(np.random.exponential(scale=2.0, size=n_products) + 0.1, 2)
    volume_m3 = np.round(np.random.exponential(scale=0.05, size=n_products) + 0.001, 4)

    return pd.DataFrame({
        "sku": skus,
        "aisle": aisle,
        "slot": slot,
        "weight_kg": weight_kg,
        "volume_m3": volume_m3,
    })


def generate_orders(n_orders: int, products: pd.DataFrame, noise_ratio: float = 0.35) -> pd.DataFrame:
    """
    Genera pedidos con una mezcla de productos del mismo cluster de
    afinidad (mayoria) y productos de otros clusters elegidos al azar
    (noise_ratio), simulando que en la practica los pedidos rara vez son
    100% homogeneos en afinidad de compra.
    """
    n_products = len(products)
    n_hidden_clusters = 8
    cluster_of_sku = np.random.randint(0, n_hidden_clusters, size=n_products)

    rows = []
    for order_id in range(n_orders):
        n_items = np.random.randint(1, MAX_ITEMS_PER_ORDER + 1)
        base_cluster = np.random.randint(0, n_hidden_clusters)
        candidate_idxs = np.where(cluster_of_sku == base_cluster)[0]

        n_noise_items = int(round(n_items * noise_ratio))
        n_affinity_items = n_items - n_noise_items

        affinity_pool = candidate_idxs if len(candidate_idxs) >= n_affinity_items else np.arange(n_products)
        affinity_chosen = np.random.choice(affinity_pool, size=n_affinity_items, replace=False)

        remaining_pool = np.setdiff1d(np.arange(n_products), affinity_chosen)
        noise_chosen = np.random.choice(remaining_pool, size=min(n_noise_items, len(remaining_pool)), replace=False)

        chosen = np.concatenate([affinity_chosen, noise_chosen])

        for sku_idx in chosen:
            rows.append({
                "order_id": f"ORD-{order_id:05d}",
                "sku": products.iloc[sku_idx]["sku"],
                "quantity": np.random.randint(1, 5),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    products_df = generate_products(N_PRODUCTS)
    orders_df = generate_orders(N_ORDERS, products_df)

    products_df.to_csv("products.csv", index=False)
    orders_df.to_csv("orders.csv", index=False)

    print(f"Generados {len(products_df)} productos y {len(orders_df)} lineas de pedido")
    print("Archivos: products.csv, orders.csv")
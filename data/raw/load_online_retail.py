"""
Carga y adapta el dataset publico Online Retail (UCI/Kaggle) al formato
utilizado por el pipeline de slotting (order_id, sku, quantity), y genera
posiciones fisicas iniciales aleatorias, ya que ningun dataset publico
incluye la ubicacion real dentro de una bodega.
"""

import numpy as np
import pandas as pd

AISLE_RANGE = (1, 10)
SLOT_RANGE = (1, 20)


def load_orders(raw_path: str) -> pd.DataFrame:
    """
    Lee el archivo de Online Retail y lo traduce al formato esperado por
    el pipeline: order_id, sku, quantity. Descarta filas sin SKU valido y
    las cancelaciones (facturas que empiezan con 'C', segun la
    documentacion del dataset).
    """
    raw = pd.read_excel(raw_path)

    raw = raw.dropna(subset=["StockCode", "InvoiceNo"])
    raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]
    raw = raw[raw["Quantity"] > 0]

    orders = raw.rename(columns={
        "InvoiceNo": "order_id",
        "StockCode": "sku",
        "Quantity": "quantity",
    })[["order_id", "sku", "quantity"]]

    return orders.reset_index(drop=True)


def build_products(orders: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Construye el catalogo de productos a partir de los SKUs unicos
    presentes en las ordenes, asignando una posicion fisica inicial
    aleatoria (situacion no optimizada), ya que el dataset publico no
    incluye informacion de layout de bodega.
    """
    rng = np.random.default_rng(random_state)
    skus = orders["sku"].unique()

    products = pd.DataFrame({
        "sku": skus,
        "aisle": rng.integers(AISLE_RANGE[0], AISLE_RANGE[1] + 1, size=len(skus)),
        "slot": rng.integers(SLOT_RANGE[0], SLOT_RANGE[1] + 1, size=len(skus)),
    })

    return products


def sample_orders(orders: pd.DataFrame, n_orders: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Extrae una muestra aleatoria de pedidos completos (no de lineas
    sueltas) para acelerar las corridas de desarrollo del Algoritmo
    Genetico, cuyo tiempo de ejecucion escala con la cantidad de pedidos
    evaluados en la funcion de fitness. El dataset completo se conserva
    aparte para una validacion final.
    """
    unique_ids = orders["order_id"].unique()
    rng = np.random.default_rng(random_state)
    sampled_ids = rng.choice(unique_ids, size=min(n_orders, len(unique_ids)), replace=False)

    return orders[orders["order_id"].isin(sampled_ids)].reset_index(drop=True)


if __name__ == "__main__":
    orders = load_orders("Online_Retail.xlsx")
    products = build_products(orders)

    orders.to_csv("orders_real_full.csv", index=False)
    products.to_csv("products_real.csv", index=False)

    orders_sample = sample_orders(orders)
    orders_sample.to_csv("orders_real_sample.csv", index=False)

    print(f"Dataset completo -> Ordenes: {orders['order_id'].nunique()}, Lineas: {len(orders)}")
    print(f"Muestra de desarrollo -> Ordenes: {orders_sample['order_id'].nunique()}, Lineas: {len(orders_sample)}")
    print(f"SKUs unicos: {len(products)}")
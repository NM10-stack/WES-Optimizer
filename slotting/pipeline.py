"""
Pipeline completo de optimizacion de Slotting (version fusionada).

Integra clustering por afinidad de compra (K-Medoids sobre indice de
Jaccard) con un Algoritmo Genetico que asigna posiciones 3D individuales
a cada producto, respetando la capacidad estructural de cada nivel de
estanteria. Calcula la comparacion entre tres situaciones: ubicacion
aleatoria, heuristica de popularidad, y ubicacion optimizada.
"""

import pandas as pd

from clustering import build_affinity_matrix, affinity_to_distance, cluster_products, silhouette_by_k
from genetic_algorithm import (
    build_order_groups,
    compute_warehouse_dimensions,
    build_all_positions,
    run_genetic_algorithm,
    random_baseline_distance,
    popularity_baseline_distance,
    evaluar_ruta_real,
)


def run_slotting_pipeline(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    capacidades_por_nivel: dict[int, float] | None = None,
    k_range: range = range(4, 13),
    num_generations: int = 40,
    poblacion: int = 20,
) -> dict:
    skus = products["sku"].tolist()

    n_aisles, slots_per_aisle, niveles = compute_warehouse_dimensions(len(products))
    all_positions = build_all_positions(n_aisles, slots_per_aisle, niveles)

    affinity = build_affinity_matrix(orders, skus)
    distance_matrix = affinity_to_distance(affinity)
    scores = silhouette_by_k(distance_matrix, k_range)
    best_k = max(scores, key=scores.get)
    products = products.copy()
    products["cluster"] = cluster_products(distance_matrix, best_k)

    order_groups = build_order_groups(orders)

    initial_distance = random_baseline_distance(products, order_groups, all_positions)
    popularity_distance = popularity_baseline_distance(products, orders, order_groups, all_positions)

    mejor_asignacion, historial = run_genetic_algorithm(
        products, order_groups, all_positions, capacidades_por_nivel, num_generations, poblacion
    )

    products["new_aisle"] = products["sku"].map(lambda s: mejor_asignacion[s][0])
    products["new_slot"] = products["sku"].map(lambda s: mejor_asignacion[s][1])
    products["new_nivel"] = products["sku"].map(lambda s: mejor_asignacion[s][2])

    optimized_distance = evaluar_ruta_real(order_groups, mejor_asignacion)

    mejora_vs_aleatorio = (initial_distance - optimized_distance) / initial_distance * 100
    mejora_vs_popularidad = (popularity_distance - optimized_distance) / popularity_distance * 100

    metrics = {
        "silhouette_score": scores[best_k],
        "n_clusters": best_k,
        "n_aisles": n_aisles,
        "slots_per_aisle": slots_per_aisle,
        "niveles_altura": niveles,
        "distancia_inicial_aleatoria": initial_distance,
        "distancia_heuristica_popularidad": popularity_distance,
        "distancia_optimizada": optimized_distance,
        "mejora_vs_aleatorio_pct": mejora_vs_aleatorio,
        "mejora_vs_heuristica_pct": mejora_vs_popularidad,
    }

    return {"products": products, "metrics": metrics, "convergence": historial}


if __name__ == "__main__":
    products = pd.read_csv("../data/synthetic/products.csv")
    orders = pd.read_csv("../data/synthetic/orders.csv")

    result = run_slotting_pipeline(products, orders)

    print("Metricas finales:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")

    result["products"].to_csv("../data/processed/slotting_result.csv", index=False)
    pd.DataFrame([result["metrics"]]).to_csv("../data/processed/baseline_metrics.csv", index=False)
    pd.DataFrame({
        "generacion": range(1, len(result["convergence"]) + 1),
        "distancia": result["convergence"],
    }).to_csv("../data/processed/ga_convergence.csv", index=False)

    print("\nArchivos guardados en data/processed/")
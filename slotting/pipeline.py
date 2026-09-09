"""
Pipeline completo de optimizacion de Slotting.

Integra el modulo de clustering (afinidad de compra -> grupos K-Medoids)
con el modulo de Algoritmo Genetico (asignacion de posiciones fisicas
individuales por producto, agrupadas segun cluster), y calcula la
comparacion entre tres situaciones: ubicacion aleatoria, ubicacion por
heuristica de popularidad, y ubicacion optimizada por el modelo.
"""

import pandas as pd

from clustering import build_affinity_matrix, affinity_to_distance, cluster_products, silhouette_by_k
from genetic_algorithm import (
    build_order_groups,
    baseline_distance,
    random_baseline_distance,
    popularity_baseline_distance,
    total_distance_for_positions,
    run_genetic_algorithm,
    assign_individual_positions,
    compute_warehouse_dimensions,
    build_all_positions,
)


def run_slotting_pipeline(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    k_range: range = range(4, 13),
) -> dict:
    skus = products["sku"].tolist()

    n_aisles, slots_per_aisle = compute_warehouse_dimensions(len(products))
    all_positions = build_all_positions(n_aisles, slots_per_aisle)

    affinity = build_affinity_matrix(orders, skus)
    distance_matrix = affinity_to_distance(affinity)

    scores = silhouette_by_k(distance_matrix, k_range)
    best_k = max(scores, key=scores.get)

    labels = cluster_products(distance_matrix, best_k)
    products = products.copy()
    products["cluster"] = labels

    order_groups = build_order_groups(orders)

    initial_distance = random_baseline_distance(products, order_groups, all_positions)
    popularity_distance = popularity_baseline_distance(products, orders, order_groups, all_positions)

    best_positions, fitness_history = run_genetic_algorithm(
        products, order_groups, best_k, all_positions, n_aisles, slots_per_aisle
    )

    cluster_anchor_positions = {i: tuple(best_positions[i]) for i in range(best_k)}
    sku_to_new_position = assign_individual_positions(products, cluster_anchor_positions, all_positions)

    products["new_aisle"] = products["sku"].map(lambda s: sku_to_new_position[s][0])
    products["new_slot"] = products["sku"].map(lambda s: sku_to_new_position[s][1])

    optimized_distance = total_distance_for_positions(order_groups, sku_to_new_position)

    improvement_pct = (initial_distance - optimized_distance) / initial_distance * 100
    improvement_vs_popularity_pct = (popularity_distance - optimized_distance) / popularity_distance * 100

    metrics = {
        "silhouette_score": scores[best_k],
        "n_clusters": best_k,
        "n_aisles": n_aisles,
        "slots_per_aisle": slots_per_aisle,
        "distancia_inicial": initial_distance,
        "distancia_heuristica_popularidad": popularity_distance,
        "distancia_optimizada": optimized_distance,
        "mejora_porcentual": improvement_pct,
        "mejora_vs_heuristica_pct": improvement_vs_popularity_pct,
    }

    return {"products": products, "metrics": metrics, "convergence": fitness_history}


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
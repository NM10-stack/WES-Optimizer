"""
Pipeline completo de optimizacion de Slotting.

Integra clustering por afinidad de compra (K-Medoids sobre indice de
Jaccard) con un Algoritmo Genetico que asigna posiciones 3D individuales
a cada producto, respetando la capacidad estructural de cada nivel de
estanteria.

Compara cuatro estrategias:

1. Ubicacion aleatoria.
2. Heuristica de popularidad.
3. Algoritmo Genetico puro.
4. K-Medoids + Algoritmo Genetico.

La comparacion se realiza sobre el mismo conjunto de productos,
pedidos, layout de almacen y parametros de optimizacion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clustering import (
    build_affinity_matrix,
    affinity_to_distance,
    cluster_products,
    silhouette_by_k,
)

from .genetic_algorithm import (
    build_order_groups,
    compute_warehouse_dimensions,
    build_all_positions,
    run_genetic_algorithm,
    random_baseline_distance,
    popularity_baseline_distance,
    evaluar_ruta_real,
)


def calcular_mejora(
    distancia_referencia: float,
    distancia_nueva: float,
) -> float:
    """
    Calcula el porcentaje de mejora de una distancia nueva
    respecto de una distancia de referencia.
    """

    if distancia_referencia == 0:
        return 0.0

    return (
        (distancia_referencia - distancia_nueva)
        / distancia_referencia
        * 100
    )


def run_slotting_pipeline(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    capacidades_por_nivel: dict[int, float] | None = None,
    k_range: range = range(4, 13),
    num_generations: int = 40,
    poblacion: int = 20,
    random_state: int = 42,
    proporcion_guiada: float = 0.5,
) -> dict:
    """
    Ejecuta la validacion completa del modulo de Slotting.

    Compara:

    - baseline aleatorio;
    - baseline por popularidad;
    - GA puro;
    - K-Medoids + GA.

    GA puro:
        proporcion_guiada = 0.0

    K-Medoids + GA:
        proporcion_guiada = 0.5 por defecto.

    Esto permite evaluar especificamente si el clustering aporta
    beneficios respecto del Algoritmo Genetico por si solo.
    """

    # =========================================================
    # COPIA DE SEGURIDAD DEL DATAFRAME
    # =========================================================

    products = products.copy()

    skus = products["sku"].tolist()

    # =========================================================
    # CONSTRUCCION DEL LAYOUT DEL ALMACEN
    # =========================================================

    n_aisles, slots_per_aisle, niveles = (
        compute_warehouse_dimensions(
            len(products)
        )
    )

    all_positions = build_all_positions(
        n_aisles,
        slots_per_aisle,
        niveles,
    )

    # =========================================================
    # CLUSTERING K-MEDOIDS
    # =========================================================

    affinity = build_affinity_matrix(
        orders,
        skus,
    )

    distance_matrix = affinity_to_distance(
        affinity
    )

    scores = silhouette_by_k(
        distance_matrix,
        k_range,
    )

    if not scores:
        raise ValueError(
            "No fue posible calcular Silhouette Score "
            "para ningun valor de K."
        )

    best_k = max(
        scores,
        key=scores.get,
    )

    products["cluster"] = cluster_products(
        distance_matrix,
        best_k,
    )

    # =========================================================
    # AGRUPACION HISTORICA DE PEDIDOS
    # =========================================================

    order_groups = build_order_groups(
        orders
    )

    # =========================================================
    # BASELINE 1: UBICACION ALEATORIA
    # =========================================================

    random_distance = random_baseline_distance(
        products,
        order_groups,
        all_positions,
        random_state=random_state,
    )

    # =========================================================
    # BASELINE 2: HEURISTICA DE POPULARIDAD
    # =========================================================

    popularity_distance = popularity_baseline_distance(
        products,
        orders,
        order_groups,
        all_positions,
    )

    # =========================================================
    # MODELO 3: ALGORITMO GENETICO PURO
    # =========================================================

    asignacion_ga_puro, historial_ga_puro = (
        run_genetic_algorithm(
            products=products,
            order_groups=order_groups,
            all_positions=all_positions,
            capacidades_por_nivel=capacidades_por_nivel,
            num_generations=num_generations,
            poblacion=poblacion,
            random_state=random_state,
            proporcion_guiada=0.0,
        )
    )

    distancia_ga_puro = evaluar_ruta_real(
        order_groups,
        asignacion_ga_puro,
    )

    # =========================================================
    # MODELO 4: K-MEDOIDS + ALGORITMO GENETICO
    # =========================================================

    asignacion_kmedoids_ga, historial_kmedoids_ga = (
        run_genetic_algorithm(
            products=products,
            order_groups=order_groups,
            all_positions=all_positions,
            capacidades_por_nivel=capacidades_por_nivel,
            num_generations=num_generations,
            poblacion=poblacion,
            random_state=random_state,
            proporcion_guiada=proporcion_guiada,
        )
    )

    distancia_kmedoids_ga = evaluar_ruta_real(
        order_groups,
        asignacion_kmedoids_ga,
    )

    # =========================================================
    # RESULTADO DE K-MEDOIDS + GA
    # =========================================================

    products_kmedoids_ga = products.copy()

    products_kmedoids_ga["new_aisle"] = (
        products_kmedoids_ga["sku"].map(
            lambda sku: asignacion_kmedoids_ga[sku][0]
        )
    )

    products_kmedoids_ga["new_slot"] = (
        products_kmedoids_ga["sku"].map(
            lambda sku: asignacion_kmedoids_ga[sku][1]
        )
    )

    products_kmedoids_ga["new_nivel"] = (
        products_kmedoids_ga["sku"].map(
            lambda sku: asignacion_kmedoids_ga[sku][2]
        )
    )

    # =========================================================
    # RESULTADO DEL GA PURO
    # =========================================================

    products_ga_puro = products.copy()

    products_ga_puro["new_aisle"] = (
        products_ga_puro["sku"].map(
            lambda sku: asignacion_ga_puro[sku][0]
        )
    )

    products_ga_puro["new_slot"] = (
        products_ga_puro["sku"].map(
            lambda sku: asignacion_ga_puro[sku][1]
        )
    )

    products_ga_puro["new_nivel"] = (
        products_ga_puro["sku"].map(
            lambda sku: asignacion_ga_puro[sku][2]
        )
    )

    # =========================================================
    # PORCENTAJES DE MEJORA
    # =========================================================

    mejora_ga_puro_vs_aleatorio = calcular_mejora(
        random_distance,
        distancia_ga_puro,
    )

    mejora_ga_puro_vs_popularidad = calcular_mejora(
        popularity_distance,
        distancia_ga_puro,
    )

    mejora_kmedoids_vs_aleatorio = calcular_mejora(
        random_distance,
        distancia_kmedoids_ga,
    )

    mejora_kmedoids_vs_popularidad = calcular_mejora(
        popularity_distance,
        distancia_kmedoids_ga,
    )

    mejora_kmedoids_vs_ga_puro = calcular_mejora(
        distancia_ga_puro,
        distancia_kmedoids_ga,
    )

    # =========================================================
    # METRICAS FINALES
    # =========================================================

    metrics = {
        # Clustering
        "silhouette_score": scores[best_k],
        "n_clusters": best_k,

        # Layout
        "n_aisles": n_aisles,
        "slots_per_aisle": slots_per_aisle,
        "niveles_altura": niveles,

        # Parametros experimentales
        "num_generations": num_generations,
        "poblacion": poblacion,
        "random_state": random_state,
        "proporcion_guiada": proporcion_guiada,

        # Distancias
        "distancia_aleatoria": random_distance,
        "distancia_popularidad": popularity_distance,
        "distancia_ga_puro": distancia_ga_puro,
        "distancia_kmedoids_ga": distancia_kmedoids_ga,

        # Mejoras GA puro
        "mejora_ga_puro_vs_aleatorio_pct":
            mejora_ga_puro_vs_aleatorio,

        "mejora_ga_puro_vs_popularidad_pct":
            mejora_ga_puro_vs_popularidad,

        # Mejoras K-Medoids + GA
        "mejora_kmedoids_ga_vs_aleatorio_pct":
            mejora_kmedoids_vs_aleatorio,

        "mejora_kmedoids_ga_vs_popularidad_pct":
            mejora_kmedoids_vs_popularidad,

        "mejora_kmedoids_ga_vs_ga_puro_pct":
            mejora_kmedoids_vs_ga_puro,
    }

    # =========================================================
    # RESULTADO DEL PIPELINE
    # =========================================================

    return {
        "products": products_kmedoids_ga,
        "products_ga_puro": products_ga_puro,
        "metrics": metrics,
        "convergence_ga_puro": historial_ga_puro,
        "convergence_kmedoids_ga": historial_kmedoids_ga,
    }


if __name__ == "__main__":

    # =========================================================
    # RUTAS DEL PROYECTO
    # =========================================================

    BASE_DIR = Path(__file__).resolve().parents[1]

    DATA_DIR = BASE_DIR / "data"

    SYNTHETIC_DIR = (
        DATA_DIR
        / "synthetic"
    )

    PROCESSED_DIR = (
        DATA_DIR
        / "processed"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # CARGA DE DATOS
    # =========================================================

    products_path = (
        SYNTHETIC_DIR
        / "products.csv"
    )

    orders_path = (
        SYNTHETIC_DIR
        / "orders.csv"
    )

    if not products_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: "
            f"{products_path}"
        )

    if not orders_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: "
            f"{orders_path}"
        )

    products = pd.read_csv(
        products_path
    )

    orders = pd.read_csv(
        orders_path
    )

    # =========================================================
    # EJECUCION DEL EXPERIMENTO
    # =========================================================

    result = run_slotting_pipeline(
        products=products,
        orders=orders,
        num_generations=40,
        poblacion=20,
        random_state=42,
        proporcion_guiada=0.5,
    )

    # =========================================================
    # RESULTADOS EN TERMINAL
    # =========================================================

    metrics = result["metrics"]

    print()
    print("======================================")
    print("       COMPARACION SLOTTING")
    print("======================================")

    print()
    print("--- CLUSTERING ---")

    print(
        f"K seleccionado: "
        f"{metrics['n_clusters']}"
    )

    print(
        f"Silhouette Score: "
        f"{metrics['silhouette_score']:.4f}"
    )

    print()
    print("--- LAYOUT ---")

    print(
        f"Pasillos: "
        f"{metrics['n_aisles']}"
    )

    print(
        f"Slots por pasillo: "
        f"{metrics['slots_per_aisle']}"
    )

    print(
        f"Niveles: "
        f"{metrics['niveles_altura']}"
    )

    print()
    print("--- PARAMETROS ---")

    print(
        f"Generaciones: "
        f"{metrics['num_generations']}"
    )

    print(
        f"Poblacion: "
        f"{metrics['poblacion']}"
    )

    print(
        f"Semilla: "
        f"{metrics['random_state']}"
    )

    print(
        f"Proporcion guiada K-Medoids: "
        f"{metrics['proporcion_guiada']:.2f}"
    )

    print()
    print("--- DISTANCIAS ---")

    print(
        f"Aleatorio: "
        f"{metrics['distancia_aleatoria']:.2f}"
    )

    print(
        f"Popularidad: "
        f"{metrics['distancia_popularidad']:.2f}"
    )

    print(
        f"GA puro: "
        f"{metrics['distancia_ga_puro']:.2f}"
    )

    print(
        f"K-Medoids + GA: "
        f"{metrics['distancia_kmedoids_ga']:.2f}"
    )

    print()
    print("--- MEJORAS GA PURO ---")

    print(
        f"vs Aleatorio: "
        f"{metrics['mejora_ga_puro_vs_aleatorio_pct']:.2f}%"
    )

    print(
        f"vs Popularidad: "
        f"{metrics['mejora_ga_puro_vs_popularidad_pct']:.2f}%"
    )

    print()
    print("--- MEJORAS K-MEDOIDS + GA ---")

    print(
        f"vs Aleatorio: "
        f"{metrics['mejora_kmedoids_ga_vs_aleatorio_pct']:.2f}%"
    )

    print(
        f"vs Popularidad: "
        f"{metrics['mejora_kmedoids_ga_vs_popularidad_pct']:.2f}%"
    )

    print(
        f"vs GA puro: "
        f"{metrics['mejora_kmedoids_ga_vs_ga_puro_pct']:.2f}%"
    )

    # =========================================================
    # GUARDAR RESULTADOS
    # =========================================================

    result["products"].to_csv(
        PROCESSED_DIR
        / "slotting_result_kmedoids_ga.csv",
        index=False,
    )

    result["products_ga_puro"].to_csv(
        PROCESSED_DIR
        / "slotting_result_ga_puro.csv",
        index=False,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        PROCESSED_DIR
        / "slotting_comparison_metrics.csv",
        index=False,
    )

    # =========================================================
    # CONVERGENCIA GA PURO
    # =========================================================

    pd.DataFrame(
        {
            "generacion": range(
                1,
                len(
                    result[
                        "convergence_ga_puro"
                    ]
                )
                + 1,
            ),

            "distancia": result[
                "convergence_ga_puro"
            ],
        }
    ).to_csv(
        PROCESSED_DIR
        / "ga_puro_convergence.csv",
        index=False,
    )

    # =========================================================
    # CONVERGENCIA K-MEDOIDS + GA
    # =========================================================

    pd.DataFrame(
        {
            "generacion": range(
                1,
                len(
                    result[
                        "convergence_kmedoids_ga"
                    ]
                )
                + 1,
            ),

            "distancia": result[
                "convergence_kmedoids_ga"
            ],
        }
    ).to_csv(
        PROCESSED_DIR
        / "kmedoids_ga_convergence.csv",
        index=False,
    )

    print()
    print("======================================")
    print("EXPERIMENTO FINALIZADO")
    print("======================================")

    print(
        "\nArchivos guardados en:"
    )

    print(PROCESSED_DIR)
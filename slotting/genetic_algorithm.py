"""
Modulo de optimizacion de posiciones fisicas mediante Algoritmo Genetico.

Toma los clusters de afinidad generados por clustering.py y determina la
mejor posicion (pasillo, slot) para cada cluster. A cada producto se le
asigna una posicion individual y distinta, ubicada dentro del bloque de
posiciones mas cercano al ancla de su cluster: esto evita que productos
de un mismo cluster compartan coordenada exacta (fisicamente imposible
en un almacen real) y evita que el resultado se vea artificialmente
perfecto solo por compartir posicion.

El tamano del almacen (numero de pasillos) se calcula dinamicamente segun
la cantidad de productos a ubicar, para que el layout siempre tenga
capacidad suficiente sin importar el tamano del dataset utilizado.
"""

import numpy as np
import pandas as pd
import pygad

SLOTS_PER_AISLE = 20
MIN_AISLES = 10


def compute_warehouse_dimensions(n_products: int, min_aisles: int = MIN_AISLES) -> tuple[int, int]:
    """
    Calcula un layout de almacen razonablemente cuadrado (pasillos x
    posiciones por pasillo similares entre si) con capacidad suficiente
    para todos los productos. Se evita fijar un numero de posiciones por
    pasillo constante (ej. 20), ya que para catalogos grandes esto
    generaria una cantidad de pasillos poco realista (cientos de
    pasillos muy angostos). En su lugar, ambos ejes escalan juntos segun
    la raiz cuadrada del numero de productos, aproximando un almacen de
    forma mas realista.
    """
    side = int(np.ceil(np.sqrt(n_products)))
    n_aisles = max(side, min_aisles)
    slots_per_aisle = max(side, min_aisles)
    return n_aisles, slots_per_aisle


def build_all_positions(n_aisles: int, slots_per_aisle: int) -> list[tuple[int, int]]:
    """Genera todas las posiciones (pasillo, slot) posibles del almacen."""
    return [(a, s) for a in range(1, n_aisles + 1) for s in range(1, slots_per_aisle + 1)]


def manhattan_distance(pos_a: tuple[float, float], pos_b: tuple[float, float]) -> float:
    """Distancia Manhattan entre dos posiciones (pasillo, slot)."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def build_order_groups(orders: pd.DataFrame) -> list[list[str]]:
    """Agrupa el historial de pedidos en listas de SKUs por order_id."""
    return orders.groupby("order_id")["sku"].apply(list).tolist()


def total_distance_for_positions(
    order_groups: list[list[str]],
    sku_to_position: dict[str, tuple[float, float]],
) -> float:
    """
    Calcula la distancia total recorrida para completar todos los pedidos,
    dado un mapeo sku -> posicion. Suma la distancia entre productos
    consecutivos segun el orden en que aparecen en cada pedido.
    """
    total = 0.0
    for skus in order_groups:
        positions = [sku_to_position[sku] for sku in skus if sku in sku_to_position]
        for i in range(len(positions) - 1):
            total += manhattan_distance(positions[i], positions[i + 1])
    return total


def baseline_distance(products: pd.DataFrame, order_groups: list[list[str]]) -> float:
    """Distancia total usando las ubicaciones actuales (situacion aleatoria)."""
    sku_to_position = {
        row["sku"]: (row["aisle"], row["slot"])
        for _, row in products.iterrows()
    }
    return total_distance_for_positions(order_groups, sku_to_position)

def random_baseline_distance(
    products: pd.DataFrame,
    order_groups: list[list[str]],
    all_positions: list[tuple[int, int]],
    random_state: int = 42,
) -> float:
    """
    Distancia total usando una asignacion aleatoria de posiciones, pero
    calculada sobre el mismo almacen (mismo numero de pasillos/slots) que
    se usa para las demas comparaciones. Es importante no reutilizar
    posiciones aleatorias generadas con un tamano de almacen distinto,
    ya que comparar distancias calculadas en almacenes de tamano
    diferente no es valido (una bodega mas grande siempre da distancias
    mayores, independiente de que tan bien organizada este).
    """
    rng = np.random.default_rng(random_state)
    shuffled_positions = list(all_positions)
    rng.shuffle(shuffled_positions)

    skus = products["sku"].tolist()
    sku_to_position = dict(zip(skus, shuffled_positions))

    return total_distance_for_positions(order_groups, sku_to_position)

def popularity_baseline_distance(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_groups: list[list[str]],
    all_positions: list[tuple[int, int]],
) -> float:
    """
    Distancia total usando heuristica de slotting por popularidad
    (analisis ABC): los productos mas pedidos se ubican mas cerca de la
    entrada del almacen.
    """
    frequency = orders["sku"].value_counts()
    skus_by_popularity = frequency.index.tolist()

    entrance = all_positions[0]
    positions_sorted = sorted(all_positions, key=lambda p: manhattan_distance(p, entrance))

    sku_to_position = {}
    for sku, pos in zip(skus_by_popularity, positions_sorted):
        sku_to_position[sku] = pos

    for sku in products["sku"]:
        if sku not in sku_to_position:
            sku_to_position[sku] = positions_sorted[-1]

    return total_distance_for_positions(order_groups, sku_to_position)


def assign_individual_positions(
    products: pd.DataFrame,
    cluster_anchor_positions: dict[int, tuple[float, float]],
    all_positions: list[tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """
    Asigna una posicion individual y distinta a cada producto. Los
    productos del mismo cluster se agrupan en las posiciones disponibles
    mas cercanas al ancla de ese cluster, sin repetir ninguna posicion
    entre productos distintos.
    """
    if len(products) > len(all_positions):
        raise ValueError("No hay suficientes posiciones en el almacen para todos los productos")

    used_positions = set()
    sku_to_position = {}

    clusters_sorted = products["cluster"].value_counts().index.tolist()

    for cluster_id in clusters_sorted:
        anchor = cluster_anchor_positions[cluster_id]
        cluster_skus = products.loc[products["cluster"] == cluster_id, "sku"].tolist()

        available = [p for p in all_positions if p not in used_positions]
        available.sort(key=lambda p: manhattan_distance(p, anchor))

        for sku, pos in zip(cluster_skus, available):
            sku_to_position[sku] = pos
            used_positions.add(pos)

    return sku_to_position


def make_fitness_function(
    products: pd.DataFrame,
    order_groups: list[list[str]],
    n_clusters: int,
    all_positions: list[tuple[int, int]],
):
    """
    Construye la funcion de fitness que evalua PyGAD. Cada solucion es un
    vector de longitud n_clusters*2, representando la posicion ancla de
    cada cluster. PyGAD maximiza el fitness, por lo que se retorna el
    inverso de la distancia total.
    """
    def fitness_func(ga_instance, solution, solution_idx):
        cluster_positions = solution.reshape(n_clusters, 2)
        cluster_anchor_positions = {i: tuple(cluster_positions[i]) for i in range(n_clusters)}

        sku_to_position = assign_individual_positions(products, cluster_anchor_positions, all_positions)
        distance = total_distance_for_positions(order_groups, sku_to_position)

        return 1.0 / (distance + 1e-6)

    return fitness_func


def run_genetic_algorithm(
    products: pd.DataFrame,
    order_groups: list[list[str]],
    n_clusters: int,
    all_positions: list[tuple[int, int]],
    n_aisles: int,
    slots_per_aisle: int,
    num_generations: int = 50,
    sol_per_pop: int = 20,
) -> tuple[np.ndarray, list[float]]:
    """
    Ejecuta el Algoritmo Genetico y retorna la mejor solucion encontrada
    (posiciones ancla por cluster) junto con el historial de fitness por
    generacion.
    """
    fitness_func = make_fitness_function(products, order_groups, n_clusters, all_positions)

    gene_space = []
    for _ in range(n_clusters):
        gene_space.append({"low": 1, "high": n_aisles})
        gene_space.append({"low": 1, "high": slots_per_aisle})

    fitness_history = []

    def on_generation(ga_instance):
        fitness_history.append(1.0 / ga_instance.best_solution()[1])

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=6,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=n_clusters * 2,
        gene_space=gene_space,
        gene_type=int,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=42,
        on_generation=on_generation,
    )

    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_positions = best_solution.reshape(n_clusters, 2)

    return best_positions, fitness_history
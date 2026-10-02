"""
Modulo de optimizacion de posiciones fisicas mediante Algoritmo Genetico.

Cada producto recibe una posicion tridimensional individual y distinta
(pasillo, posicion en el pasillo, nivel de altura), respetando la
capacidad estructural de cada nivel de estanteria (restricciones_
ergonomicas.py). El cromosoma es una permutacion de posiciones
asignadas a productos, y el fitness se calcula simulando la ruta real
de recoleccion (el operario siempre se dirige al producto no recogido
mas cercano), partiendo y regresando a la puerta de despacho.

El tamano del almacen (pasillos x posiciones por pasillo x niveles) se
calcula dinamicamente segun la cantidad de productos a ubicar.
"""

import numpy as np
import pandas as pd

from restricciones_ergonomicas import producto_cabe_estructuralmente

NIVELES_ALTURA = 4
FACTOR_COSTO_ALTURA = 2.0
PUERTA_DESPACHO = (0, 0, 0)


def compute_warehouse_dimensions(n_products: int, niveles: int = NIVELES_ALTURA, min_lado: int = 10) -> tuple[int, int, int]:
    """
    Calcula un layout de almacen de forma razonablemente cuadrada en el
    plano (pasillo x posicion), dado un numero fijo de niveles de
    altura, con capacidad suficiente para todos los productos.
    """
    posiciones_por_nivel_necesarias = int(np.ceil(n_products / niveles))
    lado = max(int(np.ceil(np.sqrt(posiciones_por_nivel_necesarias))), min_lado)
    return lado, lado, niveles


def build_all_positions(n_aisles: int, slots_per_aisle: int, niveles: int) -> list[tuple[int, int, int]]:
    """Genera todas las posiciones (pasillo, slot, nivel) posibles del almacen."""
    return [
        (a, s, n)
        for a in range(1, n_aisles + 1)
        for s in range(1, slots_per_aisle + 1)
        for n in range(niveles)
    ]


def manhattan_distance_3d(pos_a: tuple[int, int, int], pos_b: tuple[int, int, int]) -> float:
    """
    Distancia Manhattan en 3D, penalizando el desplazamiento vertical
    (cambio de nivel) con un factor mayor que el desplazamiento
    horizontal, ya que subir/bajar de nivel toma mas tiempo y esfuerzo
    que desplazarse en el plano del almacen.
    """
    dist_horizontal = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
    dist_vertical = abs(pos_a[2] - pos_b[2]) * FACTOR_COSTO_ALTURA
    return dist_horizontal + dist_vertical


def build_order_groups(orders: pd.DataFrame) -> list[list[str]]:
    """Agrupa el historial de pedidos en listas de SKUs por order_id."""
    return orders.groupby("order_id")["sku"].apply(list).tolist()


def evaluar_ruta_real(order_groups: list[list[str]], sku_to_position: dict[str, tuple[int, int, int]]) -> float:
    """
    Simula la ruta de recoleccion de cada pedido: el operario parte de
    la puerta de despacho y, en cada paso, se dirige al producto no
    recogido mas cercano a su posicion actual, hasta completar el
    pedido y retornar a la puerta. Esta aproximacion (vecino mas
    cercano) es mas realista que sumar distancias en un orden arbitrario
    de aparicion en el historial de pedidos.
    """
    costo_total = 0.0

    for skus in order_groups:
        posiciones_pendientes = [sku_to_position[s] for s in skus if s in sku_to_position]
        if not posiciones_pendientes:
            continue

        posicion_actual = PUERTA_DESPACHO
        while posiciones_pendientes:
            distancias = [manhattan_distance_3d(posicion_actual, p) for p in posiciones_pendientes]
            idx_mas_cercano = int(np.argmin(distancias))
            costo_total += distancias[idx_mas_cercano]
            posicion_actual = posiciones_pendientes.pop(idx_mas_cercano)

        costo_total += manhattan_distance_3d(posicion_actual, PUERTA_DESPACHO)

    return costo_total


def posiciones_validas_por_producto(
    products: pd.DataFrame,
    all_positions: list[tuple[int, int, int]],
    capacidades_por_nivel: dict[int, float] | None = None,
) -> dict[str, list[tuple[int, int, int]]]:
    """
    Para cada producto, filtra las posiciones del almacen que respetan
    la capacidad estructural del nivel. Si el producto no tiene peso
    declarado, se asume sin restriccion (todas las posiciones validas).
    """
    tiene_peso = "weight_kg" in products.columns

    validas = {}
    for _, row in products.iterrows():
        sku = row["sku"]
        peso = row["weight_kg"] if tiene_peso and pd.notna(row.get("weight_kg")) else None

        if peso is None:
            validas[sku] = all_positions
        else:
            validas[sku] = [
                p for p in all_positions
                if producto_cabe_estructuralmente(peso, p[2], capacidades_por_nivel)
            ] or all_positions

    return validas


def asignacion_inicial_aleatoria(
    products: pd.DataFrame,
    all_positions: list[tuple[int, int, int]],
    posiciones_validas: dict[str, list[tuple[int, int, int]]],
    rng: np.random.Generator,
) -> dict[str, tuple[int, int, int]]:
    """
    Genera una asignacion aleatoria valida (respetando restricciones de
    peso) sin repetir posiciones entre productos. Se procesan primero
    los productos con menos posiciones validas disponibles (mas
    restringidos), para reducir la probabilidad de quedarse sin
    posiciones libres al final.
    """
    productos_ordenados = sorted(products["sku"].tolist(), key=lambda s: len(posiciones_validas[s]))
    usadas = set()
    asignacion = {}

    for sku in productos_ordenados:
        disponibles = [p for p in posiciones_validas[sku] if p not in usadas]
        if not disponibles:
            disponibles = [p for p in all_positions if p not in usadas]
        elegida = disponibles[rng.integers(0, len(disponibles))]
        asignacion[sku] = elegida
        usadas.add(elegida)

    return asignacion


def cruzar(padre1: dict, padre2: dict, skus: list[str], all_positions: list, rng: np.random.Generator) -> dict:
    """
    Cruce de orden: hereda las posiciones de la primera mitad de
    productos desde el padre1, y completa el resto con las posiciones
    del padre2 que aun esten libres; si alguna posicion ya fue tomada,
    se asigna la primera posicion libre disponible.
    """
    punto_corte = len(skus) // 2
    hijo = {}
    usadas = set()

    for sku in skus[:punto_corte]:
        hijo[sku] = padre1[sku]
        usadas.add(padre1[sku])

    for sku in skus[punto_corte:]:
        pos = padre2[sku]
        if pos not in usadas:
            hijo[sku] = pos
            usadas.add(pos)
        else:
            libre = next(p for p in all_positions if p not in usadas)
            hijo[sku] = libre
            usadas.add(libre)

    return hijo


def mutar(asignacion: dict, skus: list[str], rng: np.random.Generator, tasa: float = 0.05) -> dict:
    """Intercambia la posicion de pares de productos elegidos al azar."""
    asignacion = asignacion.copy()
    n_mutaciones = max(1, int(len(skus) * tasa))
    for _ in range(n_mutaciones):
        a, b = rng.choice(skus, size=2, replace=False)
        asignacion[a], asignacion[b] = asignacion[b], asignacion[a]
    return asignacion


def reparar_restricciones(
    asignacion: dict,
    peso_por_sku: dict,
    capacidades_por_nivel: dict[int, float] | None,
    rng: np.random.Generator,
    max_intentos: int = 3,
) -> dict:
    """
    Corrige violaciones de capacidad estructural que pueden introducirse
    durante el cruce o la mutacion (operaciones que intercambian
    posiciones sin verificar si el producto resultante cumple la
    capacidad de su nuevo nivel).
    """
    asignacion = asignacion.copy()

    for _ in range(max_intentos):
        skus = list(asignacion.keys())
        violaciones = []
        for sku in skus:
            peso = peso_por_sku.get(sku)
            if peso is None:
                continue
            nivel = asignacion[sku][2]
            if not producto_cabe_estructuralmente(peso, nivel, capacidades_por_nivel):
                violaciones.append(sku)

        if not violaciones:
            break

        for sku in violaciones:
            peso = peso_por_sku[sku]
            candidato = next(
                (
                    otro for otro in skus
                    if otro != sku
                    and producto_cabe_estructuralmente(peso, asignacion[otro][2], capacidades_por_nivel)
                    and producto_cabe_estructuralmente(peso_por_sku.get(otro) or 0, asignacion[sku][2], capacidades_por_nivel)
                ),
                None,
            )
            if candidato is not None:
                asignacion[sku], asignacion[candidato] = asignacion[candidato], asignacion[sku]

    return asignacion


def run_genetic_algorithm(
    products: pd.DataFrame,
    order_groups: list[list[str]],
    all_positions: list[tuple[int, int, int]],
    capacidades_por_nivel: dict[int, float] | None = None,
    num_generations: int = 40,
    poblacion: int = 20,
    random_state: int = 42,
) -> tuple[dict[str, tuple[int, int, int]], list[float]]:
    """
    Ejecuta el Algoritmo Genetico y retorna la mejor asignacion de
    posiciones encontrada (sku -> posicion), junto con el historial de
    costo (distancia) del mejor individuo por generacion.
    """
    rng = np.random.default_rng(random_state)
    skus = products["sku"].tolist()

    posiciones_validas = posiciones_validas_por_producto(products, all_positions, capacidades_por_nivel)

    tiene_peso = "weight_kg" in products.columns
    peso_por_sku = {
        row["sku"]: (row["weight_kg"] if tiene_peso and pd.notna(row.get("weight_kg")) else None)
        for _, row in products.iterrows()
    }

    poblacion_actual = [
        asignacion_inicial_aleatoria(products, all_positions, posiciones_validas, rng)
        for _ in range(poblacion)
    ]

    historial_mejor_costo = []

    for generacion in range(num_generations):
        costos = [evaluar_ruta_real(order_groups, individuo) for individuo in poblacion_actual]

        orden = np.argsort(costos)
        poblacion_actual = [poblacion_actual[i] for i in orden]
        costos = [costos[i] for i in orden]

        historial_mejor_costo.append(costos[0])

        n_elite = max(2, poblacion // 5)
        nueva_poblacion = poblacion_actual[:n_elite]

        while len(nueva_poblacion) < poblacion:
            padre1, padre2 = poblacion_actual[rng.integers(0, poblacion // 2)], poblacion_actual[rng.integers(0, poblacion // 2)]
            hijo = cruzar(padre1, padre2, skus, all_positions, rng)
            hijo = mutar(hijo, skus, rng)
            hijo = reparar_restricciones(hijo, peso_por_sku, capacidades_por_nivel, rng)
            nueva_poblacion.append(hijo)

        poblacion_actual = nueva_poblacion

    costos_finales = [evaluar_ruta_real(order_groups, individuo) for individuo in poblacion_actual]
    mejor_idx = int(np.argmin(costos_finales))

    return poblacion_actual[mejor_idx], historial_mejor_costo


def random_baseline_distance(
    products: pd.DataFrame,
    order_groups: list[list[str]],
    all_positions: list[tuple[int, int, int]],
    random_state: int = 42,
) -> float:
    """Distancia total con asignacion aleatoria, sobre el mismo almacen que las demas comparaciones."""
    rng = np.random.default_rng(random_state)
    posiciones = list(all_positions)
    rng.shuffle(posiciones)
    sku_to_position = dict(zip(products["sku"].tolist(), posiciones))
    return evaluar_ruta_real(order_groups, sku_to_position)


def popularity_baseline_distance(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_groups: list[list[str]],
    all_positions: list[tuple[int, int, int]],
) -> float:
    """
    Distancia total con heuristica de popularidad (analisis ABC): los
    productos mas pedidos se ubican mas cerca de la puerta de despacho.
    """
    frecuencia = orders["sku"].value_counts()
    skus_por_popularidad = frecuencia.index.tolist()

    posiciones_ordenadas = sorted(all_positions, key=lambda p: manhattan_distance_3d(p, PUERTA_DESPACHO))

    sku_to_position = {}
    for sku, pos in zip(skus_por_popularidad, posiciones_ordenadas):
        sku_to_position[sku] = pos
    for sku in products["sku"]:
        if sku not in sku_to_position:
            sku_to_position[sku] = posiciones_ordenadas[-1]

    return evaluar_ruta_real(order_groups, sku_to_position)
"""
Modulo de optimizacion de posiciones fisicas mediante Algoritmo Genetico.

Cada producto recibe una posicion tridimensional individual y distinta:
(pasillo, posicion en el pasillo, nivel de altura).

El algoritmo considera:

- Distancia Manhattan 3D.
- Penalizacion por desplazamiento vertical.
- Restricciones estructurales de estanteria.
- Baseline aleatorio.
- Baseline por popularidad.
- Algoritmo Genetico puro.
- Inicializacion guiada por clusters obtenidos mediante K-Medoids.

El fitness simula la ruta de picking de cada pedido mediante una
heuristica de vecino mas cercano, partiendo y regresando a la puerta
de despacho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .restricciones_ergonomicas import (
    producto_cabe_estructuralmente,
)


# ============================================================
# PARAMETROS GENERALES
# ============================================================

NIVELES_ALTURA = 4

FACTOR_COSTO_ALTURA = 2.0

PUERTA_DESPACHO = (0, 0, 0)


# ============================================================
# DIMENSIONES DEL ALMACEN
# ============================================================

def compute_warehouse_dimensions(
    n_products: int,
    niveles: int = NIVELES_ALTURA,
    min_lado: int = 10,
) -> tuple[int, int, int]:
    """
    Calcula automaticamente las dimensiones del almacen.

    Se genera un layout aproximadamente cuadrado en el plano:

        pasillos x slots x niveles

    asegurando capacidad suficiente para todos los productos.
    """

    if n_products <= 0:
        raise ValueError(
            "n_products debe ser mayor que cero."
        )

    posiciones_por_nivel_necesarias = int(
        np.ceil(
            n_products / niveles
        )
    )

    lado = max(
        int(
            np.ceil(
                np.sqrt(
                    posiciones_por_nivel_necesarias
                )
            )
        ),
        min_lado,
    )

    return (
        lado,
        lado,
        niveles,
    )


def build_all_positions(
    n_aisles: int,
    slots_per_aisle: int,
    niveles: int,
) -> list[tuple[int, int, int]]:
    """
    Genera todas las posiciones posibles del almacen.

    Cada posicion tiene la forma:

        (pasillo, slot, nivel)
    """

    return [
        (aisle, slot, nivel)
        for aisle in range(
            1,
            n_aisles + 1,
        )
        for slot in range(
            1,
            slots_per_aisle + 1,
        )
        for nivel in range(
            niveles
        )
    ]


# ============================================================
# DISTANCIA
# ============================================================

def manhattan_distance_3d(
    pos_a: tuple[int, int, int],
    pos_b: tuple[int, int, int],
) -> float:
    """
    Distancia Manhattan tridimensional.

    El desplazamiento vertical recibe una penalizacion adicional
    debido al mayor costo asociado al cambio de nivel.
    """

    dist_horizontal = (
        abs(
            pos_a[0] - pos_b[0]
        )
        +
        abs(
            pos_a[1] - pos_b[1]
        )
    )

    dist_vertical = (
        abs(
            pos_a[2] - pos_b[2]
        )
        * FACTOR_COSTO_ALTURA
    )

    return (
        dist_horizontal
        +
        dist_vertical
    )


# ============================================================
# PEDIDOS
# ============================================================

def build_order_groups(
    orders: pd.DataFrame,
) -> list[list[str]]:
    """
    Agrupa los SKU del historial segun order_id.
    """

    columnas_requeridas = {
        "order_id",
        "sku",
    }

    faltantes = (
        columnas_requeridas
        -
        set(orders.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en orders: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    return (
        orders
        .groupby("order_id")["sku"]
        .apply(list)
        .tolist()
    )


# ============================================================
# FUNCION OBJETIVO / FITNESS
# ============================================================

def evaluar_ruta_real(
    order_groups: list[list[str]],
    sku_to_position: dict[
        str,
        tuple[int, int, int]
    ],
) -> float:
    """
    Simula la ruta de picking de todos los pedidos.

    Para cada pedido:

    1. El operario comienza en la puerta de despacho.
    2. Se dirige al producto pendiente mas cercano.
    3. Repite hasta completar el pedido.
    4. Regresa a la puerta de despacho.

    El costo total corresponde a la suma de las distancias.
    """

    costo_total = 0.0

    for skus in order_groups:

        posiciones_pendientes = [
            sku_to_position[sku]
            for sku in skus
            if sku in sku_to_position
        ]

        if not posiciones_pendientes:
            continue

        posicion_actual = (
            PUERTA_DESPACHO
        )

        while posiciones_pendientes:

            distancias = [
                manhattan_distance_3d(
                    posicion_actual,
                    posicion,
                )
                for posicion
                in posiciones_pendientes
            ]

            idx_mas_cercano = int(
                np.argmin(
                    distancias
                )
            )

            costo_total += (
                distancias[
                    idx_mas_cercano
                ]
            )

            posicion_actual = (
                posiciones_pendientes.pop(
                    idx_mas_cercano
                )
            )

        costo_total += (
            manhattan_distance_3d(
                posicion_actual,
                PUERTA_DESPACHO,
            )
        )

    return float(
        costo_total
    )


# ============================================================
# POSICIONES VALIDAS
# ============================================================

def posiciones_validas_por_producto(
    products: pd.DataFrame,
    all_positions: list[
        tuple[int, int, int]
    ],
    capacidades_por_nivel: (
        dict[int, float] | None
    ) = None,
) -> dict[
    str,
    list[
        tuple[int, int, int]
    ],
]:
    """
    Determina las posiciones validas para cada SKU.

    Si existe weight_kg, se verifica la capacidad estructural
    del nivel.

    Si no existe peso, todas las posiciones se consideran
    disponibles.
    """

    if "sku" not in products.columns:
        raise ValueError(
            "products debe contener "
            "la columna 'sku'."
        )

    tiene_peso = (
        "weight_kg"
        in products.columns
    )

    validas = {}

    for _, row in (
        products.iterrows()
    ):

        sku = row["sku"]

        if (
            tiene_peso
            and pd.notna(
                row.get(
                    "weight_kg"
                )
            )
        ):
            peso = float(
                row["weight_kg"]
            )
        else:
            peso = None

        if peso is None:

            validas[sku] = list(
                all_positions
            )

        else:

            posiciones = [
                posicion
                for posicion
                in all_positions
                if producto_cabe_estructuralmente(
                    peso,
                    posicion[2],
                    capacidades_por_nivel,
                )
            ]

            # Respaldo para no dejar
            # un SKU sin ubicaciones.
            validas[sku] = (
                posiciones
                if posiciones
                else list(
                    all_positions
                )
            )

    return validas


# ============================================================
# ASIGNACION INICIAL ALEATORIA
# ============================================================

def asignacion_inicial_aleatoria(
    products: pd.DataFrame,
    all_positions: list[
        tuple[int, int, int]
    ],
    posiciones_validas: dict[
        str,
        list[
            tuple[int, int, int]
        ],
    ],
    rng: np.random.Generator,
) -> dict[
    str,
    tuple[int, int, int],
]:
    """
    Genera una asignacion inicial aleatoria sin repetir posiciones.

    Los SKU con menos posiciones validas se procesan primero.
    """

    productos_ordenados = sorted(
        products["sku"].tolist(),
        key=lambda sku: len(
            posiciones_validas[
                sku
            ]
        ),
    )

    usadas = set()

    asignacion = {}

    for sku in productos_ordenados:

        disponibles = [
            posicion
            for posicion
            in posiciones_validas[
                sku
            ]
            if posicion
            not in usadas
        ]

        if not disponibles:

            disponibles = [
                posicion
                for posicion
                in all_positions
                if posicion
                not in usadas
            ]

        if not disponibles:
            raise ValueError(
                "No existen suficientes "
                "posiciones para todos "
                "los productos."
            )

        indice = int(
            rng.integers(
                0,
                len(disponibles),
            )
        )

        elegida = (
            disponibles[
                indice
            ]
        )

        asignacion[
            sku
        ] = elegida

        usadas.add(
            elegida
        )

    return asignacion


# ============================================================
# ASIGNACION GUIADA POR K-MEDOIDS
# ============================================================

def asignacion_inicial_guiada_clusters(
    products: pd.DataFrame,
    all_positions: list[
        tuple[int, int, int]
    ],
    posiciones_validas: dict[
        str,
        list[
            tuple[int, int, int]
        ],
    ],
    rng: np.random.Generator,
) -> dict[
    str,
    tuple[int, int, int],
]:
    """
    Genera una asignacion inicial guiada por clusters.

    Los SKU pertenecientes al mismo cluster se intentan ubicar
    fisicamente cerca entre si.

    K-Medoids guia solamente la poblacion inicial.

    Posteriormente el Algoritmo Genetico mantiene libertad para
    modificar las posiciones segun el fitness.
    """

    if (
        "cluster"
        not in products.columns
    ):
        return (
            asignacion_inicial_aleatoria(
                products,
                all_positions,
                posiciones_validas,
                rng,
            )
        )

    asignacion = {}

    usadas = set()

    grupos = [
        grupo["sku"].tolist()
        for _, grupo
        in products.groupby(
            "cluster",
            dropna=False,
        )
    ]

    orden_clusters = (
        rng.permutation(
            len(grupos)
        )
    )

    for idx_cluster in (
        orden_clusters
    ):

        skus_cluster = grupos[
            int(idx_cluster)
        ]

        skus_cluster = sorted(
            skus_cluster,
            key=lambda sku: len(
                posiciones_validas.get(
                    sku,
                    [],
                )
            ),
        )

        posicion_referencia = None

        for sku in skus_cluster:

            candidatos = [
                posicion
                for posicion
                in posiciones_validas.get(
                    sku,
                    all_positions,
                )
                if posicion
                not in usadas
            ]

            if not candidatos:

                candidatos = [
                    posicion
                    for posicion
                    in all_positions
                    if posicion
                    not in usadas
                ]

            if not candidatos:
                raise ValueError(
                    "No existen suficientes "
                    "posiciones disponibles "
                    "para asignar todos "
                    "los SKU."
                )

            if (
                posicion_referencia
                is None
            ):

                indice = int(
                    rng.integers(
                        0,
                        len(candidatos),
                    )
                )

                elegida = (
                    candidatos[
                        indice
                    ]
                )

                posicion_referencia = (
                    elegida
                )

            else:

                distancias = [
                    manhattan_distance_3d(
                        posicion_referencia,
                        posicion,
                    )
                    for posicion
                    in candidatos
                ]

                distancia_minima = (
                    min(
                        distancias
                    )
                )

                candidatos_cercanos = [
                    posicion
                    for posicion, distancia
                    in zip(
                        candidatos,
                        distancias,
                    )
                    if (
                        distancia
                        ==
                        distancia_minima
                    )
                ]

                indice = int(
                    rng.integers(
                        0,
                        len(
                            candidatos_cercanos
                        ),
                    )
                )

                elegida = (
                    candidatos_cercanos[
                        indice
                    ]
                )

            asignacion[
                sku
            ] = elegida

            usadas.add(
                elegida
            )

    return asignacion


# ============================================================
# CRUCE
# ============================================================

def cruzar(
    padre1: dict[
        str,
        tuple[int, int, int],
    ],
    padre2: dict[
        str,
        tuple[int, int, int],
    ],
    skus: list[str],
    all_positions: list[
        tuple[int, int, int]
    ],
    rng: np.random.Generator,
) -> dict[
    str,
    tuple[int, int, int],
]:
    """
    Cruce entre dos individuos.

    La primera mitad de los SKU se hereda del padre 1.

    La segunda mitad intenta heredar posiciones del padre 2.

    Si una posicion ya esta ocupada se selecciona una posicion
    libre.
    """

    punto_corte = (
        len(skus) // 2
    )

    hijo = {}

    usadas = set()

    for sku in (
        skus[:punto_corte]
    ):

        posicion = (
            padre1[sku]
        )

        hijo[sku] = posicion

        usadas.add(
            posicion
        )

    for sku in (
        skus[punto_corte:]
    ):

        posicion = (
            padre2[sku]
        )

        if (
            posicion
            not in usadas
        ):

            hijo[
                sku
            ] = posicion

            usadas.add(
                posicion
            )

        else:

            disponibles = [
                posicion_libre
                for posicion_libre
                in all_positions
                if posicion_libre
                not in usadas
            ]

            if not disponibles:
                raise ValueError(
                    "No quedan posiciones "
                    "libres durante el cruce."
                )

            indice = int(
                rng.integers(
                    0,
                    len(disponibles),
                )
            )

            libre = (
                disponibles[
                    indice
                ]
            )

            hijo[
                sku
            ] = libre

            usadas.add(
                libre
            )

    return hijo


# ============================================================
# MUTACION
# ============================================================

def mutar(
    asignacion: dict[
        str,
        tuple[int, int, int],
    ],
    skus: list[str],
    rng: np.random.Generator,
    tasa: float = 0.05,
) -> dict[
    str,
    tuple[int, int, int],
]:
    """
    Intercambia posiciones entre pares de productos.
    """

    asignacion = (
        asignacion.copy()
    )

    if len(skus) < 2:
        return asignacion

    n_mutaciones = max(
        1,
        int(
            len(skus)
            * tasa
        ),
    )

    for _ in range(
        n_mutaciones
    ):

        seleccionados = (
            rng.choice(
                skus,
                size=2,
                replace=False,
            )
        )

        sku_a = (
            seleccionados[0]
        )

        sku_b = (
            seleccionados[1]
        )

        (
            asignacion[sku_a],
            asignacion[sku_b],
        ) = (
            asignacion[sku_b],
            asignacion[sku_a],
        )

    return asignacion


# ============================================================
# REPARACION DE RESTRICCIONES
# ============================================================

def reparar_restricciones(
    asignacion: dict[
        str,
        tuple[int, int, int],
    ],
    peso_por_sku: dict,
    capacidades_por_nivel: (
        dict[int, float] | None
    ),
    rng: np.random.Generator,
    max_intentos: int = 3,
) -> dict[
    str,
    tuple[int, int, int],
]:
    """
    Corrige violaciones de capacidad estructural introducidas
    durante el cruce o la mutacion.
    """

    asignacion = (
        asignacion.copy()
    )

    for _ in range(
        max_intentos
    ):

        skus = list(
            asignacion.keys()
        )

        violaciones = []

        for sku in skus:

            peso = (
                peso_por_sku.get(
                    sku
                )
            )

            if peso is None:
                continue

            nivel = (
                asignacion[
                    sku
                ][2]
            )

            if not (
                producto_cabe_estructuralmente(
                    peso,
                    nivel,
                    capacidades_por_nivel,
                )
            ):
                violaciones.append(
                    sku
                )

        if not violaciones:
            break

        for sku in violaciones:

            peso = (
                peso_por_sku[
                    sku
                ]
            )

            candidatos = []

            for otro in skus:

                if otro == sku:
                    continue

                peso_otro = (
                    peso_por_sku.get(
                        otro
                    )
                )

                if peso_otro is None:
                    peso_otro = 0.0

                nivel_otro = (
                    asignacion[
                        otro
                    ][2]
                )

                nivel_actual = (
                    asignacion[
                        sku
                    ][2]
                )

                cabe_sku = (
                    producto_cabe_estructuralmente(
                        peso,
                        nivel_otro,
                        capacidades_por_nivel,
                    )
                )

                cabe_otro = (
                    producto_cabe_estructuralmente(
                        peso_otro,
                        nivel_actual,
                        capacidades_por_nivel,
                    )
                )

                if (
                    cabe_sku
                    and cabe_otro
                ):
                    candidatos.append(
                        otro
                    )

            if candidatos:

                candidato = (
                    candidatos[
                        int(
                            rng.integers(
                                0,
                                len(
                                    candidatos
                                ),
                            )
                        )
                    ]
                )

                (
                    asignacion[sku],
                    asignacion[candidato],
                ) = (
                    asignacion[candidato],
                    asignacion[sku],
                )

    return asignacion


# ============================================================
# ALGORITMO GENETICO
# ============================================================

def run_genetic_algorithm(
    products: pd.DataFrame,
    order_groups: list[
        list[str]
    ],
    all_positions: list[
        tuple[int, int, int]
    ],
    capacidades_por_nivel: (
        dict[int, float] | None
    ) = None,
    num_generations: int = 40,
    poblacion: int = 20,
    random_state: int = 42,
    proporcion_guiada: float = 0.5,
    verbose: bool = False,
    nombre: str = "GA",
) -> tuple[
    dict[
        str,
        tuple[int, int, int],
    ],
    list[float],
]:
    """
    Ejecuta el Algoritmo Genetico.

    proporcion_guiada:

        0.0 = GA puro.
        0.5 = mitad K-Medoids y mitad aleatorio.
        1.0 = poblacion inicial completamente guiada.

    verbose:

        True muestra el avance de cada generacion.
    """

    if not (
        0.0
        <= proporcion_guiada
        <= 1.0
    ):
        raise ValueError(
            "proporcion_guiada debe "
            "estar entre 0.0 y 1.0."
        )

    if poblacion < 4:
        raise ValueError(
            "La poblacion debe ser "
            "al menos 4."
        )

    if num_generations < 1:
        raise ValueError(
            "num_generations debe "
            "ser al menos 1."
        )

    rng = (
        np.random.default_rng(
            random_state
        )
    )

    skus = (
        products[
            "sku"
        ].tolist()
    )

    posiciones_validas = (
        posiciones_validas_por_producto(
            products,
            all_positions,
            capacidades_por_nivel,
        )
    )

    tiene_peso = (
        "weight_kg"
        in products.columns
    )

    peso_por_sku = {
        row["sku"]: (
            float(
                row["weight_kg"]
            )
            if (
                tiene_peso
                and pd.notna(
                    row.get(
                        "weight_kg"
                    )
                )
            )
            else None
        )
        for _, row
        in products.iterrows()
    }

    # --------------------------------------------------------
    # POBLACION INICIAL
    # --------------------------------------------------------

    n_guiados = int(
        poblacion
        * proporcion_guiada
    )

    n_aleatorios = (
        poblacion
        -
        n_guiados
    )

    poblacion_actual = []

    if verbose:

        print()
        print(
            "======================================"
        )
        print(
            f"INICIANDO {nombre}"
        )
        print(
            "======================================"
        )

        print(
            f"SKU: {len(skus)}"
        )

        print(
            f"Pedidos: "
            f"{len(order_groups)}"
        )

        print(
            f"Poblacion: "
            f"{poblacion}"
        )

        print(
            f"Generaciones: "
            f"{num_generations}"
        )

        print(
            f"Guiados K-Medoids: "
            f"{n_guiados}"
        )

        print(
            f"Aleatorios: "
            f"{n_aleatorios}"
        )

        print()

    for _ in range(
        n_guiados
    ):

        individuo = (
            asignacion_inicial_guiada_clusters(
                products,
                all_positions,
                posiciones_validas,
                rng,
            )
        )

        poblacion_actual.append(
            individuo
        )

    for _ in range(
        n_aleatorios
    ):

        individuo = (
            asignacion_inicial_aleatoria(
                products,
                all_positions,
                posiciones_validas,
                rng,
            )
        )

        poblacion_actual.append(
            individuo
        )

    historial_mejor_costo = []

    # --------------------------------------------------------
    # EVOLUCION
    # --------------------------------------------------------

    for generacion in range(
        num_generations
    ):

        costos = [
            evaluar_ruta_real(
                order_groups,
                individuo,
            )
            for individuo
            in poblacion_actual
        ]

        orden = (
            np.argsort(
                costos
            )
        )

        poblacion_actual = [
            poblacion_actual[
                indice
            ]
            for indice
            in orden
        ]

        costos = [
            costos[
                indice
            ]
            for indice
            in orden
        ]

        mejor_costo = float(
            costos[0]
        )

        historial_mejor_costo.append(
            mejor_costo
        )

        if verbose:

            print(
                f"[{nombre}] "
                f"Generacion "
                f"{generacion + 1}/"
                f"{num_generations} "
                f"| Mejor distancia: "
                f"{mejor_costo:.2f}"
            )

        n_elite = max(
            2,
            poblacion // 5,
        )

        n_elite = min(
            n_elite,
            poblacion,
        )

        nueva_poblacion = (
            poblacion_actual[
                :n_elite
            ]
        )

        limite_padres = max(
            2,
            poblacion // 2,
        )

        limite_padres = min(
            limite_padres,
            len(
                poblacion_actual
            ),
        )

        while (
            len(
                nueva_poblacion
            )
            <
            poblacion
        ):

            padre1 = (
                poblacion_actual[
                    int(
                        rng.integers(
                            0,
                            limite_padres,
                        )
                    )
                ]
            )

            padre2 = (
                poblacion_actual[
                    int(
                        rng.integers(
                            0,
                            limite_padres,
                        )
                    )
                ]
            )

            hijo = cruzar(
                padre1,
                padre2,
                skus,
                all_positions,
                rng,
            )

            hijo = mutar(
                hijo,
                skus,
                rng,
            )

            hijo = (
                reparar_restricciones(
                    hijo,
                    peso_por_sku,
                    capacidades_por_nivel,
                    rng,
                )
            )

            nueva_poblacion.append(
                hijo
            )

        poblacion_actual = (
            nueva_poblacion
        )

    # --------------------------------------------------------
    # EVALUACION FINAL
    # --------------------------------------------------------

    costos_finales = [
        evaluar_ruta_real(
            order_groups,
            individuo,
        )
        for individuo
        in poblacion_actual
    ]

    mejor_idx = int(
        np.argmin(
            costos_finales
        )
    )

    mejor_asignacion = (
        poblacion_actual[
            mejor_idx
        ]
    )

    mejor_costo_final = float(
        costos_finales[
            mejor_idx
        ]
    )

    if verbose:

        print()

        print(
            f"[{nombre}] "
            f"FINALIZADO"
        )

        print(
            f"[{nombre}] "
            f"Mejor distancia final: "
            f"{mejor_costo_final:.2f}"
        )

        print()

    return (
        mejor_asignacion,
        historial_mejor_costo,
    )


# ============================================================
# BASELINE ALEATORIO
# ============================================================

def random_baseline_distance(
    products: pd.DataFrame,
    order_groups: list[
        list[str]
    ],
    all_positions: list[
        tuple[int, int, int]
    ],
    random_state: int = 42,
) -> float:
    """
    Calcula la distancia total de una asignacion aleatoria.
    """

    rng = (
        np.random.default_rng(
            random_state
        )
    )

    posiciones = list(
        all_positions
    )

    rng.shuffle(
        posiciones
    )

    skus = (
        products[
            "sku"
        ].tolist()
    )

    if (
        len(posiciones)
        <
        len(skus)
    ):
        raise ValueError(
            "El almacen no tiene "
            "suficientes posiciones."
        )

    sku_to_position = dict(
        zip(
            skus,
            posiciones[
                :len(skus)
            ],
        )
    )

    return evaluar_ruta_real(
        order_groups,
        sku_to_position,
    )


# ============================================================
# BASELINE POPULARIDAD
# ============================================================

def popularity_baseline_distance(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_groups: list[
        list[str]
    ],
    all_positions: list[
        tuple[int, int, int]
    ],
) -> float:
    """
    Heuristica ABC / popularidad.

    Los SKU mas frecuentes se asignan primero a las posiciones
    mas cercanas a la puerta de despacho.
    """

    frecuencia = (
        orders[
            "sku"
        ].value_counts()
    )

    skus_productos = (
        products[
            "sku"
        ].tolist()
    )

    skus_por_popularidad = [
        sku
        for sku
        in frecuencia.index.tolist()
        if sku
        in set(
            skus_productos
        )
    ]

    # Agregar SKU que no aparezcan
    # en el historial.
    restantes = [
        sku
        for sku
        in skus_productos
        if sku
        not in set(
            skus_por_popularidad
        )
    ]

    skus_ordenados = (
        skus_por_popularidad
        +
        restantes
    )

    posiciones_ordenadas = sorted(
        all_positions,
        key=lambda posicion:
            manhattan_distance_3d(
                posicion,
                PUERTA_DESPACHO,
            ),
    )

    if (
        len(posiciones_ordenadas)
        <
        len(skus_ordenados)
    ):
        raise ValueError(
            "El almacen no tiene "
            "suficientes posiciones."
        )

    sku_to_position = dict(
        zip(
            skus_ordenados,
            posiciones_ordenadas,
        )
    )

    return evaluar_ruta_real(
        order_groups,
        sku_to_position,
    )
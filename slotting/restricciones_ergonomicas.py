"""
Restricciones de peso para asignacion de productos en estanterias.

Este modulo define dos tipos de limite, de naturaleza distinta:

1. Limite estructural del rack: la capacidad de carga certificada de
   cada nivel de estanteria, que no tiene un valor unico a nivel
   nacional (normado por NCh2369:2025 y NCh3703:2023, que exigen
   certificacion por ingeniero segun el diseño especifico del rack).
   Este dato debe ser entregado por la empresa como parte del layout
   del almacen. Si no se entrega, se utiliza un valor de referencia de
   industria para estanterias de picking manual. Esta es la unica
   restriccion aplicada en la etapa de Slotting.

2. Limite ergonomico de manejo manual (Ley N.20.001, modificada por Ley
   N.20.949 "Ley del Saco", y su reglamento Decreto Supremo N.63/2005
   del Ministerio del Trabajo y Prevision Social): 25 kg como limite
   general de manipulacion manual por una persona. No aplica si el
   producto se mueve con ayuda mecanica (Art. 211-G y 211-H del Codigo
   del Trabajo). Esta restriccion corresponde a la etapa de
   Picking/Routing (como se recolecta un producto), no a la etapa de
   Slotting (donde se ubica fisicamente).

Todos los valores son parametros ajustables.
"""

PESO_MAXIMO_MANIPULACION_MANUAL_KG = 25.0

CAPACIDAD_ESTRUCTURAL_DEFECTO_KG = 400.0


def capacidad_estructural_nivel(nivel: int, capacidades_por_nivel: dict[int, float] | None = None) -> float:
    """
    Retorna la capacidad estructural (kg) de un nivel de estanteria.
    Si la empresa entrega capacidades reales (capacidades_por_nivel),
    se usan esas. En caso contrario, se aplica un valor de referencia
    de industria decreciente con la altura.
    """
    if capacidades_por_nivel is not None and nivel in capacidades_por_nivel:
        return capacidades_por_nivel[nivel]

    reduccion = max(0.3, 1.0 - 0.2 * nivel)
    return CAPACIDAD_ESTRUCTURAL_DEFECTO_KG * reduccion


def producto_cabe_estructuralmente(
    peso_producto_kg: float,
    nivel: int,
    capacidades_por_nivel: dict[int, float] | None = None,
) -> bool:
    """
    Indica si un producto puede asignarse a un nivel dado considerando
    unicamente la capacidad estructural del rack. Restriccion utilizada
    en la etapa de Slotting.
    """
    return peso_producto_kg <= capacidad_estructural_nivel(nivel, capacidades_por_nivel)


def peso_maximo_manejo_manual(
    nivel: int,
    tiene_ayuda_mecanica: bool = False,
    capacidades_por_nivel: dict[int, float] | None = None,
) -> float:
    """
    Peso maximo permitido para recolectar un producto de un nivel dado,
    combinando la capacidad estructural del rack con el limite
    ergonomico legal cuando corresponde (sin ayuda mecanica).
    Restriccion reservada para la etapa de Picking/Routing.
    """
    capacidad = capacidad_estructural_nivel(nivel, capacidades_por_nivel)

    if tiene_ayuda_mecanica:
        return capacidad

    return min(capacidad, PESO_MAXIMO_MANIPULACION_MANUAL_KG)


def producto_cabe_en_nivel_para_picking(
    peso_producto_kg: float,
    nivel: int,
    tiene_ayuda_mecanica: bool = False,
    capacidades_por_nivel: dict[int, float] | None = None,
) -> bool:
    """
    Indica si un producto puede recolectarse de un nivel dado,
    considerando tanto la capacidad estructural como el limite
    ergonomico legal. Restriccion reservada para la etapa de
    Picking/Routing.
    """
    return peso_producto_kg <= peso_maximo_manejo_manual(nivel, tiene_ayuda_mecanica, capacidades_por_nivel)
# Motor de Optimización WES — Slotting & Routing

Motor de optimización dinámica para Sistemas de Ejecución de Almacenes (WES),
desarrollado para STG. Combina Clustering (K-Medoids) + Algoritmos Genéticos
para *slotting*, y Aprendizaje por Refuerzo Profundo (DRL) para *routing*,
expuesto vía API REST (FastAPI).

## Equipo
- Daniel Díaz Martínez
- Ignacio Molina
- Brandon Quezada Rencoret

Profesor guía: Víctor Escobar Jeria

## Estructura del repo

```
wes-optimizer/
├── slotting/           # Módulo de clustering (K-Medoids) + Algoritmo Genético
│   ├── clustering.py
│   ├── genetic_algorithm.py
│   └── pipeline.py
├── routing/             # Módulo de DRL (Gymnasium + Stable-Baselines3)
│   ├── environment.py
│   ├── train.py
│   └── evaluate.py
├── api/                 # Microservicio FastAPI
│   ├── main.py
│   ├── schemas.py
│   └── routers/
├── data/
│   ├── raw/              # Datos originales de STG (NO subir a git, ver .gitignore)
│   ├── processed/        # Datos limpios listos para modelar
│   └── synthetic/        # Datos sintéticos para desarrollar mientras no llega el dataset real
├── notebooks/            # EDA y experimentación exploratoria
├── tests/                # Tests unitarios
├── docs/                 # Informes, diagramas, actas de reunión
├── requirements.txt
└── .gitignore
```

## Setup rápido

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Flujo de trabajo en Git

- `main`: rama estable, protegida. Solo se mergea vía Pull Request.
- `feature/slotting`, `feature/routing`, `feature/api`: cada quien trabaja en la suya.
- Antes de mergear a `main`, al menos otro integrante revisa el PR.

## Estado de datos

Aún no llega el dataset real de STG. Mientras tanto, se trabaja con datos
sintéticos generados en `data/synthetic/` para no bloquear el desarrollo de
los módulos. Cuando llegue el dataset real, se evaluará si basta con
simulación (Pandas) o si se requiere infraestructura adicional — decisión
fuera del alcance definido para el paso a producción.

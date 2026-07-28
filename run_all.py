"""
Orquestador de nba_predictor: corre el pipeline de PROCESAMIENTO sobre los
crudos que ya esten en data/raw/  (limpieza -> clustering -> dataset unificado).

  python run_all.py

NO descarga: la descarga (nba_api) se corre aparte y en tu maquina:
  python -m src.ingest.download_teams   --start 2019 --end 2025
  python -m src.ingest.download_players --start 2019 --end 2025
  python -m src.ingest.download_lineups --start 2019 --end 2025
En el sandbox, en vez de descargar, se puede sembrar:
  python -m src.ingest.seed_from_repos

Para actualizar una sola temporada nueva, ver:  python update_season.py --help
"""

import runpy
import sys


STEPS = [
    ("Limpieza de equipos", "src.features.clean_teams"),
    ("Limpieza de jugadores", "src.features.clean_players"),
    ("Clustering de estilo", "src.models.style_clustering"),
    ("Fuerza del plantel", "src.features.squad_strength"),
    ("Arquetipos de jugador", "src.features.player_roles"),
    ("Cluster de tipo de plantel", "src.models.roster_type_clustering"),
    ("Dataset unificado", "src.ingest.build_dataset"),
    ("Proyeccion de jugador (Fase A)", "src.features.player_projection"),
    ("Fuerza proyectada del equipo (Fase B)", "src.features.team_projection"),
    ("Features de entrenador", "src.features.coach_features"),
    ("M1 wins + backtest (Fase C)", "src.models.m1_wins"),
    ("Graficas de exploracion", "src.viz.plots"),
    ("Graficas de fuerza y proximidad", "src.viz.analysis_plots"),
    ("Documento PDF", "src.report.build_pdf"),
]


def main() -> None:
    print("=" * 64)
    print("NBA_PREDICTOR — pipeline de procesamiento")
    print("=" * 64)
    for i, (label, module) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}  ({module})")
        print("-" * 64)
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit:
            pass
        except FileNotFoundError as e:
            print(f"  ! saltado: {e}")
    print("\n" + "=" * 64)
    print("LISTO. Revisa data/unified_team_season.csv y outputs/")
    print("=" * 64)


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]  # evita que argparse de submódulos lea flags ajenas
    main()

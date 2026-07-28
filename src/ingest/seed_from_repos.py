"""
Siembra data/raw/ de nba_predictor con los datos REALES que ya descargaste en
los otros dos repos. Sirve para construir y probar TODO el pipeline sin volver
a descargar (util en el sandbox, donde nba_api no tiene salida a internet).

  python -m src.ingest.seed_from_repos

Copia:
  nba_clustering_comp/.../all_teams/{season}/team_stats_raw.csv
      -> data/raw/teams/{season}/team_stats_raw.csv
  nba_tanking/data/processed/{lineups,standings,players}.csv
      -> data/raw/lineups/{...}.csv

NO copia stats de jugador (no existen crudas en tus repos): esas se bajan con
`python -m src.ingest.download_players` en tu maquina.
"""

import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


def seed_teams() -> int:
    src_base = os.path.join(
        config.CLUSTERING_REPO,
        "data/processed/cluster_all_teams",
    )
    # los crudos estan en data/raw/all_teams/{season}/team_stats_raw.csv
    raw_base = os.path.join(config.CLUSTERING_REPO, "data/raw/all_teams")
    n = 0
    for path in sorted(glob.glob(os.path.join(raw_base, "*", "team_stats_raw.csv"))):
        season = os.path.basename(os.path.dirname(path))
        dst_dir = os.path.join(config.RAW_DIR, "teams", season)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dst_dir, "team_stats_raw.csv"))
        n += 1
    print(f"[seed] teams: {n} temporadas copiadas desde {raw_base}")
    return n


def seed_lineups() -> int:
    src = os.path.join(config.TANKING_REPO, "data/processed")
    dst = os.path.join(config.RAW_DIR, "lineups")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in ("lineups.csv", "standings.csv", "players.csv"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, name))
            n += 1
            print(f"[seed] lineups: copiado {name}")
        else:
            print(f"[seed] lineups: FALTA {name} en {src}")
    return n


def main() -> None:
    print("=" * 60)
    print("SEMBRANDO data/raw/ con datos reales existentes")
    print("=" * 60)
    t = seed_teams()
    l = seed_lineups()
    print("-" * 60)
    print(f"[seed] listo. teams={t} temporadas, lineups files={l}")
    print("[seed] player_stats NO sembrado (no hay crudo): usa download_players "
          "en tu maquina.")


if __name__ == "__main__":
    main()

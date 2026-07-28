"""
Cliente comun para descargas de nba_api.

  >>> ESTOS SCRIPTS SE CORREN EN TU MAQUINA (necesitan internet).  <<<
  >>> En el sandbox de Cowork stats.nba.com esta bloqueado (403).  <<<

Provee: lista de temporadas, mapa de equipos, reintentos y sleeps educados.
"""

import time


def season_str(start_year: int) -> str:
    """2019 -> '2019-20'."""
    return f"{start_year}-{str(start_year + 1)[2:]}"


def seasons_range(start_year: int, end_year: int) -> list:
    """seasons_range(2019, 2025) -> ['2019-20', ..., '2025-26']."""
    return [season_str(y) for y in range(start_year, end_year + 1)]


def retry(fn, tries: int = 4, pause: float = 2.0):
    """Reintenta una llamada a la API con backoff lineal."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"   reintento {i + 1}/{tries - 1} ({e.__class__.__name__})...")
            time.sleep(pause * (i + 1))


def team_index():
    """Mapas de equipo: id->idx (0..29) e info, ordenados por abreviatura.

    El indice 0..29 es el MISMO esquema que usa nba_tanking, para que los
    lineups sean compatibles entre proyectos.
    """
    from nba_api.stats.static import teams as nba_teams
    team_list = sorted(nba_teams.get_teams(), key=lambda x: x["abbreviation"])
    id2idx = {t["id"]: i for i, t in enumerate(team_list)}
    return team_list, id2idx

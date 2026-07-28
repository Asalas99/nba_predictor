"""
Campeones, subcampeones y finalistas de conferencia NBA.
Fuente: nba_clustering_comp/src/data/champions_finalists.py (auto-contenido aqui).
Actualiza este dict cada temporada.
"""

CHAMPIONS = {
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets",
    "2023-24": "Boston Celtics",
    "2024-25": "Oklahoma City Thunder",
    "2025-26": "New York Knicks",
}
RUNNERS_UP = {
    "2019-20": "Miami Heat",
    "2020-21": "Phoenix Suns",
    "2021-22": "Boston Celtics",
    "2022-23": "Miami Heat",
    "2023-24": "Dallas Mavericks",
    "2024-25": "Indiana Pacers",
    "2025-26": "San Antonio Spurs",
}
CONFERENCE_FINALISTS = {
    "2019-20": ["Denver Nuggets", "Boston Celtics"],
    "2020-21": ["Atlanta Hawks", "LA Clippers"],
    "2021-22": ["Dallas Mavericks", "Miami Heat"],
    "2022-23": ["Los Angeles Lakers", "Boston Celtics"],
    "2023-24": ["Indiana Pacers", "Minnesota Timberwolves"],
    "2024-25": ["Minnesota Timberwolves", "New York Knicks"],
}


def playoff_status(team_name: str, season: str) -> str:
    """'champion' | 'runner_up' | 'conference_finalist' | 'none'."""
    if CHAMPIONS.get(season) == team_name:
        return "champion"
    if RUNNERS_UP.get(season) == team_name:
        return "runner_up"
    if team_name in CONFERENCE_FINALISTS.get(season, []):
        return "conference_finalist"
    return "none"


def is_labeled_season(season: str) -> bool:
    """True si la temporada ya tiene campeon conocido (excluye la temporada en curso)."""
    return season in CHAMPIONS

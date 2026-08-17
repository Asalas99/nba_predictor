"""
Conferencia de cada equipo NBA (estable 2019-2026).
"""

EAST = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
    "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
    "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards",
}
WEST = {
    "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors",
    "Houston Rockets", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
    "Minnesota Timberwolves", "New Orleans Pelicans", "Oklahoma City Thunder",
    "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
    "San Antonio Spurs", "Utah Jazz",
}


def conference(team_name: str) -> str:
    if team_name in EAST:
        return "East"
    if team_name in WEST:
        return "West"
    return "Unknown"

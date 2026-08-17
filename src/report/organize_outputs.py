"""
Organiza los outputs en carpetas por categoria, para encontrarlos facil:

  outputs/figures/{clustering, correlacion, fuerza, predicciones}/
  outputs/tables/{clustering, correlacion, fuerza, predicciones}/

Se corre al final de run_all.py. Los modulos escriben plano; este paso mueve
cada archivo a su carpeta. Las lecturas entre modulos usan config.find_table,
asi que funcionan este el archivo plano o ya organizado.

  python -m src.report.organize_outputs
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


def organize_dir(base: str) -> dict:
    counts = {}
    for name in os.listdir(base):
        src = os.path.join(base, name)
        if not os.path.isfile(src):
            continue
        cat = config.categorize(name)
        dst_dir = os.path.join(base, cat)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, name))   # sobrescribe la version vieja
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def main():
    for label, base in (("figuras", config.FIGURES_DIR), ("tablas", config.TABLES_DIR)):
        c = organize_dir(base)
        resumen = ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "nada que mover"
        print(f"[organize] {label}: {resumen}")
    print("[organize] outputs ordenados en subcarpetas por categoria.")


if __name__ == "__main__":
    main()

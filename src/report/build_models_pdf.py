"""
Resumen tecnico profundo de los 4 modelos (M1-M4): parametros, parte
matematica, parte de computo y el porque de cada decision.

  python -m src.report.build_models_pdf

Salida: outputs/Resumen_tecnico_modelos.pdf
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

NAVY = colors.HexColor("#1F3A5F")
GREY = colors.HexColor("#F2F2F4")
ORANGE = colors.HexColor("#E0763A")
ss = getSampleStyleSheet()

TITLE = ParagraphStyle("T", parent=ss["Title"], textColor=NAVY, fontSize=24, leading=28)
SUB = ParagraphStyle("S", parent=ss["Normal"], fontSize=12, textColor=colors.grey, alignment=TA_CENTER)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=16, spaceBefore=14, spaceAfter=6)
LBL = ParagraphStyle("LBL", parent=ss["Heading2"], textColor=ORANGE, fontSize=11.5, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10.2, leading=15, alignment=TA_LEFT, spaceAfter=5)
FORMULA = ParagraphStyle("F", parent=ss["Normal"], fontName="Courier", fontSize=9,
                         leading=13, backColor=GREY, borderPadding=6, spaceBefore=3, spaceAfter=7)


def P(t):
    return Paragraph(t, BODY)


def F(t):
    return Paragraph(t, FORMULA)


def block(story, params, math_items, compute, why):
    story.append(Paragraph("Parámetros que usa", LBL))
    story.append(P(params))
    story.append(Paragraph("Parte matemática", LBL))
    for it in math_items:
        story.append(F(it) if it.startswith("~") else P(it))
    story.append(Paragraph("Parte de cómputo", LBL))
    story.append(P(compute))
    story.append(Paragraph("Por qué lo hace así", LBL))
    story.append(P(why))


def build(story):
    story += [Spacer(1, 1.5 * inch), Paragraph("Resumen técnico de los modelos", TITLE),
              Spacer(1, 6), Paragraph("nba_predictor · cascada M1 → M2 → M3 → M4", SUB),
              Spacer(1, 4), Paragraph("Para cada modelo: qué parámetros usa, la parte "
              "matemática, la parte de cómputo y por qué está hecho así.", SUB)]
    story.append(PageBreak())

    # ---- INSUMOS ----
    story.append(Paragraph("0. Insumos que alimentan la cascada", H1))
    story.append(P("Antes de los modelos, tres cantidades derivadas de los datos crudos "
                   "alimentan la cadena. Se resumen aquí porque su matemática reaparece en M1-M4."))
    story.append(Paragraph("PIE (impacto del jugador)", LBL))
    story.append(P("Estadística oficial de la NBA. Mide la porción del total de acciones "
                   "positivas de un partido que aportó el jugador:"))
    story.append(F("PIE = (PTS+FGM+FTM-FGA-FTA+DREB+0.5*OREB+AST+STL+0.5*BLK-PF-TOV)<br/>"
                   "&nbsp;&nbsp;/ (mismos totales sumados de TODO el partido)"))
    story.append(Paragraph("squad_strength (fuerza del plantel)", LBL))
    story.append(P("Promedio del PIE de los jugadores ponderado por minutos, estandarizado "
                   "por temporada:"))
    story.append(F("squad_strength = z_temporada( sum_i (min_i / sum_min) * PIE_i )"))
    story.append(Paragraph("Proyección de jugador (curva de edad)", LBL))
    story.append(P("Para no usar datos del año a predecir, el PIE de cada jugador para T se "
                   "estima desde su historia con recencia + regresión a la media + edad:"))
    story.append(F("PIE_proj = base * mult(edad_T)/mult(edad_ult)      (curva estandar)<br/>"
                   "base = ( n*media_reciente + k*prior ) / (n + k)"))
    story.append(P("La fuerza proyectada del equipo (squad_strength_proj) repite la fórmula "
                   "de squad_strength pero con PIE y minutos proyectados. Correlaciona 0.61 "
                   "con las victorias reales sin mirar el año en curso."))

    # ---- M1 ----
    story.append(PageBreak())
    story.append(Paragraph("M1 — Modelo de victorias (regresión Ridge)", H1))
    block(story,
          params="Cinco variables por equipo-temporada, elegidas por backtest+ablacion: "
                 "<b>continuity</b> (continuidad de plantilla), <b>prior_net_rating</b> "
                 "(diferencial del año previo), <b>avg_age_core</b> (edad del núcleo), "
                 "<b>squad_strength_proj</b> (fuerza proyectada) y <b>best_pie_proj</b> "
                 "(la estrella). El entrenador se probó pero se descartó: empeoraba el "
                 "backtest. Objetivo a predecir: victorias reales de la temporada.",
          math_items=[
              "Es un modelo lineal regularizado. Primero se estandariza cada variable "
              "(media 0, desviación 1) usando solo el conjunto de entrenamiento:",
              "~z = (x - media_train) / desv_train",
              "Luego Ridge ajusta los coeficientes minimizando el error cuadrático MÁS una "
              "penalización L2 que castiga coeficientes grandes (alfa = 1):",
              "~beta = argmin  sum (y - X*beta)^2 + alfa * ||beta||^2",
              "que tiene solución cerrada  beta = (XᵀX + alfa*I)⁻¹ Xᵀy.  La predicción es "
              "y = X*beta, recortada al rango válido [0, 82] victorias.",
          ],
          compute="Se implementa con StandardScaler + Ridge de scikit-learn (solución "
                  "algebraica exacta, sin iteración). La validación es walk-forward: para "
                  "predecir la temporada T se reentrena con TODAS las temporadas anteriores "
                  "y se predice T; se avanza en el tiempo. Coste trivial (matrices 5x5).",
          why="Con pocas filas (30-150) y 5 variables, una regresión normal (OLS) podría "
              "sobreajustar el ruido; la penalización L2 encoge los coeficientes y da "
              "estabilidad (se probó GradientBoosting y RandomForest: peores). La "
              "estandarización hace la penalización justa entre variables y los pesos "
              "comparables (continuity pesa +4.4, prior_net_rating +2.7). El walk-forward "
              "evita fuga temporal. Se recorta a [0,82] porque no existen récords fuera de "
              "ese rango.")
    story.append(Spacer(1, 4))
    story.append(P("<b>Resultado:</b> MAE 7.02 victorias (IC 90% [6.3, 7.8]), mejor que la "
                   "persistencia (8.89) en las 5 temporadas. Sobre el calendario, 6.96 y "
                   "~63% de acierto por partido."))

    # ---- M2 ----
    story.append(PageBreak())
    story.append(Paragraph("M2 — Seeding por conferencia (ordenamiento)", H1))
    block(story,
          params="Las victorias proyectadas por M1 (wins_pred) y la conferencia de cada "
                 "equipo (Este/Oeste, tabla fija).",
          math_items=[
              "No hay entrenamiento: es un ordenamiento. Dentro de cada conferencia se "
              "asigna el seed por el rango de las victorias proyectadas (mayor a menor):",
              "~seed = rank_desc( wins_pred )   dentro de la conferencia",
              "y el tramo sale del seed: 1-6 Playoffs, 7-10 Play-in, 11-15 Lotería.",
          ],
          compute="Un groupby por (temporada, conferencia) y una función de rango. "
                  "Instantáneo. Se valida comparando el seed predicho con el real "
                  "(correlación de Spearman) y el acierto de tramo.",
          why="La NBA siembra a los equipos por su récord dentro de la conferencia; "
              "ordenar las victorias proyectadas es el análogo directo de esa regla. No "
              "necesita un modelo aparte: M2 traduce el número continuo de M1 a la "
              "estructura discreta (posiciones y tramos) que usan los playoffs. Hereda el "
              "error de M1, así que se valida en conjunto.")
    story.append(Spacer(1, 4))
    story.append(P("<b>Resultado:</b> correlación de seed 0.68; ~72% de los clasificados a "
                   "playoffs identificados."))

    # ---- M3 ----
    story.append(PageBreak())
    story.append(Paragraph("M3 — Simulación de playoffs (Monte Carlo)", H1))
    block(story,
          params="Los seeds de M2 (bracket) y la calidad de cada equipo q = wins_pred/82, "
                 "recortada a [0.05, 0.95]. Una ventaja de local HCA = 0.06.",
          math_items=[
              "Probabilidad de que A gane UN partido a B, dadas sus tasas de victoria "
              "(fórmula log5 de Bill James):",
              "~p = (qA - qA*qB) / (qA + qB - 2*qA*qB)",
              "Al que juega de local se le suma la ventaja: p_local = p + HCA (recortada). "
              "El patrón de sedes es 2-2-1-1-1 para el mejor seed.",
              "Probabilidad de ganar la SERIE (al mejor de 7): se calcula exacta por "
              "programación dinámica sobre (juego i, victorias A, victorias B):",
              "~P(i,wa,wb) = p*P(i+1,wa+1,wb) + (1-p)*P(i+1,wa,wb+1)<br/>"
              "base:  wa=4 -> 1 ;  wb=4 -> 0",
              "El campeón surge simulando el bracket completo muchas veces; la probabilidad "
              "de cada evento es su frecuencia:",
              "~P(evento) ≈ (veces que ocurre) / N_sims",
          ],
          compute="La probabilidad de serie usa recursión con memoización (lru_cache): "
                  "exacta y barata. Sobre eso se corre Monte Carlo con N = 20.000 "
                  "simulaciones por temporada; en cada una se muestrean los ganadores de "
                  "serie con un generador aleatorio y se avanza el bracket (R1, semis, "
                  "final de conferencia, Finales). Tarda ~14 s. El error de muestreo es "
                  "~raíz(p(1-p)/N) ≈ 0.3%.",
          why="Los playoffs son un árbol con azar: quién enfrenta a quién en la ronda "
              "siguiente depende de resultados anteriores, así que una fórmula cerrada para "
              "la probabilidad de campeón es enredada. Monte Carlo propaga esa "
              "incertidumbre de forma natural. Se usa log5 porque es la manera "
              "principista de obtener un enfrentamiento directo a partir de dos tasas de "
              "victoria; y se calcula la serie exacta por DP para que la única variación "
              "venga del bracket, no del cálculo de cada serie.")
    story.append(Spacer(1, 4))
    story.append(P("<b>Resultado:</b> Brier 0.059 sobre el campeón; da probabilidades "
                   "realistas y bajas al campeón preseason (lo esperable)."))

    # ---- M4 ----
    story.append(PageBreak())
    story.append(Paragraph("M4 — Probabilidad de campeón (calibración)", H1))
    block(story,
          params="La probabilidad de título de M3 (p_champion) y la proximidad de "
                 "construcción a campeón proyectada (champ_sim_proj). Un peso w que se elige.",
          math_items=[
              "M4 reajusta las probabilidades de M3 empujando a los equipos que se arman "
              "como los campeones. Con z = z-score de la proximidad por temporada:",
              "~score = p_M3 * exp( w * z_proximidad )<br/>"
              "p_M4 = score / suma(score)   (normaliza por temporada)",
              "El peso w se elige por el que minimiza la log-loss del campeón en backtest:",
              "~LL = - promedio( y*log(p) + (1-y)*log(1-p) )",
          ],
          compute="Se prueba w en una rejilla {0, 0.25, 0.5, 0.75, 1.0}, se calcula la "
                  "log-loss de cada uno y se queda el mejor. Operación ligera.",
          why="El reajuste multiplicativo con exp mantiene las probabilidades positivas y "
              "que sumen 1; es una reponderación de estilo bayesiano por un 'prior' de "
              "construcción. Elegir w por backtest es selección de modelo. Aquí el mejor "
              "resultó w = 0: la proximidad de construcción NO mejora la predicción del "
              "campeón (la empeora), así que se descarta. Es la misma disciplina que con el "
              "entrenador: una señal entra solo si el backtest la respalda. Por eso, hoy, "
              "M4 = M3.")
    story.append(Spacer(1, 4))
    story.append(P("<b>Resultado:</b> w = 0 (log-loss 0.261). La mejor estimación de "
                   "campeón es la propia simulación de M3."))

    # ---- cierre ----
    story.append(PageBreak())
    story.append(Paragraph("Panorama de la cascada", H1))
    story.append(P("<b>M1</b> convierte talento proyectado + continuidad + récord previo + "
                   "entrenador en victorias (Ridge). <b>M2</b> ordena esas victorias en "
                   "seeds por conferencia. <b>M3</b> simula el bracket miles de veces "
                   "(log5 + DP + Monte Carlo) para sacar probabilidades por ronda y de "
                   "título. <b>M4</b> intenta calibrar el título con la construcción de "
                   "plantel, y el backtest dice que no aporta, así que se mantiene M3."))
    story.append(P("Dos principios recorren todo: <b>nada de fuga temporal</b> (para "
                   "predecir T solo se usa información anterior a T) y <b>cada variable o "
                   "señal entra solo si mejora el backtest</b>. Por eso el sistema es "
                   "honesto y se puede seguir extendiendo con confianza."))


def main():
    out = os.path.join(config.OUTPUTS_DIR, "Resumen_tecnico_modelos.pdf")
    doc = SimpleDocTemplate(out, pagesize=letter, topMargin=0.8 * inch,
                            bottomMargin=0.7 * inch, leftMargin=0.9 * inch,
                            rightMargin=0.9 * inch, title="Resumen tecnico de modelos")
    story = []
    build(story)
    doc.build(story)
    print(f"[pdf] generado -> {out}")


if __name__ == "__main__":
    main()

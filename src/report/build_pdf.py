"""
Genera el documento PDF explicativo del predictor (paso a paso, con dos niveles:
explicacion sencilla + detalle tecnico) y las graficas embebidas.

  python -m src.report.build_pdf

Salida: outputs/Documentacion_nba_predictor.pdf
"""

import os
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

FIG = config.FIGURES_DIR
NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#5B8DEF")
GREEN = colors.HexColor("#E8F5EE")
GREENB = colors.HexColor("#2FB380")
GREY = colors.HexColor("#F2F2F4")

# ---- estilos ----
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17,
                    spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=13,
                    spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10.3, leading=15,
                      alignment=TA_LEFT, spaceAfter=6)
SIMPLE = ParagraphStyle("Simple", parent=BODY, backColor=GREEN, borderColor=GREENB,
                        borderWidth=0.6, borderPadding=7, leftIndent=2, rightIndent=2,
                        spaceBefore=4, spaceAfter=8)
DETAIL = ParagraphStyle("Detail", parent=BODY)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, textColor=colors.grey,
                     alignment=TA_CENTER, spaceAfter=12)
TITLE = ParagraphStyle("Title", parent=ss["Title"], textColor=NAVY, fontSize=26,
                       leading=30)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=12, textColor=colors.grey,
                     alignment=TA_CENTER)
FORMULA = ParagraphStyle("Formula", parent=ss["Normal"], fontName="Courier",
                         fontSize=9, leading=13, backColor=GREY, borderPadding=6,
                         spaceBefore=4, spaceAfter=8, alignment=TA_CENTER)
CELL = ParagraphStyle("Cell", parent=ss["Normal"], fontSize=8.3, leading=11)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
CELLC = ParagraphStyle("CellC", parent=CELL, fontName="Courier", fontSize=8)


def simple(text):
    return Paragraph(f'<b>En simple.</b> {text}', SIMPLE)


def detail(text):
    return Paragraph(f'<b>En detalle.</b> {text}', DETAIL)


def para(text):
    return Paragraph(text, BODY)


def figure(name, caption, max_w=6.4):
    path = config.find_figure(name)
    if not os.path.exists(path):
        return Paragraph(f"[falta figura: {name}]", CAP)
    w, h = PILImage.open(path).size
    disp_w = min(max_w * inch, w)
    disp_h = disp_w * h / w
    max_h = 7.2 * inch
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h * w / h
    return [KeepTogether([Image(path, width=disp_w, height=disp_h),
                          Spacer(1, 3), Paragraph(caption, CAP)])]


def table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build(story):
    # ---------- PORTADA ----------
    story += [Spacer(1, 1.6 * inch),
              Paragraph("nba_predictor", TITLE),
              Spacer(1, 6),
              Paragraph("Cómo funciona el sistema, paso a paso", SUB),
              Spacer(1, 4),
              Paragraph("Estilo de equipo · construcción de plantel · fuerza real · "
                        "predicción de victorias", SUB),
              Spacer(1, 0.5 * inch)]
    story += [Paragraph(
        "Cada sección tiene dos niveles: una explicación sencilla (recuadro verde) "
        "y el detalle técnico debajo. Puedes leer solo los recuadros para una visión "
        "general, o todo para el detalle.", CAP)]
    story.append(PageBreak())

    # ---------- 1. VISION GENERAL ----------
    story.append(Paragraph("1. Qué hace el sistema", H1))
    story.append(simple(
        "El sistema mira a cada equipo de la NBA y responde tres preguntas: ¿cómo "
        "juega? (su estilo), ¿cómo está armado? (su tipo de plantilla), y ¿qué tan "
        "bueno es de verdad? (la fuerza de sus jugadores). Con eso, predice cuántos "
        "partidos va a ganar la próxima temporada — antes de que empiece."))
    story.append(detail(
        "nba_predictor es un pipeline reproducible que descarga datos oficiales de la "
        "NBA (equipos, jugadores, quintetos, entrenadores), los limpia, y aplica una "
        "cadena de modelos: (1) clustering de estilo de equipo, (2) clustering de "
        "construcción de plantilla a partir de arquetipos de jugador, (3) una métrica "
        "de fuerza del plantel basada en impacto individual, (4) una proyección de "
        "cada jugador desde su historia, y (5) un modelo de victorias validado con "
        "backtesting. Todo corre con un solo comando (run_all.py) y se actualiza cada "
        "temporada."))
    story.append(para(
        "El objetivo final es una cascada completa: proyectar victorias, ordenar la "
        "tabla, simular playoffs y estimar al campeón. Este documento cubre lo ya "
        "construido y validado hasta el modelo de victorias (M1)."))

    # ---------- 2. DATOS ----------
    story.append(Paragraph("2. Los datos y el flujo", H1))
    story.append(simple(
        "Bajamos cuatro cosas de la NBA: estadísticas de equipos, de jugadores, los "
        "quintetos que jugaron juntos, y quién dirigió cada equipo. Todo pasa por una "
        "limpieza y queda listo para los modelos."))
    story.append(detail(
        "Fuentes vía nba_api: stats avanzadas de equipo (leaguedashteamstats), de "
        "jugador base+avanzadas (leaguedashplayerstats), quintetos (leaguedashlineups), "
        "standings y entrenadores (commonteamroster). Cobertura 2019-20 a 2025-26 "
        "(7 temporadas, 210 equipos-temporada, ~3.700 jugador-temporadas). El pipeline "
        "es idéntico con datos reales; la descarga corre en tu máquina y el "
        "procesamiento en cualquier lado."))
    story.append(para("<b>Flujo:</b> descarga → limpieza → estilo + arquetipos + "
                      "fuerza → proyección → modelo de victorias → gráficas."))

    # ---------- 3. ESTILO ----------
    story.append(PageBreak())
    story.append(Paragraph("3. Paso 1 — Estilo de juego", H1))
    story.append(simple(
        "Agrupamos a los equipos según cómo juegan: rápido o lento, ofensivos o "
        "defensivos, etc. Salen tres familias de estilo. Así podemos ver, por ejemplo, "
        "qué estilos suelen llegar a campeones."))
    story.append(detail(
        "KMeans (k=3) sobre 8 features avanzadas escaladas por temporada: OFF_RATING, "
        "DEF_RATING, AST%, OREB%, DREB%, TOV%, TS%, PACE. Resultan tres estilos: "
        "'Defensivos lentos', 'Ofensivos rápidos' y un grupo de equipos flojos. Cada "
        "equipo-temporada queda ubicado en este espacio; abajo, proyección a 2 "
        "componentes principales con campeones y finalistas etiquetados."))
    story += figure("style_pca_labeled.png",
                    "Espacio de estilo 2019-26. ★ = campeón. Los campeones tienden al "
                    "lado derecho (buena ofensiva/eficiencia).")

    # ---------- 4. ARQUETIPOS Y PLANTEL ----------
    story.append(PageBreak())
    story.append(Paragraph("4. Paso 2 — Arquetipos y tipo de plantel", H1))
    story.append(simple(
        "Primero clasificamos a cada jugador por lo que hace en cancha (anotador, "
        "creador, tirador, interior…), sin importar su posición oficial. Luego "
        "describimos a cada equipo por la mezcla de esos roles, y agrupamos equipos "
        "que se construyen parecido."))
    story.append(detail(
        "Arquetipos: KMeans (6 grupos) sobre tasas por 36 min (puntos, asistencias, "
        "rebotes, robos, tapones, volumen de triple) más USG%, AST%, REB%, TS%. Tipo "
        "de plantel: cada equipo se representa por la fracción de minutos en cada "
        "arquetipo, y otro KMeans (4 tipos) agrupa las construcciones. Hallazgo: el "
        "tipo 'creador principal + jugadores de rol' concentra 4 de los últimos "
        "campeones — la construcción clásica de estrella rodeada de especialistas."))
    story.append(para(
        "El mapa siguiente muestra en qué estadística resalta cada uno de los 6 "
        "arquetipos (rojo = por encima del resto, azul = por debajo): el creador tiene "
        "el uso más alto; el protector de aro, rebote y tapones; el wing, volumen de "
        "triple; y el jugador de rol, bajo en todo."))
    story += figure("archetype_profiles_heatmap.png",
                    "Perfil de los 6 arquetipos (z-score entre arquetipos). Rojo = la "
                    "estadística en la que ese arquetipo destaca.")
    story += figure("roster_pca_labeled.png",
                    "Espacio de tipo de plantel. ★ = campeón.")

    # ---------- 5. FUERZA DEL PLANTEL ----------
    story.append(PageBreak())
    story.append(Paragraph("5. Paso 3 — Fuerza del plantel", H1))
    story.append(simple(
        "Necesitamos medir qué tan bueno es de verdad cada equipo. Sumamos el impacto "
        "de sus jugadores (ponderado por los minutos que juegan). Un primer intento "
        "con otra métrica falló — no tenía relación con ganar — así que lo cambiamos "
        "por una que sí funciona."))
    story.append(detail(
        "squad_strength = PIE (Player Impact Estimate) ponderado por minutos, "
        "estandarizado por temporada. Un intento previo heredado (APM del proyecto de "
        "tanking) daba correlación ~0.00 con victorias en datos reales — no servía. La "
        "métrica basada en PIE correlaciona 0.76 con las victorias y separa campeones "
        "(percentil 81 vs 52 del APM). Es la señal de talento del sistema."))

    story.append(Paragraph("¿Qué es el PIE y cómo se calcula?", H2))
    story.append(simple(
        "El PIE mide qué porción de todo lo bueno que pasó en los partidos de un "
        "jugador la aportó él. Se suma lo positivo (puntos, canastas, rebotes, "
        "asistencias, robos, tapones) y se resta lo negativo (tiros fallados, "
        "pérdidas, faltas); ese total del jugador se divide entre el mismo total de "
        "TODOS los jugadores del partido. Un jugador promedio ronda 0.10 (10%); una "
        "estrella, 0.15-0.20."))
    story.append(detail(
        "El PIE es una estadística avanzada oficial de la NBA: no la calculamos "
        "nosotros, la tomamos ya lista por jugador y temporada vía nba_api "
        "(leaguedashplayerstats, tipo Advanced). Su fórmula es un cociente entre lo "
        "que produce el jugador y lo que se produce en todo el partido (ambos "
        "equipos):"))
    story.append(Paragraph(
        "PIE = (PTS + FGM + FTM - FGA - FTA + DREB + 0.5*OREB<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ AST + STL + 0.5*BLK - PF - TOV)<br/>"
        "&nbsp;&nbsp;/ (mismos totales sumados de TODO el partido)", FORMULA))
    story.append(para(
        "Donde PTS=puntos, FGM/FGA=canastas anotadas/intentadas, FTM/FTA=libres, "
        "DREB/OREB=rebotes defensivos/ofensivos, AST=asistencias, STL=robos, "
        "BLK=tapones, PF=faltas, TOV=pérdidas. Como es una proporción del partido, "
        "es comparable entre jugadores y temporadas sin depender del ritmo del equipo. "
        "La fuerza del equipo (squad_strength) es el promedio del PIE de sus jugadores "
        "ponderado por los minutos que juega cada uno: los titulares pesan más que el "
        "fondo del banquillo."))
    story += figure("strength_ranking_2526.png",
                    "Fuerza del plantel por equipo, temporada 2025-26.", max_w=5.2)

    # ---------- 6. PROYECCION JUGADOR ----------
    story.append(PageBreak())
    story.append(Paragraph("6. Paso 4 — Proyección de cada jugador", H1))
    story.append(simple(
        "Para predecir el futuro no podemos usar las estadísticas del año que aún no "
        "pasa. Así que a cada jugador le proyectamos lo que va a aportar usando su "
        "historia y su edad (los jóvenes mejoran, los veteranos declinan). Si es "
        "novato sin historia, usamos un valor base."))
    story.append(detail(
        "Para cada jugador y temporada T se estima su PIE usando solo datos < T: "
        "promedio de sus temporadas previas ponderado por recencia, con regresión a la "
        "media para muestras chicas, y ajuste por curva de edad. Probamos dos curvas: "
        "una estimada de nuestros datos (cambio de PIE año a año por edad) y una "
        "estándar (pico ~26). Ambas superan al baseline naive (usar el PIE del año "
        "pasado sin ajuste):"))
    story.append(table([
        ["Método", "MAE (error)", "Correlación"],
        ["Naive (PIE del año pasado)", "0.0176", "0.697"],
        ["Curva de edad de datos", "0.0167", "0.718"],
        ["Curva de edad estándar", "0.0165", "0.728"],
    ], col_widths=[2.6 * inch, 1.6 * inch, 1.6 * inch]))
    story.append(Spacer(1, 8))
    story.append(para("Esto cubre el caso clave: un jugador que cambió de equipo o aún "
                      "no jugó esta temporada recibe una proyección realista desde su "
                      "trayectoria."))

    # ---------- 7. FUERZA PROYECTADA + CONTINUIDAD ----------
    story.append(PageBreak())
    story.append(Paragraph("7. Paso 5 — Fuerza proyectada y continuidad", H1))
    story.append(simple(
        "Juntamos las proyecciones de los jugadores de cada equipo para estimar la "
        "fuerza del plantel del año que viene, sin usar nada de ese año. También "
        "medimos cuánto del equipo se mantiene respecto al año anterior."))
    story.append(detail(
        "squad_strength_proj = PIE proyectado ponderado por minutos proyectados, "
        "agregado por equipo. continuity = fracción de minutos proyectados que vienen "
        "de jugadores que ya estaban el año anterior. Aunque es 100% pre-temporada, la "
        "fuerza proyectada correlaciona 0.61 con las victorias reales, y la continuidad "
        "0.47. Son señales legítimas para pronosticar."))

    # ---------- 8. ENTRENADOR ----------
    story.append(Paragraph("8. Paso 6 — El entrenador", H1))
    story.append(simple(
        "Un técnico imprime su forma de jugar y suele rendir por encima o por debajo "
        "de lo que dice el talento. Extraemos su 'huella de estilo' y su historial de "
        "victorias sobre lo esperado, usando solo su pasado."))
    story.append(detail(
        "Por equipo-temporada calculamos, con la historia del coach < T: huella de "
        "estilo (pace y lean ofensivo/defensivo típicos) y residual de rendimiento "
        "(victorias reales menos las esperadas por talento), con shrinkage. El residual "
        "identifica bien a técnicos que suman (Mazzulla, Daigneault, Atkinson). Sin "
        "embargo, al probarlo en el backtest, incluirlo en M1 EMPEORABA la predicción "
        "(subía el error) — así que, por disciplina, el entrenador NO entra en el "
        "modelo final. Es un ejemplo de una señal plausible que los datos no "
        "respaldaron; queda como candidata a reformular en el futuro."))

    # ---------- 9. M1 ----------
    story.append(PageBreak())
    story.append(Paragraph("9. Paso 7 — Modelo de victorias (M1)", H1))
    story.append(simple(
        "Con todo lo anterior, un modelo estima cuántos partidos ganará cada equipo. "
        "Lo probamos 'como si viviéramos en el pasado': entrenamos con temporadas "
        "viejas y predecimos la siguiente, sin hacer trampa. Le gana a simplemente "
        "asumir que un equipo repetirá sus victorias del año pasado."))
    story.append(detail(
        "M1 es una regresión Ridge sobre 5 features: squad_strength_proj, continuity, "
        "prior_net_rating (diferencial del año previo), best_pie_proj (la estrella) y "
        "avg_age_core (edad del núcleo). Estas se eligieron por backtest y ablación: "
        "cada una entra solo si baja el error. Validación walk-forward: para cada "
        "temporada de prueba se entrena con todas las anteriores y se predice esa."))

    story.append(Paragraph("Las 5 variables de M1, una por una", H2))
    story.append(para(
        "El 'peso' es cuánto empuja cada una la predicción (coeficiente estandarizado): "
        "a mayor valor absoluto, más influye; el signo indica si suma o resta victorias."))

    def C(t, st=CELL):
        return Paragraph(t, st)

    feats_tbl = [
        [C("Variable", CELLH), C("Qué es (en simple)", CELLH),
         C("Cómo se calcula (técnico)", CELLH), C("Peso", CELLH)],
        [C("<b>continuity</b><br/>(continuidad)"),
         C("Cuánto del equipo se mantiene respecto al año pasado. Los equipos que "
           "conservan su núcleo rinden de forma más estable."),
         C("Fracción de los minutos proyectados que aportan jugadores que ya estaban "
           "en el equipo la temporada anterior (0 a 1)."),
         C("+4.4", CELLC)],
        [C("<b>prior_net_rating</b><br/>(diferencial previo)"),
         C("El diferencial de puntos del año pasado. Predice mejor el futuro que el "
           "récord de victorias (tiene menos ruido de suerte)."),
         C("NET_RATING de la temporada T-1 del mismo equipo."),
         C("+2.7", CELLC)],
        [C("<b>avg_age_core</b><br/>(edad del núcleo)"),
         C("La edad promedio de la rotación. Un núcleo más asentado tiende a ganar más; "
           "los muy jóvenes aún están desarrollándose."),
         C("Edad media de los 8 con más minutos proyectados, ponderada por minutos."),
         C("+2.3", CELLC)],
        [C("<b>squad_strength_proj</b><br/>(fuerza proyectada)"),
         C("Qué tan bueno se espera que sea el equipo por el talento de sus jugadores, "
           "estimado desde su historia — sin mirar el año en curso."),
         C("PIE proyectado de cada jugador (edad + recencia + trayectoria) ponderado "
           "por minutos, estandarizado por temporada."),
         C("+2.1", CELLC)],
        [C("<b>best_pie_proj</b><br/>(la estrella)"),
         C("El nivel del mejor jugador. En la NBA el techo lo marca la superestrella, "
           "que la fuerza promedio diluye entre 8-10 jugadores."),
         C("PIE proyectado del jugador más impactante del plantel."),
         C("+1.8", CELLC)],
    ]
    t = Table(feats_tbl, colWidths=[1.15 * inch, 2.05 * inch, 2.35 * inch, 0.55 * inch],
              hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(para(
        "Lectura: la continuidad y el diferencial previo pesan fuerte, y la fuerza del "
        "plantel se reparte entre el promedio y la estrella. El entrenador se probó pero "
        "se descartó porque empeoraba el backtest (ver sección 8)."))

    story.append(Paragraph("¿Qué es el MAE de victorias?", H2))
    story.append(simple(
        "El MAE (error absoluto medio) es cuántas victorias, en promedio, se equivoca "
        "la predicción. Un MAE de 7.0 significa que, en promedio, el número predicho "
        "se aleja del real unas 7 victorias — hacia arriba o hacia abajo. Cuanto "
        "más bajo, mejor. Ejemplo: si predecimos 48 victorias y el equipo gana 45, ese "
        "error es de 3."))
    story.append(detail(
        "MAE = promedio de |victorias_predichas - victorias_reales| sobre todos los "
        "equipos de prueba. Se prefiere al error cuadrático porque está en las mismas "
        "unidades (victorias) y es fácil de interpretar. Comparamos el MAE del modelo "
        "contra dos referencias: asumir que el equipo repetirá sus victorias del año "
        "pasado (8.89) y asumir la media de la liga (~10.7). M1 logra 7.02, mejor que "
        "ambas en todas las temporadas."))
    story.append(table([
        ["Temporada", "MAE M1", "MAE 'año pasado'", "Correlación M1"],
        ["2021-22", "7.99", "8.33", "0.549"],
        ["2022-23", "5.77", "8.07", "0.738"],
        ["2023-24", "6.21", "8.47", "0.819"],
        ["2024-25", "7.20", "9.13", "0.728"],
        ["2025-26", "7.93", "10.43", "0.682"],
        ["GLOBAL", "7.02", "8.89", "0.693"],
    ], col_widths=[1.5 * inch, 1.2 * inch, 1.9 * inch, 1.6 * inch]))
    story.append(Spacer(1, 8))
    story += figure("m1_pred_vs_real.png",
                    "Predicho vs real (walk-forward). El modelo acierta la tendencia y "
                    "comprime los extremos, típico de un modelo pre-temporada.", max_w=4.8)

    # ---------- 10. QUE USA / QUE NO ----------
    story.append(PageBreak())
    story.append(Paragraph("10. Qué usa y qué no (sin trampa)", H1))
    story.append(simple(
        "Para predecir una temporada, el modelo NUNCA mira los resultados de esa "
        "temporada (ni victorias ni estadísticas de ese año). Solo usa el pasado: la "
        "historia de los jugadores, las victorias del año anterior y el historial del "
        "entrenador. Lo único del año en curso que usa es quién está en cada equipo "
        "(el roster), que es información que sí se conoce antes de empezar."))
    story.append(detail(
        "Sin fuga de rendimiento: squad_strength_proj y best_pie_proj se construyen con "
        "PIE proyectado de temporadas < T; prior_net_rating es de T-1; avg_age_core es "
        "conocido antes de empezar. Matices honestos: (a) el roster de T se tomó de datos reales, así "
        "que incluye traspasos de media temporada — para el backtest es una suposición "
        "razonable, y para la predicción viva de 2026-27 se resuelve dando el roster "
        "conocido antes de empezar; (b) los minutos de novatos usan un placeholder — es "
        "una fuga menor a corregir. Nada de esto toca el resultado que se predice."))

    # ---------- 11. PROXIMIDAD A CAMPEONES ----------
    story.append(PageBreak())
    story.append(Paragraph("11. Fuerza vs. parecido a campeones", H1))
    story.append(simple(
        "Combinamos dos ideas: qué tan fuerte es un plantel y qué tanto se parece su "
        "estilo al de los campeones históricos. Los equipos arriba-derecha son los que "
        "tienen las dos cosas."))
    story.append(detail(
        "La proximidad a campeón es la distancia del estilo de un equipo al centroide "
        "de los campeones en el espacio de 8 features. El mapa de contendientes cruza "
        "esa proximidad con la fuerza del plantel para la temporada actual. Nota: usa "
        "el estilo realizado de 2025-26, así que es descriptivo del año, no un "
        "pronóstico puro — pero valida el enfoque: el campeón real (Knicks) aparece "
        "arriba-derecha."))
    story += figure("contention_map_2526.png",
                    "Mapa de contendientes 2025-26. Arriba-derecha: fuertes y con "
                    "estilo de campeón (Spurs, Thunder, Cavaliers, Knicks, Celtics).")
    story.append(PageBreak())
    story += figure("strength_heatmap.png",
                    "Fuerza del plantel de cada equipo a lo largo de las temporadas "
                    "(verde = fuerte).", max_w=5.4)

    # ---------- 12. CIERRE ----------
    story.append(PageBreak())
    story.append(Paragraph("12. Cómo se actualiza y qué sigue", H1))
    story.append(simple(
        "Cada año se bajan los datos nuevos, se corre un comando, y todo se recalcula "
        "solo — incluidas las gráficas. La cascada completa (victorias, seeding, "
        "playoffs y campeón) ya está construida."))
    story.append(detail(
        "Actualización anual: descargar la temporada nueva (download_teams/players/"
        "lineups/coaches), correr run_all.py, y añadir el campeón en champions.py. La "
        "cascada M1-M4 está lista y validada. Próximos pasos del roadmap: predicción "
        "partido a partido con el calendario, mejor proyección de minutos, y "
        "actualización en vivo (Elo) para capturar lesiones y forma durante el año. "
        "Métrica actual: MAE 7.02 victorias, mejor que la persistencia (8.89); acierto "
        "por partido ~63%."))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Archivos clave: run_all.py (pipeline), PLAN.md y "
                          "PREDICTOR_DESIGN.md (diseño), outputs/ (tablas y figuras).",
                          CAP))


def main():
    out = os.path.join(config.OUTPUTS_DIR, "Documentacion_nba_predictor.pdf")
    doc = SimpleDocTemplate(out, pagesize=letter, topMargin=0.8 * inch,
                            bottomMargin=0.7 * inch, leftMargin=0.9 * inch,
                            rightMargin=0.9 * inch, title="Documentacion nba_predictor")
    story = []
    build(story)
    doc.build(story)
    print(f"[pdf] generado -> {out}")


if __name__ == "__main__":
    main()

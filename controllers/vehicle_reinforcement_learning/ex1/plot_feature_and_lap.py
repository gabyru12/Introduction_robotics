"""
Grid 1x2, ambos sobre a MESMA volta (LAP_TO_PLOT):
  - Esquerda: variação de FEATURE_TO_PLOT ao longo dos steps dessa volta
    (eixo X normalizado ao intervalo de steps/sim_time dessa volta).
  - Direita: percurso (x, y) dessa mesma volta, desenhado sobre a pista,
    colorido consoante o drift_angle.

Para ver a variação da feature ao longo do episódio inteiro, com as
voltas separadas por linhas verticais, usa o plot_feature_all_laps.py.

Basta editar as variáveis em CONFIG e correr o script:
    python plot_feature_and_lap.py
"""

import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


# =========================================================
# CONFIG (edita só isto)
# =========================================================

# Caminho para o JSON gerado pelo TrackingEvalCallback
METRICS_JSON_PATH = "logs\\eval_metrics\\inference_metrics_12.json"

# Qual evalback e episódio queres ver
EVALCALLBACK_NUMBER = 12
EPISODE_NUMBER = 1

# Métrica a mostrar no gráfico da esquerda (variação ao longo do episódio)
FEATURE_TO_PLOT = "dist2centerline"

# Eixo X do gráfico da esquerda: "step" ou "sim_time"
X_AXIS = "step"

# Qual volta mostrar em detalhe no gráfico da direita (1 = primeira volta detetada)
LAP_TO_PLOT = 1


# Pista: círculo central (interior) e circunferência exterior, ambos
# centrados no centro da pista (não no centro/origem do mundo)
TRACK_CENTER = (150, 0)
TRACK_INNER_RADIUS = 30
TRACK_OUTER_RADIUS = 50

# Limiares (graus) para colorir o trajeto do lado direito consoante o drift_angle
DRIFT_LOW_THRESHOLD = 30    # < 30            -> verde
DRIFT_HIGH_THRESHOLD = 60   # >= 30 e < 60    -> amarelo
                             # >= 60           -> vermelho

# Pasta e nome do ficheiro de saída
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = f"{FEATURE_TO_PLOT}_and_lap_{LAP_TO_PLOT}_evalback_{EVALCALLBACK_NUMBER}_episode_{EPISODE_NUMBER}.png"


# =========================================================
# Lógica (não precisas de mexer daqui para baixo)
# =========================================================

def load_metrics(json_path: str) -> list[dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def get_episode_rows(rows: list[dict], evalback: int, episode: int) -> list[dict]:
    ep_rows = [r for r in rows if r["evalback"] == evalback and r["episode"] == episode]
    ep_rows.sort(key=lambda r: r["step"])
    return ep_rows


def split_into_laps(rows: list[dict], track_center: tuple[float, float]) -> list[list[dict]]:
    if not rows:
        return []

    cx, cy = track_center
    laps: list[list[dict]] = []
    current_lap: list[dict] = []
    cumulative_angle = 0.0
    prev_angle = None

    for row in rows:
        angle = math.atan2(row["y"] - cy, row["x"] - cx)

        if prev_angle is not None:
            delta = angle - prev_angle
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            cumulative_angle += delta

        current_lap.append(row)
        prev_angle = angle

        if abs(cumulative_angle) >= 2 * math.pi:
            laps.append(current_lap)
            current_lap = [row]
            cumulative_angle = 0.0

    if len(current_lap) > 1:
        laps.append(current_lap)

    return laps


def drift_color(drift_angle: float, low: float, high: float) -> str:
    drift_angle = abs(drift_angle)
    if drift_angle < low:
        return "green"
    if drift_angle < high:
        return "gold"
    return "red"


# ---------------------------------------------------------
# Gráfico da esquerda: feature ao longo do tempo + limites de volta
# ---------------------------------------------------------

def plot_feature_for_lap(ax, lap: list[dict], feature: str, x_axis: str) -> None:
    x = [r[x_axis] for r in lap]
    y = [r[feature] for r in lap]
    ax.plot(x, y, color="tab:blue", linewidth=1.2)

    ax.set_xlim(min(x), max(x))
    #ax.set_title(f"{feature} — volta")
    ax.set_xlabel("Step" if x_axis == "step" else "Sim time (s)")
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------
# Gráfico da direita: percurso de uma volta, colorido por drift_angle
# ---------------------------------------------------------

def draw_track(ax, center: tuple[float, float], inner_radius: float, outer_radius: float) -> None:
    inner = Circle(center, inner_radius, fill=False, edgecolor="black", linewidth=3, zorder=1)
    outer = Circle(center, outer_radius, fill=False, edgecolor="black", linewidth=3, zorder=1)
    ax.add_patch(inner)
    ax.add_patch(outer)


def plot_lap_track(ax, lap: list[dict], low: float, high: float,
                    track_center: tuple[float, float], inner_radius: float, outer_radius: float) -> None:
    draw_track(ax, track_center, inner_radius, outer_radius)

    x = [r["x"] for r in lap]
    y = [r["y"] for r in lap]
    drift = [r.get("drift_angle", 0.0) for r in lap]

    points = list(zip(x, y))
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    seg_colors = [drift_color(drift[i], low, high) for i in range(len(points) - 1)]

    lc = LineCollection(segments, colors=seg_colors, linewidths=2, zorder=2)
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], color="black", marker="o", s=30, zorder=3)  # início da volta

    margin = outer_radius * 0.1
    ax.set_xlim(track_center[0] - outer_radius - margin, track_center[0] + outer_radius + margin)
    ax.set_ylim(track_center[1] - outer_radius - margin, track_center[1] + outer_radius + margin)
    ax.set_aspect("equal", adjustable="box")

    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label=f"drift_angle < {low}"),
        Line2D([0], [0], color="gold", lw=2, label=f"{low} <= drift_angle < {high}"),
        Line2D([0], [0], color="red", lw=2, label=f"drift_angle >= {high}"),
        Line2D([0], [0], color="black", lw=3, label="borders"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)


def main() -> None:
    rows = load_metrics(METRICS_JSON_PATH)
    ep_rows = get_episode_rows(rows, EVALCALLBACK_NUMBER, EPISODE_NUMBER)

    if not ep_rows:
        print(f"Não há dados para evalback {EVALCALLBACK_NUMBER}, episódio {EPISODE_NUMBER}.")
        return

    laps = split_into_laps(ep_rows, TRACK_CENTER)
    print(f"Voltas detetadas: {len(laps)}")

    if not laps or LAP_TO_PLOT > len(laps):
        print(f"LAP_TO_PLOT={LAP_TO_PLOT} inválido (só há {len(laps)} volta(s) detetada(s)).")
        return

    lap = laps[LAP_TO_PLOT - 1]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    plot_feature_for_lap(ax_left, lap, FEATURE_TO_PLOT, X_AXIS)
    plot_lap_track(ax_right, lap, DRIFT_LOW_THRESHOLD, DRIFT_HIGH_THRESHOLD,
                    TRACK_CENTER, TRACK_INNER_RADIUS, TRACK_OUTER_RADIUS)
    #ax_right.set_title(f"Lap {LAP_TO_PLOT}")

    #fig.suptitle(f"Evalback {EVALCALLBACK_NUMBER}, episódio {EPISODE_NUMBER}")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()

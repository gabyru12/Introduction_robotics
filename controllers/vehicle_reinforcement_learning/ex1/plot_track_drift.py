"""
Plota um grid 1 x N_VOLTAS com o percurso (x, y) de cada uma das
N_VOLTAS primeiras voltas do evalback EVALCALLBACK_NUMBER, no episódio
EPISODE_NUMBER — desenhando também a pista (círculo interior e exterior)
por baixo do trajeto, e colorindo o trajeto consoante o drift_angle:
    < 30            -> verde
    >= 30 e < 60     -> amarelo
    >= 60            -> vermelho

Como o JSON não guarda diretamente o número da volta, este script deteta
voltas de forma precisa através do ângulo acumulado em torno do centro da
pista (TRACK_CENTER): a cada step somamos a variação angular da posição do
carro relativamente a esse centro, e consideramos que uma volta terminou
assim que esse ângulo acumulado atinge 360 graus.

Basta editar as variáveis em CONFIG e correr o script:
    python plot_track_drift.py
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
METRICS_JSON_PATH = "logs\\eval_metrics\\drift_12h_norm_rewards_timestep_32_facilitate_drifting_130_170_progress_reward_01.json"

# Qual evalback e episódio queres ver
EVALCALLBACK_NUMBER = 12
EPISODE_NUMBER = 3

# Quantas voltas mostrar (uma coluna do grid por volta) -> grid 1 x N_VOLTAS
N_VOLTAS = 3


# Pista: círculo central (interior) e circunferência exterior, ambos
# centrados no mesmo ponto (centro da pista, não o centro/origem do mundo)
TRACK_CENTER = (150, 0)
TRACK_INNER_RADIUS = 30
TRACK_OUTER_RADIUS = 50

# Limiares (graus) para colorir o trajeto consoante o drift_angle
DRIFT_LOW_THRESHOLD = 30    # < 30            -> verde
DRIFT_HIGH_THRESHOLD = 60   # >= 30 e < 60    -> amarelo
                             # >= 60           -> vermelho

# Pasta e nome do ficheiro de saída
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = f"track_drift_evalback_{EVALCALLBACK_NUMBER}_episode_{EPISODE_NUMBER}.png"


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


def draw_track(ax, center: tuple[float, float], inner_radius: float, outer_radius: float) -> None:
    inner = Circle(center, inner_radius, fill=False, edgecolor="black", linewidth=3, zorder=1)
    outer = Circle(center, outer_radius, fill=False, edgecolor="black", linewidth=3, zorder=1)
    ax.add_patch(inner)
    ax.add_patch(outer)


def plot_lap_on_axis(ax, lap: list[dict], low: float, high: float) -> None:
    x = [r["x"] for r in lap]
    y = [r["y"] for r in lap]
    drift = [r.get("drift_angle", 0.0) for r in lap]

    points = list(zip(x, y))
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    seg_colors = [drift_color(drift[i], low, high) for i in range(len(points) - 1)]

    lc = LineCollection(segments, colors=seg_colors, linewidths=2, zorder=2)
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], color="black", marker="o", s=30, zorder=3)  # início da volta


def plot_grid(
    laps: list[list[dict]],
    n_voltas: int,
    evalback: int,
    episode: int,
    track_center: tuple[float, float],
    track_inner_radius: float,
    track_outer_radius: float,
    drift_low: float,
    drift_high: float,
    output_dir: str,
    output_filename: str,
) -> None:
    laps_to_plot = laps[:n_voltas]
    if not laps_to_plot:
        print("Não foi possível identificar nenhuma volta completa com esta configuração.")
        return

    n = len(laps_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]

    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label=f"drift_angle < {drift_low}"),
        Line2D([0], [0], color="gold", lw=2, label=f"{drift_low} <= drift_angle < {drift_high}"),
        Line2D([0], [0], color="red", lw=2, label=f"drift_angle >= {drift_high}"),
        Line2D([0], [0], color="black", lw=3, label="borders"),
    ]

    for i, lap in enumerate(laps_to_plot):
        ax = axes[i]
        draw_track(ax, track_center, track_inner_radius, track_outer_radius)
        plot_lap_on_axis(ax, lap, drift_low, drift_high)

        margin = track_outer_radius * 0.1
        ax.set_xlim(track_center[0] - track_outer_radius - margin, track_center[0] + track_outer_radius + margin)
        ax.set_ylim(track_center[1] - track_outer_radius - margin, track_center[1] + track_outer_radius + margin)

        #ax.set_title(f"Volta {i + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    #fig.suptitle(f"Percurso — evalback {evalback}, episódio {episode}")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main() -> None:
    rows = load_metrics(METRICS_JSON_PATH)
    ep_rows = get_episode_rows(rows, EVALCALLBACK_NUMBER, EPISODE_NUMBER)

    if not ep_rows:
        print(f"Não há dados para evalback {EVALCALLBACK_NUMBER}, episódio {EPISODE_NUMBER}.")
        return

    laps = split_into_laps(ep_rows, TRACK_CENTER)
    print(f"Voltas detetadas: {len(laps)}")

    plot_grid(
        laps=laps,
        n_voltas=N_VOLTAS,
        evalback=EVALCALLBACK_NUMBER,
        episode=EPISODE_NUMBER,
        track_center=TRACK_CENTER,
        track_inner_radius=TRACK_INNER_RADIUS,
        track_outer_radius=TRACK_OUTER_RADIUS,
        drift_low=DRIFT_LOW_THRESHOLD,
        drift_high=DRIFT_HIGH_THRESHOLD,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()

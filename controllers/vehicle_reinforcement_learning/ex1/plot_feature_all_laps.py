"""
Plota a variação de FEATURE_TO_PLOT ao longo de todo o episódio (evalback
EVALCALLBACK_NUMBER, episódio EPISODE_NUMBER), com linhas verticais a
tracejado a marcar o momento em que cada volta termina.

Voltas detetadas com precisão através do ângulo acumulado em torno do
centro da pista (TRACK_CENTER) — ver plot_track.py para mais detalhe.

Basta editar as variáveis em CONFIG e correr o script:
    python plot_feature_all_laps.py
"""

import json
import math
import os

import matplotlib.pyplot as plt

# =========================================================
# CONFIG (edita só isto)
# =========================================================

# Caminho para o JSON gerado pelo TrackingEvalCallback
METRICS_JSON_PATH = "logs\\eval_metrics\\drift_12h_norm_rewards_timestep_32_facilitate_drifting_130_170_progress_reward_01.json"

# Qual evalback e episódio queres ver
EVALCALLBACK_NUMBER = 7
EPISODE_NUMBER = 3

# Métrica a mostrar
FEATURE_TO_PLOT = "drift_angle"

# Eixo X: "step" ou "sim_time"
X_AXIS = "step"

# Centro da pista, usado para detetar as voltas (não desenhado neste gráfico)
TRACK_CENTER = (150, 0)

# Mostrar linha horizontal a tracejado com a média da feature (episódio inteiro)
SHOW_MEAN_LINE = True

# Mostrar uma banda sombreada à volta da média com +/- 1 desvio padrão
SHOW_STD_BAND = False

# Pasta e nome do ficheiro de saída
OUTPUT_DIR = "plots"
OUTPUT_FILENAME = f"{FEATURE_TO_PLOT}_all_laps_evalback_{EVALCALLBACK_NUMBER}_episode_{EPISODE_NUMBER}_without_sd.png"


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


def plot_feature_with_lap_boundaries(
    ax, ep_rows: list[dict], laps: list[list[dict]], feature: str, x_axis: str,
    show_mean_line: bool, show_std_band: bool,
) -> None:
    x = [r[x_axis] for r in ep_rows]
    y = [r[feature] for r in ep_rows]
    ax.plot(x, y, color="tab:blue", linewidth=1.2, label=feature, zorder=3)

    ax.set_ylim(-30, 100) # apenas para dist2centerline

    if show_mean_line or show_std_band:
        mean_y = sum(y) / len(y)
        std_y = (sum((v - mean_y) ** 2 for v in y) / len(y)) ** 0.5

        if show_std_band:
            ax.axhspan(
                mean_y - std_y, mean_y + std_y,
                color="tab:blue", alpha=0.12, zorder=1,
                label=f"±1 desvio padrão ({std_y:.2f})",
            )

        if show_mean_line:
            ax.axhline(
                mean_y, color="tab:blue", linestyle="--", linewidth=1.3, alpha=0.8, zorder=2,
                label=f"média ({mean_y:.2f})",
            )

    for i, lap in enumerate(laps):
        boundary_x = lap[-1][x_axis]
        ax.axvline(boundary_x, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(
            boundary_x, ax.get_ylim()[1],
            f"lap {i + 1}", rotation=90, va="top", ha="right", fontsize=7, alpha=0.7,
        )

    #ax.set_title(f"{feature} ao longo do episódio (evalback {EVALCALLBACK_NUMBER}, episódio {EPISODE_NUMBER})")
    ax.set_xlabel("Step" if x_axis == "step" else "Sim time (s)")
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3)
    if show_mean_line or show_std_band:
        ax.legend(loc="best", fontsize=8)


def main() -> None:
    rows = load_metrics(METRICS_JSON_PATH)
    ep_rows = get_episode_rows(rows, EVALCALLBACK_NUMBER, EPISODE_NUMBER)

    if not ep_rows:
        print(f"Não há dados para evalback {EVALCALLBACK_NUMBER}, episódio {EPISODE_NUMBER}.")
        return

    laps = split_into_laps(ep_rows, TRACK_CENTER)
    print(f"Voltas detetadas: {len(laps)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_feature_with_lap_boundaries(
        ax, ep_rows, laps, FEATURE_TO_PLOT, X_AXIS,
        show_mean_line=SHOW_MEAN_LINE, show_std_band=SHOW_STD_BAND,
    )
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()


import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


def _load_experiment_points(index: int):
    """Return (path, DataFrame) for the first XLSX file that starts with the given index."""
    candidates = sorted(
        f for f in os.listdir(".") if f.startswith(f"{index}") and f.endswith(".xlsx")
    )
    if not candidates:
        return None, None

    path = candidates[0]
    df = pd.read_excel(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # print(df)
    df[['x(m)', 'y(m)']] = df[['x(m)', 'y(m)']] * 26.3/35
    
    return path, df


def _plot_points(ax, df):
    """Scatter measurement points on the given axis, grouped by left/right column when available."""
    x_col = next((c for c in df.columns if "x(" in c.lower()), None)
    y_col = next((c for c in df.columns if "y(" in c.lower()), None)
    if not x_col or not y_col:
        return

    flag_col = next((c for c in df.columns if "1/0" in c), None)

    if flag_col and flag_col in df:
        categories = {0: ("Left magnet", "tab:blue"), 1: ("Right magnet", "tab:red")}
        for value, (label, color) in categories.items():
            subset = df[df[flag_col] == value]
            subset = subset[[x_col, y_col]].dropna()
            if subset.empty:
                continue
            ax.scatter(
                subset[x_col],
                subset[y_col],
                s=25,
                c=color,
                edgecolors="black",
                linewidths=0.4,
                label=label,
                alpha=0.9,
            )
    else:
        points = df[[x_col, y_col]].dropna()
        ax.scatter(
            points[x_col],
            points[y_col],
            s=25,
            c="tab:blue",
            edgecolors="black",
            linewidths=0.4,
            label="Measurements",
            alpha=0.9,
        )


# for j in ["Length", "Height", "Damp", "MagneticMoment", "Vx", "Vy", "Mass", "Separation", "Experiment"]:
for j in ["Height"]:
    if j in ["MagneticMoment", "Vx", "Vy", "Mass", "Separation"]:
        end = 6
    elif j in ["Damp", "Length"]:
        end = 7
    elif j in ["Height"]:
        end = 11
    elif j in ["Experiment"]:
        end = 11
    for i in range(3, 5):
        file = f"fractal_basin_{j}_{i}.txt"
        output_image = f"{file}.png"
        field = []  # reset accumulator for each file

        with open(file, "r") as f:
            text = f.read().split("\n")
            Sx, Sy = [float(i) for i in text[0].split(" ")]

            for line in text[1:-1]:
                field.append([int(i) for i in list(line)])

        fig, ax = plt.subplots()
        # colors = [(158, 205, 99), (255, 255, 255), (67, 147, 118)]
        colors = [(255,255,255)]
        colors = [tuple([i / 255 for i in a]) for a in colors]

        cmap = ListedColormap(colors)

        extent = [-Sx, Sx, -Sy, Sy]  # 物理坐标范围

        ax.imshow(field, cmap=cmap, extent=extent, origin="lower")

        if j == "Experiment":
            xlsx_path, df = _load_experiment_points(i)
            if df is not None:
                _plot_points(ax, df)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # ax.legend(loc="upper right", title=os.path.basename(xlsx_path))
                    ax.legend(loc="upper right", title="Experiment Points")

        ax.set_box_aspect(1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        fig.tight_layout()
        fig.savefig(output_image, dpi=300)
        # plt.show()
        plt.close(fig)  # clear figure after export
        

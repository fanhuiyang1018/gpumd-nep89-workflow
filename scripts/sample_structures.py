import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; no display is required
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from pylab import *
from ase.io import read, write

# Example: python sample_structures.py select pca 50
# xyz_file and des_file must have the same order; use "0" when a descriptor file is unavailable
nep_name = os.environ.get("NEP89_PATH", "../../models/nep89/nep.txt")
all_points_plot = 'combine'  # combine plots all points together; separate plots each set independently
sel_des_method = 'before'    # after selects structures after dimensionality reduction; before selects structures before dimensionality reduction
xyz_file = ["dump.xyz"]
des_file = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]

# =========================
# Unified plotting colors and fonts
# =========================
COLORS = {
    "blue": "#609CC8",
    "deep_blue": "#3E79A8",
    "orange": "#F58518",
    "green": "#009E73",
    "red": "#C44E52",
    "purple": "#8E79B9",  # retained for compatibility; not used for selected points
    "gray": "#BDBDBD",
    "light_gray": "#D9D9D9",
    "dark_gray": "#4D4D4D",
}

# Low-saturation blue gradient for energy color bars
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "subjournal_blues",
    ["#F7FBFF", "#D6E5F3", "#8FB6D9", COLORS["blue"], COLORS["deep_blue"]]
)

all_color = [
    COLORS["blue"], "#E69F00", "#009E73", "#C44E52", "#8E79B9",
    "#7F7F7F", "#B279A2", "#59A14F", "#9C755F", "#76B7B2",
    "#A0CBE8", "#FFBE7D", "#8CD17D", "#FF9D9A", "#B699D2",
    "#BAB0AC", "#D4A6C8", "#CFCFCF", "#B9CF6A", "#86BCB6"
]

strucs, des, all_strucs, append_des, extend_des, des_names, atom_counts = {}, {}, [], [], [], [], [0]

# Global style: Arial with publication-friendly vector font embedding
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})

for i, (xyz, descriptor) in enumerate(zip(xyz_file, des_file)):
    if os.path.exists(xyz):
        strucs[i] = read(xyz, ":")
        all_strucs.extend(strucs[i])
    else:
        print(f"File {xyz} does not exist.")
        sys.exit(1)

    if os.path.exists(descriptor):
        des = np.loadtxt(descriptor)
    elif descriptor == "0":
        des_name = os.path.splitext(os.path.basename(xyz))[0]
        des_names.append(des_name)
        if os.path.exists(f'descriptor-{des_name}.out'):
            des = np.loadtxt(f'descriptor-{des_name}.out')
        else:
            from NepTrain.core.nep import *
            nep = Nep3Calculator(nep_name)
            descriptor = [nep.get_descriptors(j).mean(0) for j in strucs[i]]
            with open(f'descriptor-{des_name}.out', "w") as f:
                for descriptor in descriptor:
                    descriptor_str = " ".join(map(str, descriptor))
                    f.write(f"{descriptor_str}\n")
            des = np.loadtxt(f'descriptor-{des_name}.out')
    append_des.append(des)
    extend_des.extend(des)
    atom_counts.append(atom_counts[-1] + len(des))


def main():
    if len(sys.argv) < 3:
        print("method: pca umap tsne kpca svd isomap se ica fa mds grp srp")
        print("Usage: python sample_structures.py all <method> # Observe the position of all points")
        print("Usage: python sample_structures.py select <method> min_distance_1/max_select_1 min_distance_2/max_select_2 ......")
        print("Usage: python sample_structures.py pick <method> x1_start x1_end y1_start y1_end x2_start x2_end y2_start y2_end ......")
        sys.exit(1)


def FarthestPointSample(new_data, now_data=[], min_distance=0.1, min_select=1, max_select=None, metric='euclidean'):
    from scipy.spatial.distance import cdist
    max_select = max_select or len(new_data)
    to_add = []
    if len(new_data) == 0:
        return to_add
    if len(now_data) == 0:
        to_add.append(0)
        now_data.append(new_data[0])
    distances = np.min(cdist(new_data, now_data, metric=metric), axis=1)
    while np.max(distances) > min_distance or len(to_add) < min_select:
        i = np.argmax(distances)
        to_add.append(i)
        if len(to_add) >= max_select:
            break
        distances = np.minimum(distances, cdist([new_data[i]], new_data, metric=metric)[0])
    return to_add


def pick_points(proj, range_x, range_y):
    pick_strucs = []
    for i, point in enumerate(proj):
        if range_x[0] <= point[0] <= range_x[1] and range_y[0] <= point[1] <= range_y[1]:
            pick_strucs.append(i)
    return pick_strucs


def get_energies_per_atom(strucs):
    energies_per_atom = []
    for atoms in strucs:
        if not atoms.has('energy'):
            energy_per_atom = atoms.info.get('Energy') / len(atoms)
        else:
            energy_per_atom = atoms.get_potential_energy() / len(atoms)
        energies_per_atom.append(energy_per_atom)
    return energies_per_atom


def set_tick_params():
    ax = plt.gca()
    ax.tick_params(axis='x', which='both', direction='in', top=False, bottom=True,
                   labelsize=20, length=5, width=1.0)
    ax.tick_params(axis='y', which='both', direction='in', left=True, right=False,
                   labelsize=20, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color(COLORS["dark_gray"])


def save_png_pdf(filename_base):
    """Save both PNG and PDF for publication layout."""
    tight_layout()
    savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    savefig(f"{filename_base}.pdf", bbox_inches='tight')
    close()


if __name__ == "__main__":
    main()

if sys.argv[2] == "umap":
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_jobs=1)
elif sys.argv[2] == "pca":
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
elif sys.argv[2] == "tsne":
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2)
elif sys.argv[2] == "isomap":
    from sklearn.manifold import Isomap
    reducer = Isomap(n_components=2)
elif sys.argv[2] == "svd":
    from sklearn.decomposition import TruncatedSVD
    reducer = TruncatedSVD(n_components=2)
elif sys.argv[2] == "kpca":
    from sklearn.decomposition import KernelPCA
    reducer = KernelPCA(n_components=2)
elif sys.argv[2] == "mds":
    from sklearn.manifold import MDS
    reducer = MDS(n_components=2, random_state=42)
elif sys.argv[2] == "se":
    from sklearn.manifold import SpectralEmbedding
    reducer = SpectralEmbedding(n_components=2, random_state=42)
elif sys.argv[2] == "fa":
    from sklearn.decomposition import FactorAnalysis
    reducer = FactorAnalysis(n_components=2, random_state=42)
elif sys.argv[2] == "ica":
    from sklearn.decomposition import FastICA
    reducer = FastICA(n_components=2, max_iter=1000, random_state=42)
elif sys.argv[2] == "grp":
    from sklearn.random_projection import GaussianRandomProjection
    reducer = GaussianRandomProjection(n_components=2, random_state=42)
elif sys.argv[2] == "srp":
    from sklearn.random_projection import SparseRandomProjection
    reducer = SparseRandomProjection(n_components=2, random_state=42)
else:
    raise ValueError(f"Unknown reduction method: {sys.argv[2]}")

all_proj = reducer.fit_transform(extend_des)

if sys.argv[1] == "all":
    for i in range(len(append_des)):
        if len(append_des[i]) == len(strucs[i]):
            if all_points_plot == 'combine':
                energy_train = get_energies_per_atom(all_strucs)
                sc = scatter(
                    all_proj[:, 0], all_proj[:, 1],
                    c=energy_train,
                    cmap=ENERGY_CMAP,
                    s=22,
                    alpha=0.85,
                    edgecolors='none'
                )
            else:
                proj = all_proj[atom_counts[i]:atom_counts[i + 1], :]
                energy_train = get_energies_per_atom(strucs[i])
                sc = scatter(
                    proj[:, 0], proj[:, 1],
                    c=energy_train,
                    cmap=ENERGY_CMAP,
                    edgecolor=all_color[i],
                    linewidth=0.6,
                    alpha=0.80,
                    label=des_names[i],
                    s=22
                )
        else:
            element_descriptors, all_des_symbols = {element: [] for element in elements}, []
            for des_struc in all_strucs:
                des_symbol = des_struc.get_chemical_symbols()
                all_des_symbols.extend(des_symbol)
            for symbol, des in zip(all_des_symbols, extend_des):
                element_descriptors[symbol].append(des)
            element_des = {element: des for element, des in element_descriptors.items() if des}
            for idx, element in enumerate(element_des.keys()):
                scatter(
                    [i[0] for i in element_des[element]],
                    [i[1] for i in element_des[element]],
                    color=all_color[idx % len(all_color)],
                    alpha=0.70,
                    label=element,
                    s=22,
                    edgecolors='none'
                )
            legend(frameon=False, fontsize=20, loc='upper right')
            set_tick_params()
            xlabel('PC1')
            ylabel('PC2')
            save_png_pdf(f"each-elements-{xyz_file[i]}-{sys.argv[2]}")

    if len(append_des[0]) == len(strucs[0]):
        cbar = colorbar(sc, cax=gca().inset_axes([0.70, 0.95, 0.27, 0.025]), orientation='horizontal')
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar.ax.tick_params(axis='x', labelsize=20, direction='in', length=3, width=0.8)
        cbar.outline.set_linewidth(0.8)
        cbar.set_label('E/atom (eV)', fontsize=120, labelpad=2)
        legend(ncol=5, frameon=False, fontsize=20, loc='upper left')
        set_tick_params()
        xlabel('PC1')
        ylabel('PC2')
        save_png_pdf(f"all-points-{sys.argv[2]}")

elif sys.argv[1] == "select":
    min_distances, counts = [], []
    for arg in sys.argv[3:]:
        if float(arg) < 1:
            min_distances.append(float(arg))
        else:
            counts.append(int(arg))
    select_values = min_distances + counts

    select_des = all_proj if sel_des_method == 'after' else extend_des
    for select_value in select_values:
        if float(select_value) < 1:
            select_ids = sorted(set(FarthestPointSample(select_des, min_distance=select_value)))
        else:
            select_ids = sorted(set(FarthestPointSample(select_des, min_distance=0, max_select=select_value)))

        write(f"select_{select_value}.xyz", [all_strucs[j] for j in select_ids], format='extxyz')
        abandon_ids = [i for i in range(len(all_strucs)) if i not in select_ids]
        write(f"abandon_{select_value}.xyz", [all_strucs[j] for j in abandon_ids], format='extxyz')

        # Use light gray for background points and low-saturation blue for selected points
        scatter(
            all_proj[:, 0], all_proj[:, 1],
            alpha=0.38,
            c=COLORS["gray"],
            label="All",
            s=18,
            edgecolors='none',
            zorder=1
        )
        selected_proj = array([all_proj[i] for i in select_ids])
        label_text = f"min_distance={select_value}" if float(select_value) < 1 else f"select_counts={select_value}"
        scatter(
            selected_proj[:, 0], selected_proj[:, 1],
            alpha=0.90,
            c=COLORS["blue"],
            label=label_text,
            s=32,
            edgecolors='none',
            linewidth=0.45,
            zorder=2
        )

        set_tick_params()
        xlabel('PC1')
        ylabel('PC2')
        legend(fontsize=16, frameon=False, loc='best', handletextpad=0.4)
        save_png_pdf(f"select-{select_value}-{sys.argv[2]}")

elif sys.argv[1] == "pick":
    pick_ids = set()
    for i in range(3, len(sys.argv), 4):
        x_start, x_end, y_start, y_end = map(float, sys.argv[i:i+4])
        range_x = (x_start, x_end)
        range_y = (y_start, y_end)
        current_ids = pick_points(all_proj, range_x, range_y)
        pick_ids.update(current_ids)
    pick_ids = list(pick_ids)
    write("pick.xyz", [all_strucs[j] for j in pick_ids], format='extxyz')
    retain_ids = [i for i in range(len(all_strucs)) if i not in pick_ids]
    write("retain.xyz", [all_strucs[j] for j in retain_ids], format='extxyz')

    scatter(
        all_proj[retain_ids, 0], all_proj[retain_ids, 1],
        alpha=0.38,
        color=COLORS["gray"],
        label="Retain",
        s=18,
        edgecolors='none',
        zorder=1
    )
    scatter(
        all_proj[pick_ids, 0], all_proj[pick_ids, 1],
        alpha=0.90,
        color=COLORS["orange"],
        label="Pick",
        s=32,
        edgecolors=COLORS["dark_gray"],
        linewidth=0.45,
        zorder=2
    )
    set_tick_params()
    xlabel('PC1')
    ylabel('PC2')
    legend(fontsize=120, frameon=False, loc='best', handletextpad=0.4)
    save_png_pdf(f"retain-pick-{sys.argv[2]}")

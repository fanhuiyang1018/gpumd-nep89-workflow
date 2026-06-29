import os, glob, math
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ====================== Publication-oriented style: Arial and low-saturation colors ======================
# Main blue is kept consistent with the other plotting scripts.
COLORS = {
    'blue': '#609CC8',
    'deep_blue': '#3E79A8',
    'light_blue': '#BDD3E6',
    'orange': '#F3A65A',
    'green': '#6BAF75',
    'red': '#C86A6A',
    'purple': '#8E79B9',
    'gray': '#C7C7C7',
    'dark_gray': '#4D4D4D',
}

plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})

COLOR_ENERGY = COLORS['blue']
COLOR_FORCE  = COLORS['orange']
COLOR_VIRIAL = COLORS['green']
COLOR_STRESS = COLORS['red']

ENERGY_CMAP = LinearSegmentedColormap.from_list(
    'subjournal_blues',
    ['#F7FBFF', '#DCEAF5', '#9FC0DD', COLORS['blue'], COLORS['deep_blue']]
)

three_six_component = 0
use_range = 0
element_force = 0
charge_sign = 1
plot_range = {'energy': (-9, -8), 'force': (-20, 20), 'virial': (-10, 10), 
       'stress': (-10, 10), 'dipole': (-10, 10), 'polarizability': (-10, 10)}

train_colors = [COLOR_ENERGY, COLOR_FORCE, COLOR_VIRIAL, COLORS['purple'], COLORS['red'], COLORS['gray']]
test_colors = [COLORS['deep_blue'], '#D88C3A', '#3F8F5A', '#6E5F9E', '#A94F4F', COLORS['dark_gray']]

def get_diag_color(data):
    if data == 'energy':
        return COLOR_ENERGY, COLORS['light_blue']
    elif data == 'force':
        return COLOR_FORCE, '#F8C99A'
    elif data == 'virial':
        return COLOR_VIRIAL, '#A8D5AF'
    elif data == 'stress':
        return COLOR_STRESS, '#E2A3A3'
    else:
        return COLORS['blue'], COLORS['orange']

def generate_colors(data):
    if three_six_component == 0 or data == 'energy':
        return get_diag_color(data)
    else:
        if data in ['force', 'dipole']:
            return train_colors[:3], test_colors[:3]
        else:
            return train_colors, test_colors

dipole_files, polar_files = glob.glob('dipole*'), glob.glob('polarizability*')
model_type = 'dipole' if dipole_files else 'polarizability' if polar_files else None
in_file = 'gnep.in' if os.path.exists('gnep.in') else 'nep.in'
lambda_v = 0.1
batch = 1 if os.path.exists('gnep.in') else 1000
with open(in_file) as file:
    for line in file:
        line = line.strip()
        if 'lambda_v' in line and not line.startswith('#'):
            lambda_v = line.split()[1]
        if 'batch' in line and not line.startswith('#'):
            batch = int(line.split()[1])

train_novirial_indices, test_novirial_indices = [], []
train_length, test_length = 0, 0

def get_novirial_indices(path, marker='-1e+06'):
    idx = []
    with open(path) as f:
        for i, line in enumerate(f):
            *_, last = line.split()
            if last == marker:
                idx.append(i)
        total = i + 1 if 'i' in locals() else 0
    return idx, total

if lambda_v != '0':
    if os.path.exists('virial_train.out'):
        train_novirial_indices, train_length = get_novirial_indices('virial_train.out')
        if len(train_novirial_indices) > 0:
            train_indices = [i for i in range(train_length) if i not in train_novirial_indices]
    if os.path.exists('virial_test.out'):
        test_novirial_indices, test_length = get_novirial_indices('virial_test.out')
        if len(test_novirial_indices) > 0:
            test_indices = [i for i in range(test_length) if i not in test_novirial_indices]

def set_tick_params():
    tick_params(axis='x', which='both', direction='in', top=True, bottom=True, labelsize=15, width=1.0)
    tick_params(axis='y', which='both', direction='in', left=True, right=True, labelsize=15, width=1.0)
    tick_params(axis='x', which='major', length=5)
    tick_params(axis='y', which='major', length=5)
    ax = gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color(COLORS['dark_gray'])

def get_counts2two(out_file):
    file_nums = int(out_file.shape[1]//2)
    new_nep, new_dft = out_file[:, :file_nums].flatten(), out_file[:, file_nums:].flatten()
    return np.column_stack((new_nep, new_dft))

def calc_r2_rmse(out_file):
    file_columns = int(out_file.shape[1]//2)
    numerator = np.sum((out_file[:, :file_columns] - out_file[:, file_columns:]) ** 2)
    denominator = np.sum((out_file[:, :file_columns] - np.mean(out_file[:, :file_columns])) ** 2)
    r2_data = 1.0 if denominator == 0 else 1 - (numerator / denominator)
    rmse_origin = np.sqrt(np.mean((out_file[:, :file_columns]-out_file[:, file_columns:])**2))
    rmse_data = rmse_origin * 1000 if rmse_origin < 1 else rmse_origin
    return rmse_origin, rmse_data, r2_data

def plot_value(values, color):
    columns = int(values.shape[1]//2)
    if three_six_component == 0 or data == 'energy':
        plot(values[:, 1], values[:, 0], '.', color=color, markersize=3, alpha=0.65)
    else:
        for i in range(columns):
            plot(values[:, i+columns], values[:, i], '.', color=color[i % len(color)], markersize=3, alpha=0.65)

units = {'force': 'eV/Å', 'stress': 'GPa', 'energy': 'eV/atom','virial': 'eV/atom', 'dipole': 'a.u./atom', 'polarizability': 'a.u./atom'}
munits = {'force': 'meV/Å', 'stress': 'MPa', 'energy': 'meV/atom','virial': 'meV/atom', 'dipole': 'ma.u./atom', 'polarizability': 'ma.u./atom'}

def get_unit(data, rmse_origin):
    return munits.get(data, 'unknown unit') if rmse_origin < 1 else units.get(data, 'unknown unit')

def generate_legs(data, types, prefixes):
    comps = {3: ['x', 'y', 'z'], 6: ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']}
    if os.path.exists(f"{data}_test.out"):
        return {typ: [f"{prefix}_{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types for prefix in prefixes}
    else:
        return {typ: [f"{comp}" for comp in comps[3 if typ in ['force', 'dipole'] else 6]] for typ in types}

properties, prefixes = ['force', 'stress', 'virial', 'dipole', 'polarizability'], ['train', 'test']

def get_range(data, data_file):
    if data == 'energy':
        return np.floor(data_file.min() * 10) / 10, np.ceil(data_file.max() * 10) / 10
    else:
        return np.floor(data_file.min()), np.ceil(data_file.max())
        
def process_data(data, data_type):
    if three_six_component == 0:
        return globals().get(f"{data}_{data_type}") if data == 'energy' else get_counts2two(globals().get(f"{data}_{data_type}"))
    else:
        return globals().get(f"{data}_{data_type}")

def get_element_property(file, atoms_property):
    potential_file = 'gnep.txt' if os.path.exists('gnep.txt') else 'nep.txt'
    with open(potential_file, 'r') as txtfile:
        first_line = txtfile.readline().strip()
        elements = first_line.split()[2:]
    from ase.io import read
    atoms = read(f'{file}.xyz', index=':')
    atom_symbols = []
    for atom in atoms:
        atom_symbol = atom.get_chemical_symbols()
        atom_symbols.extend(atom_symbol)
    element_lists = {element: [] for element in elements}
    for symbol, atom_property in zip(atom_symbols, atoms_property):
        element_lists[symbol].append(atom_property)
    non_empty_elements_lists = {element: atom_property for element, atom_property in element_lists.items() if atom_property}
    return non_empty_elements_lists

# ====================== Use a smaller legend font only for the loss plot ======================
def plot_loss():
    loss = np.loadtxt('loss.out')
    label_Lgnep, label_Lnep = [r'$L_{\text{total}}$'], [r'$L_1$', r'$L_2$']
    label_ef, label_ef_train, label_ef_test = ['Energy', 'Force'], ['E-train', 'F-train'], ['E-test', 'F-test']
    gca().set_prop_cycle(color=[COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'], COLORS['purple'], COLORS['dark_gray']])
    
    if os.path.exists('gnep.in'):
        loss_L, learning_rate = loss[:, 1], loss[:, 8]
        loss_train, loss_train_v = loss[:, 2:4], loss[:, 2:5]
        loss_test, loss_test_v= loss[:, 5:7], loss[:, 5:8]
    else:
        loss_L = loss[:, 2:4]
        loss_train, loss_train_v = loss[:, 4:6], loss[:, 4:7]
        loss_test, loss_test_v = loss[:, 7:9], loss[:, 7:10]
    
    if loss.shape[1] < 7:
        loglog(loss[:, 2:5], linewidth=1.5)
        if os.path.exists('test.xyz'):
            loglog(loss[:, 5], linewidth=1.5)
            legend(label_Lnep + [f'{model_type}-train', f'{model_type}-test'], ncol=4, frameon=False, fontsize=9.5, loc='upper right')
        else:
            legend(label_Lnep + [f'{model_type}'], ncol=3, frameon=False, fontsize=9.5, loc='upper right')
    else: 
        if lambda_v == '0' or (train_novirial_indices is not None and len(train_novirial_indices) == train_length):
            loglog(loss_L, linewidth=1.5)
            loglog(loss_train, linewidth=1.5)
            if os.path.exists('test.xyz'):
                loglog(loss_test, linewidth=1.5)
                loss_label = label_Lgnep + label_ef_train + label_ef_test if os.path.exists('gnep.in') else label_Lnep + label_ef_train + label_ef_test
                legend(loss_label, ncol=3, frameon=False, fontsize=9.5, loc='lower left')
            else:
                loss_label = label_Lgnep + label_ef if os.path.exists('gnep.in') else label_Lnep + label_ef
                legend(loss_label, ncol=4, frameon=False, fontsize=9.5, loc='upper right')
        else:
            loglog(loss_L, linewidth=1.5)
            loglog(loss_train_v, linewidth=1.5)
            if os.path.exists('test.xyz'):
                loglog(loss_test_v, linewidth=1.5)
                loss_label_v = label_Lgnep + label_ef_train + ['V-train'] + label_ef_test + ['V-test'] if os.path.exists('gnep.in') else label_Lnep + label_ef_train + ['V-train'] + label_ef_test + ['V-test']
                legend(loss_label_v, ncol=2, frameon=False, fontsize=9.5, loc='lower left')
            else:
                loss_label_v = label_Lgnep + label_ef + ['Virial'] if os.path.exists('gnep.in') else label_Lnep + label_ef + ['Virial']
                legend(loss_label_v, ncol=5, frameon=False, fontsize=9.5, loc='upper right')

    set_tick_params()
    if os.path.exists('gnep.in'):
        xlabel('Epoch', fontsize=15)
    else:
        xlabel('Generation/100', fontsize=15)
    ylabel('Loss', fontsize=15)
    tight_layout()

def plot_learning_rate():
    plot(range(1, len(loss) + 1), loss[:, 8], linewidth=1.5, color=COLORS['blue'])
    set_tick_params()
    xlabel('Epoch', fontsize=15)
    ylabel('Learning Rate', fontsize=15)
    tight_layout()

def plot_diagonal(data):
    color_train, color_test = generate_colors(data)
    label_unit = units.get(data, 'unknown unit')
    train_legs, test_legs = generate_legs(data, properties, prefixes[0::2]), generate_legs(data, properties, prefixes[1::2])
    train_leg, test_leg = train_legs.get(data, 'train'), test_legs.get(data, 'test')

    if not os.path.exists(f'{data}_train.out'):
        print(f'{data}_train.out not found')
        return
    globals()[f'{data}_train'] = np.loadtxt(f'{data}_train.out')

    if data in ['virial', 'stress'] and len(train_novirial_indices) > 0:
        globals()[f'{data}_train'] = globals()[f'{data}_train'][train_indices]

    data_train = process_data(data, 'train')
    train_min, train_max = get_range(data, data_train)
    plot_value(data_train, color_train)
    rmse_train_o, rmse_train, r2_train = calc_r2_rmse(data_train)

    data_test = None
    if os.path.exists(f"{data}_test.out"):
        globals()[f'{data}_test'] = np.loadtxt(f'{data}_test.out')
        if data in ['virial', 'stress'] and len(test_novirial_indices) > 0:
            globals()[f'{data}_test'] = globals()[f'{data}_test'][test_indices]
        data_test = process_data(data, 'test')
        test_min, test_max = get_range(data, data_test)
        plot_value(data_test, color_test)
        rmse_test_o, rmse_test, r2_test = calc_r2_rmse(data_test)
        unit = get_unit(data, rmse_test_o)
    else:
        unit = get_unit(data, rmse_train_o)

    if use_range == 0:
        range_min = train_min if (data_test is None or train_min < test_min) else test_min
        range_max = train_max if (data_test is None or train_max > test_max) else test_max
    else:
        range_min, range_max = plot_range.get(data, (-10,10))

    plot(linspace(range_min, range_max), linspace(range_min, range_max), '--', color=COLORS['dark_gray'], lw=1.2, zorder=0)
    set_tick_params()
    xlabel(f"DFT {data} ({label_unit})", fontsize=15)
    ylabel(f"NEP {data} ({label_unit})", fontsize=15)

    if data_test is not None:
        legend([f'train RMSE={rmse_train:.2f} {unit} R²={r2_train:.3f}',
                f'test RMSE={rmse_test:.2f} {unit} R²={r2_test:.3f}'],
               frameon=False, fontsize=13, loc='upper left')
    else:
        legend([f'train RMSE={rmse_train:.2f} {unit} R²={r2_train:.3f}'],
               frameon=False, fontsize=13, loc='upper left')
    tight_layout()

def plot_charge():
    if not os.path.exists('charge_train.out'): return
    charge_train = np.loadtxt('charge_train.out')
    try:
        e_len = len(globals()['energy_train'])
    except:
        e_len = -1
    if batch < e_len:
        print('Please use fullbatch charge_*.out')
        return
    import seaborn as sns
    sns.set_palette([COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'], COLORS['purple'], COLORS['dark_gray']])
    figure(figsize=(5.5,5))
    element_charges_train = get_element_property('train', -charge_train) if charge_sign == -1 else get_element_property('train', charge_train)
    if os.path.exists('charge_test.out'):
        charge_test = np.loadtxt('charge_test.out')
        element_charges_test = get_element_property('test', -charge_test) if charge_sign == -1 else get_element_property('test', charge_test)
        for e, (et, tt) in enumerate(zip(element_charges_train, element_charges_test)):
            sns.histplot(element_charges_train[et], bins=300, alpha=0.6, label=f'{et}-train', kde=True, lw=1.5)
            sns.histplot(element_charges_test[tt], bins=300, alpha=0.6, label=f'{tt}-test', kde=True, lw=1.5)
        legend(ncol=2, frameon=False, fontsize=15)
    else:
        for e, elem in enumerate(element_charges_train):
            sns.histplot(element_charges_train[elem], bins=300, alpha=0.6, label=elem, kde=True, lw=1.5)
        legend(frameon=False, fontsize=15)
    xlabel('Charge', fontsize=15)
    ylabel('Frequency', fontsize=15)
    set_tick_params()
    tight_layout()
    savefig('nep-charge.png', dpi=300, bbox_inches='tight')
    savefig('nep-charge.pdf', bbox_inches='tight')

def plot_descriptor():
    if not os.path.exists('descriptor.out'): return
    descriptor = np.loadtxt('descriptor.out')
    from sklearn.decomposition import PCA
    proj = PCA(n_components=2).fit_transform(descriptor)
    figure(figsize=(5.5,5))
    try:
        if len(descriptor) == len(globals()['energy_train']):
            sc = scatter(proj[:,0], proj[:,1], c=globals()['energy_train'][:,1], cmap=ENERGY_CMAP, edgecolor='none', alpha=0.85, s=15)
            cbar = colorbar(sc, cax=gca().inset_axes([0.73,0.95,0.23,0.03]), orientation='horizontal')
            cbar.set_label('E/atom (eV)', fontsize=13)
            cbar.outline.set_linewidth(0.8)
            cbar.ax.tick_params(labelsize=12, direction='in', width=0.8, length=3)
        elif len(descriptor) == len(globals()['force_train']):
            ed = get_element_property('train', descriptor)
            for i, elem in enumerate(ed):
                scatter([p[0] for p in ed[elem]], [p[1] for p in ed[elem]], edgecolor='none', alpha=0.75, s=15, label=elem)
            legend(frameon=False, fontsize=15)
    except:
        pass
    xlabel('PC1', fontsize=15)
    ylabel('PC2', fontsize=15)
    set_tick_params()
    tight_layout()
    savefig('nep-descriptor.png', dpi=300, bbox_inches='tight')
    savefig('nep-descriptor.pdf', bbox_inches='tight')

def plot_element_force():
    if not element_force: return
    try:
        force_train = globals()['force_train']
        fet = get_element_property('train', force_train)
    except:
        return
    for elem, fe in fet.items():
        figure(figsize=(5.5,5))
        tef = get_counts2two(np.array(fe))
        plot(tef[:,1], tef[:,0], '.', color=COLOR_FORCE, label=f'{elem}-train', ms=3, alpha=0.65)
        if os.path.exists('force_test.out'):
            force_test = globals()['force_test']
            fetest = get_element_property('test', force_test)
            ttef = get_counts2two(np.array(fetest[elem]))
            plot(ttef[:,1], ttef[:,0], '.', color=COLOR_ENERGY, label=f'{elem}-test', ms=3, alpha=0.65)
        rmin, rmax = get_range('force', tef)
        plot(linspace(rmin,rmax), linspace(rmin,rmax), '--', color=COLORS['dark_gray'], lw=1.2)
        set_tick_params()
        xlabel('DFT force (eV/Å)', fontsize=15)
        ylabel('NEP force (eV/Å)', fontsize=15)
        legend(frameon=False, fontsize=15)
        title(f'Force of {elem}', fontsize=15)
        tight_layout()
        savefig(f'{elem}-force.png', dpi=300, bbox_inches='tight')
        savefig(f'{elem}-force.pdf', bbox_inches='tight')

base_diag_types = ['energy', 'force', 'virial', 'stress']

def plot_diagonals(diag_types, hang, lie, start):
    for i, dt in enumerate(diag_types):
        subplot(hang, lie, i+start)
        plot_diagonal(dt)

def plot_base_picture():
    if model_type:
        figure(figsize=(5.5,5))
        plot_diagonal(model_type)
        savefig(f'nep-{model_type}-diagonal.png', dpi=300, bbox_inches='tight')
        savefig(f'nep-{model_type}-diagonal.pdf', bbox_inches='tight')
    else:
        if lambda_v == '0' or len(train_novirial_indices) == train_length:
            figure(figsize=(11,5))
            plot_diagonals(base_diag_types[:2], 1,2,1)
            savefig('nep-ef-diagonals.png', dpi=300, bbox_inches='tight')
            savefig('nep-ef-diagonals.pdf', bbox_inches='tight')
        elif not os.path.exists('stress_train.out'):
            figure(figsize=(16.5,5))
            plot_diagonals(base_diag_types[:3],1,3,1)
            savefig('nep-efv-diagonals.png', dpi  =300, bbox_inches='tight')
            savefig('nep-efv-diagonals.pdf', bbox_inches='tight')
        else:
            figure(figsize=(12,10))
            plot_diagonals(base_diag_types, 2,2,1)
            savefig('nep-efvs-diagonals.png', dpi=300, bbox_inches='tight')
            savefig('nep-efvs-diagonals.pdf', bbox_inches='tight')

# Main program
if __name__ == '__main__':
    if os.path.exists('loss.out'):
        print('NEP Train')
        loss = np.loadtxt('loss.out')
        if os.path.exists('gnep.in'):
            figure(figsize=(11,5))
            subplot(121); plot_loss()
            subplot(122); plot_learning_rate()
            savefig('gnep-loss-lr.png', dpi=300, bbox_inches='tight')
            savefig('gnep-loss-lr.pdf', bbox_inches='tight')
        else:
            figure(figsize=(5.5,5))
            plot_loss()
            savefig('nep-loss.png', dpi=300, bbox_inches='tight')
            savefig('nep-loss.pdf', bbox_inches='tight')
        plot_base_picture()
        plot_charge()
        plot_element_force()
    else:
        print('NEP Prediction')
        plot_base_picture()
        plot_charge()
        plot_descriptor()
        plot_element_force()

plt.close('all')
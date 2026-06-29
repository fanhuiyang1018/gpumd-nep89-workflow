from ase.io import read, write
import os, sys, shutil, subprocess

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 convert_extxyz_to_dft_inputs.py <xyz_file> <number_of_perturbations> vasp/abacus")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Initialize global variables
dpdata_available = False
ntype = 0
upf_orb = ""
number_structures = int(sys.argv[2])
original_cwd = os.getcwd()

# Read XYZ structure metadata
with open(sys.argv[1], "r") as input_file:
    first_line = input_file.readline().strip()
    try:
        structure_lines = int(first_line) + 2  # Lines per XYZ structure (number of atoms + 2)
    except ValueError:
        print("Error: the first line of the XYZ file must be an integer atom count.")
        sys.exit(1)
    total_lines = sum(1 for _ in input_file) + 1  # Add back the first line already read
    structures_count = total_lines // structure_lines
    if structures_count < number_structures:
        print(f"Error: XYZ contains {structures_count} structures, but {number_structures} were requested.")
        sys.exit(1)

# Handle output format argument
if sys.argv[3] == 'abacus':
    try:
        import dpdata
        dpdata_available = True
    except ImportError:
        print("Error: dpdata is required for ABACUS output.")
        sys.exit(1)
else:
    dpdata_available = False

def create_train_folders():
    """Create training folder structure"""
    if not os.path.exists('train_folders'):
        os.makedirs('train_folders')
    for i in range(number_structures):
        folder_name = f'activate-learning-{i+1}'
        folder_path = os.path.join('train_folders', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def split_xyz():
    """Split the XYZ file into subfolders"""
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()
    
    # Calculate the sampling interval to select structures uniformly
    step = max(1, structures_count // number_structures)
    selected_indices = [i * step for i in range(number_structures) if i * step < structures_count]
    
    for idx, i in enumerate(selected_indices):
        start_index = i * structure_lines
        end_index = start_index + structure_lines
        if end_index > len(lines):
            print(f"Warning: structure {i+1} is outside the file range and was skipped.")
            continue
        
        group_lines = lines[start_index:end_index]
        folder_name = f'activate-learning-{idx+1}'
        group_filepath = os.path.join('train_folders', folder_name, f'{folder_name}.xyz')
        
        with open(group_filepath, 'w') as group_file:
            group_file.writelines(group_lines)

def convert_xyz_to_poscar():
    """Convert XYZ files to POSCAR and group repeated elements"""
    train_folder_path = os.path.join(os.getcwd(), 'train_folders')
    folders = os.listdir(train_folder_path)
    
    for folder_name in folders:
        folder_path = os.path.join(train_folder_path, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directory entries
        
        os.chdir(folder_path)
        # Find XYZ files in the folder
        xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]
        if not xyz_files:
            print(f"Warning: folder {folder_name} contains no XYZ file.")
            os.chdir(original_cwd)
            continue
        xyz_file = xyz_files[0]
        
        # Read atoms and sort by element symbol
        try:
            atoms = read(xyz_file, format="extxyz")
            # Sort by element symbol so identical elements are contiguous
            sorted_indices = sorted(range(len(atoms)), key=lambda x: atoms.get_chemical_symbols()[x])
            atoms_sorted = atoms[sorted_indices]
            write("POSCAR", atoms_sorted, format="vasp")
        except Exception as e:
            print(f"Failed to convert {xyz_file} to POSCAR: {e}")
            os.chdir(original_cwd)
            continue
        
        # Handle ABACUS format
        if sys.argv[3] == 'abacus' and dpdata_available:
            try:
                # Read POSCAR to get the number of element types
                d_poscar = dpdata.System('POSCAR', fmt="vasp/poscar")
                global ntype, upf_orb
                ntype = d_poscar.get_ntypes()
                
                # Generate and post-process STRU
                d_poscar.to("abacus/stru", "STRU")
                
                # Read the POSCAR.STRU template if present
                if os.path.exists(os.path.join(original_cwd, "POSCAR.STRU")):
                    with open(os.path.join(original_cwd, "POSCAR.STRU"), 'r') as file:
                        upf_orb = ''.join([next(file) for _ in range(2 * ntype + 3)])
                    
                    # Modify the STRU file
                    with open("STRU", 'r') as file:
                        lines = file.readlines()
                    if len(lines) >= ntype + 1:
                        lines[:ntype+1] = upf_orb.split('\n')
                        with open("STRU", 'w') as file:
                            file.writelines('\n'.join(lines))
            except Exception as e:
                print(f"Failed to process ABACUS format: {e}")
        
        os.chdir(original_cwd)

# Run the main workflow
create_train_folders()
split_xyz()
convert_xyz_to_poscar()

# Copy input files and run downstream commands
for j in range(number_structures):
    folder_name = f'activate-learning-{j+1}' 
    folder_path = os.path.join('train_folders', folder_name)
    if not os.path.exists(folder_path):
        print(f"Warning: folder {folder_path} does not exist and was skipped.")
        continue
    
    try:
        if sys.argv[3] == 'abacus':
            # Copy ABACUS input files
            if os.path.exists("INPUT-scf"):
                shutil.copy("INPUT-scf", os.path.join(folder_path, 'INPUT'))
                os.chdir(folder_path)
                atomkit_command = 'echo "3\n 301\n 0\n 101 STRU\n 0.03" | atomkit'
                subprocess.run(atomkit_command, shell=True, check=True)
        else:
            # Copy VASP input files
            if os.path.exists("INCAR-scf"):
                shutil.copy("INCAR-scf", os.path.join(folder_path, 'INCAR'))
                os.chdir(folder_path)
                vaspkit_command = "vaspkit -task 102 -kpr 0.03" 
                subprocess.run(vaspkit_command, shell=True, check=True)
    except Exception as e:
        print(f"Failed to process folder {folder_name}: {e}")
    finally:
        os.chdir(original_cwd)
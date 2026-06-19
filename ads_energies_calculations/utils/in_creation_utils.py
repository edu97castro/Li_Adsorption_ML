# utils/in_creation_utils.py

import csv
import os

import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pathlib import Path

from .structure_utils import first_surface_layers
from .in_out_management_utils import extract_relaxed_positions_from_out, redefine_atomic_positions_in

def _find_potential(element, csv_file='potenciales.csv'): # Internal use only
    """
    Search for an element in the pseudopotentials database, saved in a CSV file, and return a string with the format '{element} {mass} {potential}'.

    Parameters:
    -----------
        element : str
            The chemical symbol of the element to search.
        csv_file : str, default="potenciales.csv"
            path of the CSV file where search the element.

    Returns:
    --------
    str
        A string with the format '{element} {mass} {potential}', where:
            - element is the chemical symbol
            - mass is the atomic mass
            - potential is the name of the pseudopotential file.
    """
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == element:
                return f"{row[0]} {row[1]} {row[2]}\n"
    print(f"Element {element} not found.")
    return None

def structure_in(structure, path, file_name, outdir, prefix, pseudo_dir, kpoints = [4,4,4], atoms_to_relax = None, nosym=False, electron_maxstep = 200, vdw = None, dftd3_threebody = True):
    """
    Writes the input of a relaxation pw.x quantum expresso simulation from a given structure.

    For more details of the meaning of some of the parameters in the calculations, see QE pw.x doc.

    Parameters:
    -----------
        surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The atomic structure for which the input file will be generated. This structure should be compatible with the slab or bulk generation process.
        path: str
            folder's dir where the input file is going to be saved (It must not end with "/").
        file_name: str
            The name of the .in file.
        outdir: str
            Directory´s path to where save output files (It must not end with "/").
        prefix: str
            Prefix for output files.
        pseudo_dir: str
            The path of the pseudopotentials (It must not end with "/").
        kpoints: list, default=[4,4,4]
            A list of the nk1, nk2, nk3 parameters, wich specify the k-point grid (nk1 x nk2 x nk3) as in Monkhorst-Pack grids.
        atoms_to_relax: None | list | tuple, default=None
            If set to list or tuple of indices (int values), the indices of the atoms to relax in the calculation. If set to None, all atoms will be relaxed.
        nosym: bool: Default=False
            Wether to write the "nosym = .true.," line in &SYSTEM section of the .in file, to not use structure simmetries in the simulation.
        electron_maxstep : int, default=200
            maximum number of iterations in a scf step.
        vdw: None | str, default=None
            If set into str, van der waals corrections are considered. The available options are the same than for vdw_corr option in pw.x input fles.
        dftd3_threebody: bool, default=True
            Wether to consider three-body terms in Grimme-D3 van der Waals correction.

    Output:
    -------
    This function writes the .in file to the pw.x QE simulation. the file is saved in '{path}/{filename}.in'.
    """
    assert type(electron_maxstep) == int, "\'electron_maxstep\' must be int type"

    #Create the folder where the file is going to be saved if doesn´t exist
    if not os.path.exists(f"{path}/"):
        os.makedirs(f"{path}/")
    # Open and read the model input file
    with open("model.in", 'r') as model:
        model = model.readlines()

    vectores = [" ".join([str(elemento) for elemento in vec]) + '\n' for vec in structure.lattice.matrix]

    # Get the species of each site
    site_species = [site.species_string for site in structure]

    # Get the list of elements present in the structure
    elementos = list(set([site.species_string for site in structure]))

    # Prepare atomic positions lines
    coords = [f'{es} ' + " ".join([str(elemento) for elemento in atomic_position]) + '\n' for es, atomic_position in zip(site_species,structure.cart_coords)]

    for i, line in enumerate(model):
        # Update the output directory and prefix in the input file
        if "outdir" in line:
            model[i] = f"outdir = '{outdir}' ,\n"
            continue
        if "prefix" in line:
            model[i] = f"prefix = '{prefix}' ,\n"
            continue

        # Update the number of atoms in the input file
        if "nat" in line:
            model[i] = f'nat = {structure.num_sites},\n'
            continue
        if "ntyp" in line:
            model[i] = f'ntyp = {len(structure.elements)},\n'
            continue

        # Update lattice vectors
        if "CELL_PARAMETERS" in line:
            for vec in reversed(vectores):
                model.insert(i+1, vec)
            continue

        if "ATOMIC_SPECIES" in line:
            for elemento in elementos:
                model.insert(i+1, _find_potential(elemento))
            continue

        # Add atomic positions
        if "ATOMIC_POSITIONS" in line:
            if atoms_to_relax == None:
                atoms_to_relax = range(len(structure))
            for j, coord in enumerate(coords):
                if not np.isin(j, atoms_to_relax):
                    model.insert(i+1, coord[:-1] + ' 0 0 0\n')  # Fixed atoms
                else:
                    model.insert(i+1, coord)  # Relaxed atoms
            continue

        if "pseudo_dir" in line:
            model[i]=f"pseudo_dir = '{pseudo_dir}/' ,\n"
            continue
        if "K_POINTS" in line:
            model[i+1] = " ".join(str(k) for k in kpoints)+"   0 0 0"
            continue
        if "smearing =" in line:
            if nosym:
                model.insert(i+1, "nosym = .true.,\n")
            if vdw:
                model.insert(i+1, f"vdw_corr = '{vdw}' ,\n")
                if vdw in ('grimme-d3', 'Grimme-D3', 'DFT-D3', 'dft-d3') and not dftd3_threebody:
                    model.insert(i+2, "dftd3_threebody = .false. ,\n")
            continue
        if "electron_maxstep" in line:
            model[i] = f"electron_maxstep = {electron_maxstep}\n"
            continue

    # Write the modified input file
    with open(path + f'/{file_name}', 'w') as file:
        for line in model:
            file.write(line)

def surface_in(surface, material, miller_index, path, kpoints=[4,4,1], file_name=None, pseudo_dir=None, nosym=False, electron_maxstep = 200,  vdw = None, dftd3_threebody = True):
    """
    Writes the input of a relaxation pw.x quantum expresso simulation for a given material, structure and Miller index.

    Parameters:
    -----------
        surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The atomic structure for which the input file will be generated. This structure should be compatible with the slab generation process.
        material : str
            The formula of the material, used to name output directories and files.
        miller_index : list of int
            The Miller index of the surface to be generated, provided as a list of three integers.
        path: str
            folder's dir where the input file is going to be saved.
        kpoints: list, default=[4,4,1]
            A list of the nk1, nk2, nk3 parameters, wich specify the k-point grid (nk1 x nk2 x nk3) as in Monkhorst-Pack grids.
        filename: str | None, default=None
            If given, the name of the output file. If is None, the name of the file is f'{material}_{index_str}.in'.
        pseudo_dir: str | None, default=None
            If given, the path of the pseudopotentials (It must not end with "/"). If None, the path is "../pseudos".
        nosym: bool: Default=False
            Wether to write the "nosym = .true.," line in &SYSTEM section of the .in file, to not use structure simmetries in the simulation.
        electron_maxstep : int, default=200
            maximum number of iterations in a scf step.
        vdw: None | str, default=None
            If set into str, van der waals corrections are considered. The available options are the same than for vdw_corr option in pw.x input fles.
        dftd3_threebody: bool, default=True
            Wether to consider three-body terms in Grimme-D3 van der Waals correction.

    Output:
    -------
    This function writes the .in file to the pw.x QE simulation. the file is saved, as default, in '{material}_{index_str}.in'.
    """

    # Determine atoms in the first and second layers
    layers = first_surface_layers(surface, n = 2)
    two_layers = layers[0] + layers[1]

    # Convert Miller index to string format for naming
    index_str = "".join([str(elemento) for elemento in miller_index])

    # Set .in file options
    if file_name==None:
        file_name=f'{material}_{index_str}.in'
    outdir = f"./{material}_{index_str}"
    prefix = f"{material}_{index_str}"
    if not isinstance(pseudo_dir, str):
        pseudo_dir = "../pseudos"

    structure_in(
        surface,
        path,
        file_name,
        outdir,
        prefix,
        pseudo_dir,
        kpoints = kpoints,
        atoms_to_relax = two_layers,
        nosym = nosym,
        electron_maxstep = electron_maxstep,
        vdw = vdw,
        dftd3_threebody = dftd3_threebody
    )

def _is_Li_in_model(model):
    """
    Identify if Li is one of the atomic species present in model. Model is a list of strings. Each string is one of the lines of an .in file of pw.x QE.

    Parameters:
    -----------
        model : list of str
            A list with the lines of an input file.
    """
    output = False
    in_section = False
    for line in model:
        if 'ATOMIC_SPECIES' in line:
            in_section = True
            continue
        if in_section:
            if 'Li' in line:
                output = True
                break
        if 'K_POINTS' in line:
            break
    return output

def ad_in_file_creation(model, surface, li_coord, name, site, k, path_in):
    """
    Modify the apropiate model for an .in file of an cleaned surface and add a Li atom in an given adsortion site. Then creates the .in file for the surface + Li.

    Parameters:
    --------------------
        model: list
            A list of strings. The strings correspondes to the lines of the .in file of the cleaned surface.
        surface: pymatgen.core.structure.Structure
            The slab object from which the .in file is created.
        li_coord: list | tuple
            A list or tuple of the coordinates of the adsortion site.
        name: str
            base of the prefix and outdir.
        site: str
            Type of the adsortion site. Options: "hollow", "bridge", "ontop".
        k: int
            Number that identify the adsortion site.
        path_in: string
            The path of the folder in which the generated .in files are going to be saved.

    Returns:
    --------------------
        The modified model (a list of strings, each string being each line of the .in created file).
        Creates a .in file for the surface + Li system.
    """
    model2 = model.copy()
    add_li = f'Li ' + " ".join([str(elemento) for elemento in li_coord]) + '\n'
    Li_in_model_2 = _is_Li_in_model(model2)
    for i, line in enumerate(model2):
        if 'outdir' in line:
            model2[i] = f"outdir = './{name}_{site}_{k}' ,\n"
            continue
        if 'prefix' in line:
            model2[i] = f"prefix = '{name}_{site}_{k}' ,\n"
            continue
        if 'nat' in line:
            model2[i] = f'nat = {surface.num_sites + 1},\n'
            continue
        if 'ntyp' in line and not Li_in_model_2:
            model2[i] = f'ntyp = {len(surface.elements) + 1},\n'
            continue
        if 'ATOMIC_SPECIES' in line and not Li_in_model_2:
            model2.insert(i+1, _find_potential('Li'))
            continue
        if 'K_POINTS' in line:
            li_position_index = i-1
            break
    model2.insert(li_position_index, add_li)

    with open(path_in + f'/{name}_{site}_{k}.in', 'w') as file2:
        for line in model2:
            file2.write(line)

    return model2

def _redefine_structure_atomic_positions(surface, new_atomic_positions):
    """
    Returns a copy of surface, which is a pymatgen´s slab. The copy has new atomic coordinates given by new_atomic_positions, which is a list of strings of the form "{atomic_specis}      {x_positions}      {y_positions}      {z_positions}"} taken from an .in or .out QE pw.x file.

    Parameters:
    -----------
        surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            Slab object which atomic positions modify.
        new_atomic_positions : str
            List of strings of the form "{atomic_specis}      {x_positions}      {y_positions}      {z_positions}"} taken from an .in or .out QE pw.x file

    Returns:
    -----------
        new_surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            A copy of surface with new atomic positions.
    """
    processed_atomic_positions = []
    for line in new_atomic_positions:
        # Split the line into components, handling multiple spaces
        parts = line.split()
        # The first part is the element (string)
        element = parts[0]
        # The next three parts are coordinates (convert to float)
        new_cart_coords = [float(x) for x in parts[1:4]]
        # Transform cartesian coordinates into fractional coordinates
        new_frac_coords = surface.lattice.get_fractional_coords(new_cart_coords)
        # Combine into a single list and add to results
        processed_atomic_positions.append([element, new_frac_coords])

    # Make a deep copy of surface and change its coordinates into the new ones
    new_surface = surface.copy()
    for nueva_pos in processed_atomic_positions:
        el = nueva_pos[0]
        coord = nueva_pos[1]
        new_surface.append(el, coord)  # Updates the position of each atom.

    new_surface.remove_sites(range(len(processed_atomic_positions))) # Remove old atomic positions

    return new_surface

def ads_in(surface, file_out, file_in, path_in=None, verbose=False):
    """
    Create pw.x input files for slabs with adsorbed Li atoms from a .out file obtained from a pw.x relaxation calculation.

    This function takes the relaxed atomic positions from the .out file and return .in files for each adsortion site detected by pymatgen.analysis.adsorption's AdsorbateSiteFinder. Returns a dictionary with the surface´s adsorption sites.

    Parameters:
    -----------
        material_id : str
            The MP identifier for the material, used to name output directories and files.
        miller_index : list of int
            The Miller index of the surface, provided as a list of three integers.
        surface: pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
            The structure of the material. Must match the species and number of atoms of the .out file
        file_out: str
            Path direction of the relax.out file of the clean surface (i.e., without atoms adsorbed).
        file_in: str
            Name of the .in file of the clean surface, which was used to run de QE pw.x relax calculation from we got the relax.out file
        path_in: str | None, Default: None
            If provided, the folder's dir where the .in files of the surface with atoms adsorbed are going to be saved. If set to None, the .in files will be saved in the same folder of the relax.out file.
        verbose : bool, default=False
            Wether to print information about the process.
    Returns:
    -----------
        Returns a dictionary of adsorption sites. The keys are 'bridge', 'hollow', 'ontop', and the values are the cartesian coordinates.   
    """

    #Extract the final atomic positions from the .out file
    print("\tExtrayendo coordenadas de los átomos relajados...", end="\t") if verbose else None
    new_atomic_coordinates = extract_relaxed_positions_from_out(file_out)
    print("\tHecho") if verbose else None

    # Read .in of cleaned surface file in model and replace its atomic positions by new_atomic_coordinates.
    print("\tModificamos el modelo de tarjeta de entrada con las nuevas coordenadas relajadas...", end="\t") if verbose else None
    model = redefine_atomic_positions_in(file_in, new_atomic_coordinates)
    print("\tHecho") if verbose else None

    # Transform the atomic_positions into a list of 4 elements lists, with their first element of type string and the next three elements of type float
    # Make a deep copy of surface and change its coordinates into the new ones
    print("\tModificamos las coordenadas en el objeto Structure que almacena el slab...", end="\t") if verbose else None
    new_surface = _redefine_structure_atomic_positions(surface, new_atomic_coordinates)
    print("\tHecho") if verbose else None

    # For each adsortion site, create a new input file
    print("\tDeterminando los sitios de adsorción...") if verbose else None
    if path_in == None:
        path_in = os.path.dirname(file_out)
    name = Path(file_out).stem  # nombre del archivo _relax.out sin ruta ni extensión
    name = name.replace("_relax", "")  # nombre del archivo sin sufijo _relax
    AF = AdsorbateSiteFinder(new_surface)
    sites = ['ontop', 'hollow', 'bridge']
    z_li = max(new_surface.cart_coords[:, 2]) + 2.0
    for site in sites:
        coords = AF.find_adsorption_sites()[site]
        print(f"\t\tCantidad de sitios tipo {site}: {len(coords)}") if verbose else None
        for k, coord in enumerate(coords):
            print(f"\t\t\tCreando archivo .in para sitio {site} número {k}") if verbose else None
            coord[-1] = z_li
            ad_in_file_creation(model, surface, coord, name, site, k, path_in)
    return AF.find_adsorption_sites()

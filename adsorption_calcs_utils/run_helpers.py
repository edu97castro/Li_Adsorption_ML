# utils/run_helpers.py

import os
import re

import pandas as pd
from joblib import load
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib.pyplot as plt
import numpy as np

from .structure_utils import count_layers_local, modify_vac_size, get_slab_size, slab_truncator, first_surface_layers
from .in_creation_utils import ads_in

def _verify_calculation_history(material, material_id, idx, csv_path='./historial_de_calculos.csv'): # Internal use only
    """
    Verify if a surface given by material, material_id and idx (Miller index) is already loaded in calculations history database.

    Parameters:
    -----------
        material : str
            Material´s pretty formula.
        material_id : str
            Material´s ID in MP.
        idx : int
            Miller index in int format, for example, 111 represents (1,1,1) Miller index
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.

    Returns:
    -----------
        True if the surface is already loaded in calculations history database. Otherwise, False.
    """
    df_history = pd.read_csv(csv_path, low_memory=False)
    df_filtered_history = df_history[
        (df_history['material'] == material) &
        (df_history['material-id'] == material_id) &
        (df_history['indice_de_miller'] == idx)
    ]
    if len(df_filtered_history) != 0:
        output = True
    else:
        output = False
    return output

def _replace_history_row(material, material_id, idx, new_row, csv_path='./historial_de_calculos.csv'):
    """
    Replaces the row corresponding to material, material_id and idx in calculations history database by new_row. No return is given

    Parameters:
    -----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        idx : int
            Miller index in integer format. For example, 100, 110, ...
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
        new_row : pandas.DataFrame
    """
    df_history = pd.read_csv(csv_path, low_memory=False)
    indice = df_history[
        (df_history['material'] == material) &
        (df_history['material-id'] == material_id) &
        (df_history['indice_de_miller'] == idx)
    ].index[0]
    df_history.loc[indice] = new_row.iloc[0]
    df_history.to_csv(csv_path, index=False)

def _prepare_surface(surface, n, vac_size, tolerance):
    """
    Return a copy of surface for the creation of the .in files to the pw.x simulations, with required number of layers and vacuum size. This function doesn´t modify surface in place.

    Parameters:
    -----------
        surface: pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
            Surface from which create the .in file.
        n: int
            Number of layers of the surface required for the simulation.
        vac_size : float
            Size of vacuum size along the c-axis required for the simulation.
        tolerance : float
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.

    Returns:
    -----------
        surface : pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
            A copy of surface with the required number of layers and vacuum size.
    """
    # Truncate surface´s layers
    surf_layers = count_layers_local(surface)
    truncated_surface = slab_truncator(surface, min(n, surf_layers), tolerance=tolerance)

    # Adjust vacuum size
    surface = modify_vac_size(truncated_surface, vac_size)

    return surface

def _save_calc_in_calculation_history(material, material_id, e_hull, idx, surface, is_surface_stteped, csv_path='./historial_de_calculos.csv'):
    """
    Saves new surface in calculations history database, or update its row if already exists. No return is given.

    Parameters:
    -----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        ehull : float
            Energy above hull of the material.
        idx : int
            Miller index in integer format. For example, 100, 110, ...
        surface : pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
            Pymatgen object of the surface.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
        is_surface_stteped : bool
            Whether the user identify the surface as an stteped one.
    """
    new_row = pd.DataFrame({
        'material' : [material],
        'material-id' : [material_id],
        'e_hull' : [e_hull],
        'indice_de_miller' : [idx],
        'capas' : [count_layers_local(surface)],
        'nsites' : [len(surface)],
        'altura_slab' : [get_slab_size(surface)],
        'a' : [surface.lattice.a],
        'b' : [surface.lattice.b],
        'c' : [surface.lattice.c],
        'alpha' : [surface.lattice.alpha],
        'beta' : [surface.lattice.beta],
        'gamma' : [surface.lattice.gamma],
        'escalonada' : [is_surface_stteped],
        'exito_limpia' : [pd.NA],
        'sitios_bridge' : [pd.NA],
        'sitios_hollow' : [pd.NA],
        'sitios_ontop' : [pd.NA],
        'sitios_totales' : [pd.NA],
        'bridge_enviados' : [pd.NA],
        'hollow_enviados' : [pd.NA],
        'ontop_enviados' : [pd.NA],
        'bridge_exitosos' : [pd.NA],
        'hollow_exitosos' : [pd.NA],
        'ontop_exitosos' : [pd.NA],
        'prob_convergencia' : [0],
        'prob_tiempo' : [0],
        'errores' : [0]
    })

    if _verify_calculation_history(material, material_id, idx):
        _replace_history_row(material, material_id, idx, new_row)
    else:
        df_history = pd.read_csv(csv_path, low_memory=False)
        df_history = pd.concat([df_history, new_row], ignore_index=True)
        df_history.to_csv(csv_path, index=False)

def _verify_surface_quality(file_name, csv_path='./historial_de_calculos.csv'):
    """
    Visualizes a surface from a QE pw.x .in file using ASE. The user must indicate 'y' if the surface is stteped or 'n' otherwise. Returns True if 'y' was written by the user, 'n' otherwise.

    Parameters:
    -----------
        file_name : str
            Path of the .in file.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.

    Returns:
    -----------
        bool, whether the user indicates the surface is stteped or no.
    """
    surface = load(file_name)
    surface = surface*(3,3,1)
    atoms_surface = AseAtomsAdaptor().get_atoms(surface)
    view(atoms_surface)

    while True:
        user_input = input("¿Is the surface stteped? (y/n): ").strip().lower()  # Convierte a minúscula y elimina espacios
        if user_input in ('y', 'n'):
            break
        print("Invalid input. Please enter 'y' o 'n'.")
    print(f"\nYou entered: '{user_input}'.\n\n")
    return user_input == 'y'

def _verify_calculation_result(file_path):
    """
    Verify if calculation given by material, material_id, idx Miller index and path was successfully achieved.

    Parameters:
    -----------
        file_path : str
            Path of the .out file

    Returns:
    -----
        calc_resul  t : str
            String containing the info of the final state of the calculation.
            Possible values are:
                'is_completed' : The calculation has succesfully completed
                'has_error' : The calculation hasn´t completed due an error
                'has_convergence_issue' : The calculation hasn´t completed since the SCF calculation didn´t converged
                'has_time_issue' : The calculation hasn´t converged since it requires excessive time
    """
    # Define strings that indicate, by their presence or not in the .out file, the final state of the calculation.
    succes_str = 'Final energy'
    no_convergence_str = "convergence NOT achieved"
    error_str = "stopping ..."
    time_exceeded_str = "Maximum CPU time exceeded"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Define variable which will save final calculation state. Unless one of the strings defined above are found in lines, the calculation failed due to time issues.
    calc_result = 'has_time_issue'

    # Iterate since last line
    for line in reversed(lines):
        # Verify if calculation was successfull
        if succes_str in line:
            calc_result = 'is_completed'
            break
        # Verify if calculation had an error
        elif error_str in line:
            calc_result = 'has_error'
            break
        # Verify if calculation hasn´t converged
        elif no_convergence_str in line:
            calc_result = 'has_convergence_issue'
            break
        elif time_exceeded_str in line:
            break

    return calc_result

def _create_all_adsorption_files(path, material, material_id, idx):
    """
    Creates a .in files for surface+Li for each adsorption site for a given surface (determined by material, material_id, idx). Returns a dictionary with the asorption sites´s coords.

    Parameters:
    -----------
        path : str
            Folder of the .in file of the cleaned surface, .out file of the relaxed surface and the .joblib file of the pymatgen slab object. The .in files of the surface+Li are going to be saved in the same folder.
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        idx : str
            Miller index in string format. For example, '100', '110', ...

    Output:
    -----------
        Returns a dictionary of adsorption sites. The keys are 'bridge', 'hollow', 'ontop', and the values are the cartesian coordinates.   
        Creates adsorption .in files in path.

    """
    # Files .in y .out
    file_in = f'{path}/{material}_{material_id}_{idx}.in'
    file_out = f'{path}/{material}_{material_id}_{idx}_relax.out'
    # Load slab pymatgen object
    surface = load(f'{path}/surface_{material}_{material_id}_{idx}.joblib')
    # Create adsorption .in files
    as_dict = ads_in(surface, file_out, file_in)
    return as_dict

def _count_adsorption_sites(idx, path):
    """
    Counts the number of adsroption sites in a folder for a surface given by a certain Miller index by counting the number of .in files in the folder with the Miller index in their names.

    Parameters:
    -----------
        idx: str
            Miller index in string formta, like '100', '110', '111'.
        path: str
            Path of the folder containing the .in files.

    Returns:
    -----------
        bridge_count, hollow_count, ontop_count, total_count : int
            Number of bridge, hollow, ontop and total sites.
    """
    bridge_count = 0
    hollow_count = 0
    ontop_count = 0

    for file_name in os.listdir(path):
        if idx in file_name and file_name.endswith("in"):
            if 'bridge' in file_name:
                bridge_count += 1
            if 'hollow' in file_name:
                hollow_count += 1
            if 'ontop' in file_name:
                ontop_count += 1
    total_count = bridge_count + hollow_count + ontop_count

    return bridge_count, hollow_count, ontop_count, total_count

def _generate_colors(n, colormap='gist_rainbow', alpha=1.0):
    """
    Generates a list of N distinct colors using a colormap of matplotlib.
    
    Parameters:
    -----------
        n : int
            Number of colors to generate.
        colormap : str, default='gist_rainbow'
            Name of colormap to use.
        alpha : float, default=1.0
            Reference´s transparence value. Must be between 0 and 1.
        
    Returns:
    --------
        list
            List of colors in RGB format.
    """
    # Obtener el colormap
    cmap = plt.get_cmap(colormap)
    
    # Generar valores equidistantes en el rango del colormap
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    
    # Ajustar alpha si es necesario
    if alpha != 1.0:
        colors = [(r, g, b, alpha) for r, g, b, a in colors]
    
    return colors

def _bidim_supercell(x_coords, y_coords, a, b, na, nb):
    """
    Given two lists of x and y coordinates of atoms in a primitive cell of a bidimensional lattice, generates lists of x and y coordinates of atoms in a supercell of size naxnb.

    Parameters:
    -----------
        x_coords : list | tuple | numpy.array
            list of x coordinates. Must have same size as y_coords.
        y_coords : list | tuple | numpy.array
            list of y coordinates. Must have same size as y_coords.
        a : list | tuple | numpy.array
            list of components of lattice vector a.
        b : list | tuple | numpy.array
            list of components of lattice vector b.
        na : int
            number of primitive cells along a axis.
        nb : int
            number of primitive cells along b axis.

    Returns:
    -----------
    sup_x_cords, sup_y_coords : numpy.array
        list of x and y coordinates of the atoms in the supercell.
    """
    sup_x_cords, sup_y_coords = [], []
    for i in range(na):
        for j in range(nb):
            displaced_x = x_coords + i * a[0] + j * b[0]
            displaced_y = y_coords + i * a[1] + j * b[1]
            sup_x_cords.extend(displaced_x)
            sup_y_coords.extend(displaced_y)
    return np.array(sup_x_cords), np.array(sup_y_coords)

def _show_ads(as_dict, path, material, material_id, idx, tolerance=0.85):
    """
    Shows adsorption´s sites  of a given surface.

    Parameters:
    -----------
        as_dict : Dictionary
            A dictionary with adsorption sites coords. The keys are 'bridge', 'hollow', 'ontop', and the values are the cartesian coordinates.
        path : str
            Folder of the .in file of the cleaned surface, .out file of the relaxed surface and the .joblib file of the pymatgen slab object. The .in files of the surface+Li are going to be saved in the same folder.
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        idx : str
            Miller index in string format. For example, '100', '110', ...
        tolerance : float, default=0.85
            Tolerance to determine layers.
        
    """
    surface = load(f'{path}/surface_{material}_{material_id}_{idx}.joblib')
    # Get first and second layers indices
    f_layers = first_surface_layers(surface, 2, tolerance=tolerance)
    first_layer = f_layers[0]
    second_layer = f_layers[1]
    # Creates list for x and y coordinates of each layer
    first_coord_x = []
    first_coord_y = []
    first_symbols = []
    for i in first_layer:
        coord = surface[i].coords
        first_coord_x.append(coord[0])
        first_coord_y.append(coord[1])
        first_symbols.append(surface[i].species_string)
    second_coord_x = []
    second_coord_y = []
    second_symbols = []
    for i in second_layer:
        coord = surface[i].coords
        second_coord_x.append(coord[0])
        second_coord_y.append(coord[1])
        second_symbols.append(surface[i].species_string)
    # Get lattice vectors to translate the primitive cell
    a = surface.lattice.a
    b = surface.lattice.b
    alpha = surface.lattice.alpha
    beta = surface.lattice.beta
    gamma = surface.lattice.gamma
    alpha, beta, gamma = np.array([alpha, beta, gamma])*np.pi/180
    a_vec = np.array([a*np.sin(beta), 0])
    b_vec = np.array([b * np.sin(alpha) * np.cos(gamma), b * np.sin(alpha) * np.sin(gamma)])
    # Generates first and second layers coordinates for an supercell of 3x3x1 primitive cells
        # (Done in this way to prevent problems in cases where alpha and beta aren´t equal to 90°)
    first_coord_x, first_coord_y = _bidim_supercell(first_coord_x, first_coord_y, a_vec, b_vec, 3, 3)
    second_coord_x, second_coord_y = _bidim_supercell(second_coord_x, second_coord_y, a_vec, b_vec, 3, 3)
    # Generates supercell for symbols (no displacement needed)
    first_symbols = first_symbols * 9  # 3x3 repetitions
    second_symbols = second_symbols * 9  # 3x3 repetitions
    # Graphic first and second layers atoms
    plt.figure(figsize=(14, 7))
    plt.scatter(second_coord_x, second_coord_y, label="Second layers atoms", s=400, marker="o", c='grey')
    plt.scatter(first_coord_x, first_coord_y, label="First layers atoms", s=400, marker="o", c='black')
    # Add atomic symbols as text labels
    for x, y, s in zip(first_coord_x, first_coord_y, first_symbols):
        plt.text(x, y, s, ha='center', va='center', color='white', fontweight='bold')
    for x, y, s in zip(second_coord_x, second_coord_y, second_symbols):
        plt.text(x, y, s, ha='center', va='center', color='white', fontweight='bold')
    markers = {
        "Bridge" : "^",
        "Hollow" : "s",
        "On-top" : "*"
    }

    # Change sites´s names for the figure    
    as_dict['Bridge'] =    as_dict.pop('bridge')
    as_dict['Hollow'] =    as_dict.pop('hollow')
    as_dict['On-top'] =    as_dict.pop('ontop')

    # Iterate over all adsorption sitestypes
    sites = ['Bridge', 'Hollow', 'On-top']
    for site in sites:
        coords = as_dict[site]
        coords = [np.array([coord[0],coord[1]])+a_vec+b_vec for coord in coords]
        colors = _generate_colors(len(coords))
        for k, coord in enumerate(coords):
            plt.scatter([coord[0]], [coord[1]], label=f"{site} {k}", s=100, marker=markers[site], c=colors[k])
    # Show limits of centered primitive cell
    origin_x = a_vec[0]+b_vec[0]
    origin_y = a_vec[1]+b_vec[1]
    cell_vertices = np.array([
        [origin_x, origin_y],
        [origin_x + a_vec[0], origin_y + a_vec[1]],
        [origin_x + a_vec[0] + b_vec[0], origin_y + a_vec[1] + b_vec[1]],
        [origin_x + b_vec[0], origin_y + b_vec[1]],
        [origin_x, origin_y]
    ])
    plt.plot(cell_vertices[:, 0], cell_vertices[:, 1], 'b-', linewidth=2, linestyle='--', c='grey', label="Primitive cell´s boundary") # Draw primitive cell´s border
    plt.title(f'Sitios {idx}')
    plt.legend(loc='upper left')  
    plt.show()


def _after_cleaned_calculation_update(material, material_id, idx, calc_result, n_b, n_h, n_o, csv_path='./historial_de_calculos.csv'):
    """
    Updates the calculation history database after finishing cleaned surface calculation. Updates 'exito_limpia', 'sitios_bridge', 'sitios_hollow', 'sitios_ontop' and 'sitios_totales' columns. No return is given.

    Parameters:
    -----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        idx : int
            Miller index in int format, like 100, 110 or 111.
        calc_result : str
            String containing the info of the final state of the calculation.
            Possible values are:
                'is_completed' : The calculation has succesfully completed
                'has_error' : The calculation hasn´t completed due an error
                'has_convergence_issue' : The calculation hasn´t completed since the SCF calculation didn´t converged
                'has_time_issue' : The calculation hasn´t converged since it requires excessive time
        n_b : int
             Number of bridge sites.
        n_h : int
             Number of hollow sites.
        n_o : int
            Number of ontop sites.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
    """
    df_history = pd.read_csv(csv_path, low_memory=False)

    # Find row´s index for material, material_id and idx
    indice = df_history[
        (df_history['material'] == material) &
        (df_history['material-id'] == material_id) &
        (df_history['indice_de_miller'] == idx)
    ].index[0]

    # Compute total number of sites
    n_t = n_b + n_h + n_o

    # Update database
    df_history.at[indice, 'sitios_bridge'] = n_b
    df_history.at[indice, 'sitios_hollow'] = n_h
    df_history.at[indice, 'sitios_ontop'] = n_o
    df_history.at[indice, 'sitios_totales'] = n_t

    # Update final cleaned surface calculation state
    df_history.at[indice, 'exito_limpia'] = (calc_result=='is_completed')

    # Saves updated history
    df_history.to_csv(csv_path, index=False)

def _failed_calculations_update(material, material_id, path, miller, csv_path='./historial_de_calculos.csv'):
    """
    Complete the columns in calculations history database that indicate the count of calculation problems.

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        path : str
            Path of the folder containing the .out files.
        miller : list of string
            List of Miller index in string format, like '100' or '111'.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
    """
    df_history = pd.read_csv(csv_path, low_memory=False)

    for idx in miller:
        indice = df_history[
            (df_history['material'] == material) &
            (df_history['material-id'] == material_id) &
            (df_history['indice_de_miller'] == int(idx))
        ].index[0]
        # List to save final states calculations
        calculations_results_per_idx = []
        for file_name in os.listdir(path):
            if idx in file_name and file_name.endswith("_relax.out"):
                calculations_results_per_idx.append(_verify_calculation_result(path+'/'+file_name))
        # Complete dataframe colums
        df_history.loc[indice, 'errores'] = calculations_results_per_idx.count('has_error')
        df_history.loc[indice, 'prob_convergencia'] = calculations_results_per_idx.count('has_convergence_issue')
        df_history.loc[indice, 'prob_tiempo'] = calculations_results_per_idx.count('has_time_issue')
        # Saves dataframe
        df_history.to_csv(csv_path, index=False)

def _parse_vasp_filename(filename):
    """
    Parse .out files´s names with format:
        {material}_{material_id}_{idx}_{site_type}_{site_number}_relax.out
    where material_id may contain strings like '_v2', '_v3', etc.

    Parameters:
    ----------
        filename : (str)
            File name without its diretory path (e.g.. "Li_mp-567337_v2_111_ontop_0_relax.out")

    Returns:
    ----------
        dict: A dictionary with the extracted parts:
            {
                'material': {material},
                'material_id': {material_id},
                'idx': {idx},
                'site_type': {site_type},
                'site_number': {site_number}
            }
    """
    pattern = r'''
        ^
        (?P<material>[^_]+)       # Material (hasta primer _)
        _
        (?P<material_id>.+?)      # Material ID (puede contener _v2, _v3)
        _
        (?P<idx>\d{3})           # Índice Miller (3 dígitos)
        _
        (?P<site_type>\w+)       # Tipo de sitio (bridge, hollow, ontop)
        _
        (?P<site_number>\d+)     # Número de sitio
        _relax\.out$
    '''
    match = re.match(pattern, filename, re.VERBOSE)

    if not match:
        raise ValueError(f"Formato de archivo no reconocido: {filename}")

    return match.groupdict()

def _get_elements_from_formula(formula):
    """
    Extracts all chemical elements present in a given chemical formula.

    Parameters:
    ----------
    formula :str
        A chemical formula string

    Returns:
        elements : tuple
            A tuple of element symbols present in the formula
    """
    # Remove all numbers and special characters to isolate element symbols
    # This regex matches element symbols (1-2 letters, first uppercase, second lowercase)
    return tuple(sorted(set(re.findall('[A-Z][a-z]?', re.sub('[0-9\[\]\(\)\{\}]', '', formula)))))

__all__ = [
    '_verify_calculation_history',
    '_prepare_surface',
    '_save_calc_in_calculation_history',
    '_verify_surface_quality',
    '_verify_calculation_result',
    '_create_all_adsorption_files',
    '_count_adsorption_sites',
    '_after_cleaned_calculation_update',
    '_failed_calculations_update',
    '_parse_vasp_filename',
    '_get_elements_from_formula',
    '_show_ads'
]

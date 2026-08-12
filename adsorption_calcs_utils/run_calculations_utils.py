# utils/run_calculations_utils.py

import os
from joblib import dump, load
import pandas as pd

from .cif_utils import get_structure_from_cif
from .structure_utils import create_slab, count_layers_local, cut_surface_n_first_layers
from .run_helpers import *
from .in_creation_utils import surface_in
from .in_out_management_utils import get_relaxed
from .diff_calculator_utils import EnergyDiffCalculator

def select_material(materials, n):
    """
    Select a material from a material list given by .cif_utils.get_material_and_cif

    Parameters:
    ----------
        materials : list of tuples
            A list of tuples provided by .cif_utils.get_material_and_cif. Each tuple contains, in order, the chemical formula (pretty format), the corresponding material ID from the MP database, the energy above hull and the number of sites in a primitive cell.
        n : int
            An indice to select a material in materials.

    Output:
    ----------
        material, material_id, e_hull, nsites, where:
            material: str
                Material´s chemical formula.
            material_id: str
                Material´s ID in MP.
            e_hull : float
                The energy above hull.
            nsites : int
                Number of sites in a primitive cell.

        Also prints material and material_id.
    """
    material = materials[n][0]
    material_id = materials[n][1]
    e_hull = materials[n][2]
    nsites = materials[n][3]
    print(f"Material Elegido:\n{material}, {material_id}")
    return material, material_id, e_hull, nsites

def create_slabs_per_miller_index(material, material_id, miller=[(1,0,0),  (1,1,0), (1,1,1)], path="./slabs", min_slab=10.0, min_vac=20.0, center_slab=True, verbose=True, if_exists_do="nothing"):
    """
    Generate and returns pymatgen.core.surface.Slab objects from given material, Miller indices and slab and vacuum size parameters. Each Slab object is created from a cif file provided by Materials Project (MP).

    Saves each generated Slab object in f'{path}/{material}_{index_str}_{material_id}.joblib', where index_str is the Miller index in string: 111 for (1,1,1), and so on.

    If a file already exists and if_exists_do != "replace", the function performs no operation and returns the pymatgen.core.structure.Slab object stored in that file.

    If the file doesn't exist or if if_exists_do = "replace", the function creates the slab, saves it to the specified file and returns the Slab object.

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        miller : list of sublists/tuples of int, default=[(1,0,0),  (1,1,0), (1,1,1)]
            List of Miller indices. Each indice is given as a list or tuple of int, as [1,0,0] or [1,1,1].
        path: str, default="./slabs"
            Path of the folder where the .joblib objects with the slabs will be saved.
        min_slab: float, default=10.0
            Minimun size of the slab in Angstrom
        min_vac: float, default=20.0
            Minimun vacuum size along the c-axis in the supercell, in Angstrom.
        center slab, default=True
            Whether to center the slab in the supercell. If set to False, the slab is placed in the bottom of the supercell.
        verbose: bool, default=True
            Whether to print the number of atoms and layers of each slab.
        if_exists_do: str, default="nothing"
            What to do if an slab already exists.
                Options:
                    "nothing": Skip and don´t do nothing.
                    "replace": Replace the previous slab by a new one.
                    Each other entry would do the same as "nothing" option.

    Output:
    -------
        Save a pymatgen.core.surface.Slab object in a .joblib file for each Miller index, corresponding to the material identified with material and material_id.
    """
    print(material)

    # Get structure from cif file and save it in a pymatgen.core.structure.Structure object
    struct = get_structure_from_cif(material_id)

    # Create slab objects for each Miller index, and save it in an pymatgen.core.surface.Slab object
    for idx in miller:
        index_str = "".join([str(elemento) for elemento in idx])
        file_path = path+f'/{material}_{index_str}_{material_id}.joblib'
        print(f"{idx}:", end="\t")
        if not os.path.exists(file_path) or if_exists_do=="replace":
            surface = create_slab(struct, idx, min_slab=min_slab, min_vac=min_vac, center_slab=center_slab)
            if surface == None:
                continue
            dump(surface, file_path)
            if verbose == True:
                print(f"Cantidad de átomos: {len(surface)},\tCantidad de capas: {count_layers_local(surface)}")
            else:
                print("\n")
        else:
            print("El slab ya existe", end="\t")
            surface = load(file_path)
            if verbose == True:
                print(f"Cantidad de átomos: {len(surface)},\tCantidad de capas: {count_layers_local(surface)}")
            else:
                print("\n")

def cut_first_layers(material, material_id, layers_dic, version_name, path='./slabs', csv_path='./historial_de_calculos.csv'):
    """
    Load a pymatgen.core.surface.Slab object from a joblib file and creates a new copy without its first layers. This new version of the Slab object is saved in a ne joblib file.

    No return is given.

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        layers_dic : dict
            A dictionary that contains the info of the number of layers to cut depending of the Miller index. Its keys are Miller indices ('100', '110, for example) and its values are the number of layers to cut (int).
        version_name : str
            The material_id of the new surface will be material_id+version_name.
        path : str, default='./slabs'
            Path of the folder that will contain the generated joblib file.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.

    Output:
    ----------
        Saves the new copy of the Slab object with its first layers cut (if its corredponding value in layers_dic is non_zero) in a new joblib object named f'{material}_{miller_index}_{material_id}{version}.joblib'.
    """
    miller = layers_dic.keys()
    for idx in miller:
        if layers_dic[idx] != 0 and not _verify_calculation_history(material, material_id+version_name, int(idx), csv_path=csv_path):
            surface = load(f'{path}/{material}_{idx}_{material_id}.joblib')
            surface = cut_surface_n_first_layers(surface, layers_dic[idx])
            dump(surface, f'{path}/{material}_{idx}_{material_id}{version_name}.joblib')
            continue
        if _verify_calculation_history(material, material_id+version_name, int(idx), csv_path=csv_path):
            print(f"There is already a material in historial_de_calculos.csv identified as {material_id+version_name} with Miller index {int(idx)}. Please consult the database.")


def cleaned_in_files_creation(material, material_id, ehull, folder_path, miller=[(1,0,0),  (1,1,0), (1,1,1)], replace_in=False, num_layers=6, vac_size=20.0, k=8, pseudo_dir="/home/ecastro/trabajo_final/pseudos", nosym=True, vdw='DFT-D3', dftd3_threebody=False, tolerance=0.85, csv_path='./historial_de_calculos.csv'):
    """
    Creates the .in files for cleaned surfaces corresponding to material, material_id and Miller indices in miller parameter.

    No return is given

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        ehull : float
            Energy above hull of the material.
        miller : list of sublists/tuples of int, default=[(1,0,0),  (1,1,0), (1,1,1)]
            List of Miller indices. Each indice is given as a list or tuple of int, as [1,0,0] or [1,1,1].
        folder_path : str
            Path of the calculation family folder (Pure materials, materials+Li, ...)
        replace_in : bool, default=False
            Whether to replace the .in files if they are already created. Also replace the corresponding rows for each surface in calculations history database.
        num_layers : int, default=6
            Number of layers required for surfaces.
        vac_size : float, refault=20.0
            Size of vacuum layer along the c-axis in Angstrom.
        k : int, default=8
            Defines the k-point grid (k x k x 1) as in Monkhorst-Pack grids.
        pseudo_dir : str, default="/home/ecastro/trabajo_final/pseudos"
            Path of the pseudopotentials (It must not end with "/").
        nosym : bool, default=True
            Wether to write the "nosym = .true.," line in &SYSTEM section of the .in file, to not use structure simmetries in the simulation.
        vdw : str | None, default='DFT-D3'
            If set into str, van der waals corrections are considered. The available options are the same than for vdw_corr option in pw.x input fles.
        dftd3_threebody : bool, default=False
            Wether to consider three-body terms in Grimme-D3 van der Waals correction.
        tolerance : float, default=0.85
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.

    Output:
    ----------
        Saves the created .in files for pw.x simulations in folder_path+f"/{material}_{material_id}/{material}_{material_id}_{index_str}.in", and creates .joblib files to saves the costumized surfaces used in the creation of the pw.x inputs. The joblib files are saved as folder_path+f"/{material}_{material_id}/surface_{material}_{material_id}_{index_str}.joblib".

        Also, this function updates the calculations history database to add the new surfaces.
    """
    print("===========================")
    print(f"{material}_{material_id}")
    print("===========================")

    for idx in miller:
        # Transform miller index to string
        index_str = "".join([str(elemento) for elemento in idx])
        print(index_str)

        # Verify if the file for this surface is already created in calculations history database
        if _verify_calculation_history(material, material_id, int(index_str), csv_path=csv_path) and not replace_in:
            print("\tThe .in file was already created for this surface. See historial_de_calculos.csv.")
            continue

        # Load the surface given by material, material_id and idx and create a copy with required number of layers and vacuum size
        surface = load(f'./slabs/{material}_{index_str}_{material_id}.joblib')
        modified_surface = _prepare_surface(surface, num_layers, vac_size, tolerance)

        # Define path of the containing folder and the file
        path = folder_path+f"/{material}_{material_id}"
        file_name = f'{material}_{material_id}_{index_str}.in'

        # Creates .in file
        surface_in(
            modified_surface,
            material,
            idx,
            path,
            kpoints = [k,k,1],
            file_name = file_name,
            pseudo_dir = pseudo_dir,
            nosym = nosym,
            vdw = vdw,
            dftd3_threebody = dftd3_threebody
        )

        # Saves and visualize pymatgn slab object
        surface_file_path = f'{path}/surface_{material}_{material_id}_{index_str}.joblib'
        dump(modified_surface, surface_file_path)
        is_surface_stteped = _verify_surface_quality(surface_file_path, csv_path=csv_path)

        # Guardo información en df_history
        _save_calc_in_calculation_history(material, material_id, ehull, int(index_str), modified_surface, is_surface_stteped, csv_path=csv_path)

def ads_in_files_creation(material, material_id, folder_path, miller=['100', '110', '111'], csv_path='./historial_de_calculos.csv', tolerance=0.85):
    """
    Verifies if the cleaned surfaces calculations for material and material_id were achieved successfully. If this is the case, creates .in files for surfaces with adsorbed Li atoms.

    This function also updates the calculations history database with the cleaned surfaces calculations result and the number of adsorption sites, if cleaned surfaces calculations were successfull.

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        folder_path : str
            Path of the calculation family folder (Pure materials, materials+Li, ...)
        miller : list of str, default=['100', '110', '111']
            List of Miller indices. Each indice is given as a string, as '100' or '111'.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
        tolerance : float, default=0.85
            Tolerance to determine the layers.
    """
    print("========================")
    print(f'{material}_{material_id}')
    print("========================\n")
    # Path for the cleaned surfaces .out files
    for idx in miller:
        print(f'-----\n{idx}\n-----')
        path = f'{folder_path}/{material}_{material_id}'
        file_path = path+f'/{material}_{material_id}_{idx}_relax.out'
        # Verify if the surface is loaded in calculations history database
        is_surface_created = _verify_calculation_history(material, material_id, int(idx))
        if not is_surface_created:
            print(f"ERROR! The surface given by {material}, {material_id}, {idx} is not loeaded in the calculations history database.")
            break
        if not os.path.exists(file_path):
            print(f"El archivo {os.path.basename(file_path)} no existe.")
            continue
        calc_result = _verify_calculation_result(file_path)
        if calc_result == 'is_completed':
            as_dict = _create_all_adsorption_files(path, material, material_id, idx)
            n_b, n_h, n_o, n_t = _count_adsorption_sites(idx, path)
            # Print information
            print(f'Number of sites:')
            print(f'\tbridge: {n_b}')
            print(f'\thollow: {n_h}')
            print(f'\tontop:  {n_o}')
            print(f'\nSitios totales: {n_t}')
            _show_ads(as_dict, path, material, material_id, idx, tolerance=tolerance)
            print("\n")
        else:
            print("Calculation failed.")
            n_b, n_h, n_o = pd.NA, pd.NA, pd.NA
        _after_cleaned_calculation_update(material, material_id, int(idx), calc_result, n_b, n_h, n_o, csv_path=csv_path)
    _failed_calculations_update(material, material_id, path, miller, csv_path=csv_path) if is_surface_created else None

def complete_history_with_ads_calcs(folder_path, material, material_id, csv_path='./historial_de_calculos.csv'):
    """
    Updates calculations history database with the results of adsorption sites calculations. No return is given.

    Parameters:
    ----------
        folder_path : str
            Path of the calculation family folder (Pure materials, materials+Li, ...)
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        csv_path : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the calculations history database.
    """
    df_history = pd.read_csv('./historial_de_calculos.csv', low_memory=False)
    miller = ["100", "110", "111"]
    path = f'{folder_path}/{material}_{material_id}'

    for idx in miller:
        indice = df_history[
            (df_history['material'] == material) &
            (df_history['material-id'] == material_id) &
            (df_history['indice_de_miller'] == int(idx))
        ].index[0]

        # Create lists to save the final state of bridge, hollow, ontop and cleaned surfaces calculations.
        bridge_list = []
        hollow_list = []
        ontop_list = []
        cleaned_list = []

        for file_name in os.listdir(path):
            if idx in file_name and file_name.endswith("out"):
                if 'bridge' in file_name:
                    bridge_list.append(_verify_calculation_result(f'{path}/{file_name}'))
                elif 'hollow' in file_name:
                    hollow_list.append(_verify_calculation_result(f'{path}/{file_name}'))
                elif 'ontop' in file_name:
                    ontop_list.append(_verify_calculation_result(f'{path}/{file_name}'))
                else:
                    cleaned_list.append(_verify_calculation_result(f'{path}/{file_name}'))

        # Complete last columns refered to adsorption sites in calculations history database for idx Miller index
        df_history.at[indice, 'bridge_enviados'] = len(bridge_list)
        df_history.at[indice, 'hollow_enviados'] = len(hollow_list)
        df_history.at[indice, 'ontop_enviados'] = len(ontop_list)
        df_history.at[indice, 'bridge_exitosos'] = len([state for state in bridge_list if state == 'is_completed'])
        df_history.at[indice, 'hollow_exitosos'] = len([state for state in hollow_list if state == 'is_completed'])
        df_history.at[indice, 'ontop_exitosos'] = len([state for state in ontop_list if state == 'is_completed'])
        # Complete columns refered to calculation problems
        result_list = bridge_list+hollow_list+ontop_list+cleaned_list
        df_history.at[indice, 'prob_convergencia'] = len([state for state in result_list if state == 'has_convergence_issue'])
        df_history.at[indice, 'prob_tiempo'] = len([state for state in result_list if state == 'has_time_issue'])
        df_history.at[indice, 'errores'] = len([state for state in result_list if state == 'has_error'])
    df_history.to_csv(csv_path, index=False)

def update_energy_adsorption_database(material, material_id, folder_path, if_energies_are_in_base_do='nothing', energies_csv='./base_energias.csv'):
    """
    Update adsorption energies database with the final results of the calculations of all surfaces of a given material. No return is given.

    Parameters:
    ----------
        material: str
            Material´s chemical formula.
        material_id: str
            Material´s ID in MP.
        folder_path : str
            Path of the calculation family folder (Pure materials, materials+Li, ...)
        if_energies_are_in_base_do : str, default='nothing'
            Indicates what to do if there are rows in adsorption energies database of the given material. Options:
                - 'nothing': Don´t do nothing. A warning message is printed.
                - 'replace': Remove all material´s rows and add the new rows got from the .out files.
        energies_csv : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the adsorption energiesdatabase.
    """
    df_energies = pd.read_csv(energies_csv, low_memory=False)
    # Path of the .out files
    path = f'{folder_path}/{material}_{material_id}'
    # Create a instance of EnergyDiffCalculator to compute the adsorption energies
    outs = get_relaxed(path)
    calculator = EnergyDiffCalculator(outs)
    df = calculator.to_dataframe()

    # Apply the parse function to each path
    if len(df) != 0:
        df[['material', 'material id', 'miller index', 'site type', 'site number']] = df['Adsorption Path'].apply(
            lambda x: pd.Series(_parse_vasp_filename(x))
        )
        df = df[['material', 'material id', 'miller index', 'site type', 'site number', 'Adsorption energy']]

        # Verify if the calculations for the material are already loaded in the database
        df_filtered = df_energies[
            (df_energies['material'] == material) &
            (df_energies['material id'] == material_id)
        ]
        if len(df_filtered) == 0:
            df_energies = pd.concat([df_energies, df], ignore_index=True)
        elif if_energies_are_in_base_do == 'replace':
            # Removo the row given by material and material_id
            mask = (df_energies['material'] == material) & (df_energies['material id'] == material_id)
            df_energies.drop(df_energies[mask].index, inplace=True)
            # Add new data
            df_energies = pd.concat([df_energies, df], ignore_index=True)
        else:
            print("The adsorption energies of the given material´s surfaces where already loaded in adsorption energies database.")
        # Save new dataframe in csv file
        df_energies.to_csv(energies_csv, index=False)

def adsorption_energies_count(energies_csv='./base_energias.csv', elements=None):
    """
    Prints the count of adsorption sites calculated for each chemical elements combination and adsorption type, and the total cases succesfully calculated (surface+adsorption site).  No return is given.

    Parameters:
    ----------
        energies_csv : str, default='./historial_de_calculos.csv'
            Path of the CSV file containing the adsorption energiesdatabase.
        elements : list of str | None,default=None
            If given, prints only the rows where material contain any of the elements.
    """
    df_energies = pd.read_csv(energies_csv, low_memory=False)

    # Create new column with element combinations
    df_energies['element_combination'] = df_energies['material'].apply(_get_elements_from_formula)

    # Count adsorption sites for each element combination and site type
    result = df_energies.groupby(['element_combination', 'site type']).size().unstack(fill_value=0)
    if isinstance(elements, list):
        mask = result.index.get_level_values('element_combination').map(
            lambda combo: all(el in combo for el in elements)
        )
        result_fil = result[mask]
    else:
        result_fil = result

    print(result_fil)

    total_cases = result['bridge'].sum() + result['hollow'].sum() + result['ontop'].sum()
    print(f"\n\nTotal cases calculated (surface+adsorption site): {total_cases}")
    print(f"Number of chemical elements combinations tried: {len(result)}")

def consult_history_db(material, material_id=None, idx=None, history_csv='./historial_de_calculos.csv'):
    """
    Returns a pandas DataFrame with the rows in calculations history database (saved in the file given by history_csv) corresponding with given material and material ID (and Miller index if provided).

    Parameters:
    ----------
        material: str
            Requested material´s chemical formula.
        material_id: str | None, default=None
            Material´s ID in MP. If set to None, all rows of material will be returned.
        idx: list of integers | None, default=None
            A list of integers representing Miller indices of the requested rows, like '100', '111'. If set to None, the rows with 100, 110 and 111 Miller indices will be returned.
        history_csv : str, default='./historial_de_calculos.csv'
            Path of the CSV file that saves the calculations history database.

    Returns:
    ----------
        filtered_df: pandas.DataFrame
            A pandas dataframe with the rows corresponding to material, material_id, and if it was requested, the idx Miller index.
    """
    df_history = pd.read_csv(history_csv, low_memory=False)
    # Set miller indices
    if isinstance(idx, list):
        miller = idx
    else:
        miller = [100, 110, 111]

    if isinstance(material_id, str):
        df_filtered = df_history[
            (df_history['material'] == material) &
            (df_history['material-id'] == material_id) &
            (df_history['indice_de_miller'].isin(miller))
        ]
    else:
        df_filtered = df_history[
            (df_history['material'] == material) &
            (df_history['indice_de_miller'].isin(miller))
        ]
    
    return df_filtered

def consult_energies_database(material, material_id, idx=None, site_types=None, energies_csv='./base_energias.csv'):
    """
    Returns a pandas DataFrame with the rows in energies database (saved in the file given by energies_csv) corresponding with given material and material ID (and Miller index if provided).

    Parameters:
    ----------
        material: str
            Requested material´s chemical formula.
        material_id: str
            Requested material´s ID in MP.
        idx: list of integers | None, default=None
            A list of integers representing Miller indices of the requested rows, like '100', '111'. If set to None, the rows with 100, 110 and 111 Miller indices will be returned.
        site_types: list of str, default=None
            Types of adsorption sites requested. If set to None, return the rows corresponding to all adsorption sites types (bridge, hollow and ontop).
        energies_csv : str, default='./base_energias.csv'
            Path of the CSV file that saves the energies database.

    Returns:
    ----------
        filtered_df: pandas.DataFrame
            A pandas dataframe with the rows corresponding to material, material_id, and if it was requested, the idx Miller index.
    """
    df_energies = pd.read_csv(energies_csv, low_memory=False)
    # Set miller indices
    if isinstance(idx, list):
        miller = idx
    else:
        miller = [100, 110, 111]
    # Set adsorption sites types
    if isinstance(site_types, list):
        ads_types = site_types
    else:
        ads_types = ['bridge', 'hollow', 'ontop']
    # Filter database
    df_filtered = df_energies[
        (df_energies['material'] == material) &
        (df_energies['material id'] == material_id) &
        (df_energies['miller index'].isin(miller)) &
        (df_energies['site type'].isin(ads_types))
    ]
    return df_filtered

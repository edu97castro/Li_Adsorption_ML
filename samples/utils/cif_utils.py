# utils/cif_utils.py

import pandas as pd

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def _count_unique_elements(formula): # Internal use only
    """
    Function to count unique elements based on uppercase letters from a formula.

    Parameters:
    ----------
        formula: str
            Formula of a material.

    Returns:
    -------
        int value, number of unique chemical elementes in formula.
    """
    # Count uppercase letters, which indicate unique elements
    return sum(1 for char in formula if char.isupper())

def _are_elements_in_formula(formula, elements): # Internal use only
    """
    Indicates if the elementes provided are in formula.

    Parameters:
    ----------
        formula: str
            chemical formula of an material.
        elements: list of str
            list of chemical symbols of elements.

    Returns:
    -------
        True if all chemical symbols in elements are present in formula, False otherwise. This function can differentiates correctly different chemical symbols with the same first letter, for example, "C" and "Cd".
    """
    assert isinstance(elements, list), "'elements' parameter must be a list o strings"
    assert "" not in elements, "The 'elements' attribute must not contain an empty string"
    assert len(elements) > 0, "The 'elements' attribute must contain at least one string representing a chemical element"
    assert isinstance(formula, str) and len(formula) > 0, "The 'formula' attribute must be a non-empty string"

    output = True
    for el in elements:
        if el in formula and len(el)>1:
            continue
        # If el is in formula, but its length is 1, Is necessary to verify if the element found in formula is el or another element with chemical formula starting with
            # the same letter.
        elif el in formula:
            # split formula by el, and verify the substring that follows el
            cut_formula = formula.split(el, 1)
            # If the substring that follows el is not empty, verify if its first character is an lowercase letter
            if len(cut_formula[1]) >0:
                output = output and not(cut_formula[1][0].islower())
            # If the substring that follows el is empty, then el in formula correspond to the dessire chemical element
            else:
                continue
        else:
            # If el is not in formula, then the dessire chemical element is not in the provided formula
            output = False
    return output

def get_material_and_cif(e_hull_max, composition=1, elements=None, filter_formula_by_min=None, order_by_ehull = True, csv_file = './materials_project_DB.csv', verbose = False):
    """
    Retrieve a list of materials from MP database. The list is a list of tuples, each one with the material´s formula, material´s ID in MP database, its energy above hull and its number of sites in a primitive cell.

    The selected materials are filtered to those that have:
        - energy above hull below certain value
        - a certain number of chemical elements
        - certain chemical elements in their composition, if required.

    For materials with the same formula you can choose to keep them all, only the material with lowest energy above hull or only the material with lowest number of sites in a primitive cell.

    Parameters:
    ----------
        e_hull_max : float
            The maximum energy above hull value for filtering materials.
        composition : int, default=1
            The number of unique elements required in the material formula.
        elements: list of string, default=None
            The chemical formulas of the elements we want in the material.
        filter_formula_by_min: None | string, default=None
            Criteria to filter the materials with the same formula. Options:
                - "nsites": Keep the material with lowestnumber of sites in primitive cell
                - "energy_above_hull": Keep the material with the lowset energy above hull
                - None: Keep all the materials with same formula.
        order_by_ehull : bool, default=True
            Wether to sort in increasing order the materials in the output list by their energy above hull.
        csv_file : str, default='./materials_project_DB.csv'
            Path of the CSV file containing the MP materials database.
        verbose : bool, default=False
            Wether to print the output list of materials.

    Returns:
    -------
    material_list : list of tuples
        A list of tuples where each tuple contains the chemical formula (pretty format), the corresponding material ID from the MP database, the energy above hull and the number of sites in a primitive cell.
    """
    assert isinstance(elements, list), "'elements' parameter must be a list o strings"

    # Load the material database CSV file into a DataFrame
    material_DB = pd.read_csv(csv_file, low_memory=False)
    
    # Filter the DataFrame to include only materials with energy above hull less than e_hull_max
    material_DB_filtered = material_DB[material_DB['energy_above_hull'] < e_hull_max]

    if isinstance(elements, list):
        material_DB_filtered = material_DB_filtered[
            material_DB_filtered['formula_pretty'].apply(lambda x: _are_elements_in_formula(x, elements))
        ]
    
    # Group by 'formula_pretty' and get the row with the minimum 'nsites' or 'e_above_hull" in each group
    if isinstance(filter_formula_by_min, str):
        material_DB_min_grouped = material_DB_filtered.loc[
            material_DB_filtered.groupby('formula_pretty')[filter_formula_by_min].idxmin()
        ]
    else:
        material_DB_min_grouped = material_DB_filtered
    
    # Filter tuples based on the number of unique elements in the formula
    material_list = [
        (row.formula_pretty, row.material_id, row.energy_above_hull, row.nsites) 
        for row in material_DB_min_grouped.itertuples(index=False)
        if _count_unique_elements(row.formula_pretty) == composition
    ]
    
    # Order materials by e_above_hull if order_by_ehull == True
    if order_by_ehull:
        material_list = _order_materials_list_by_ehull(material_list)
    
    # Print the list if verbose
    if verbose:
        _print_materials_list(material_list)

    return material_list

def _order_materials_list_by_ehull(materials_list): # Internal use only
    """
    Sort in increasing order the materials in the materials_list provided by get_material_and_cif by their energy above hull.

    Parameters:
    ----------
        materials_list : list
            The list of materials. Each element is a tuple-like object of the form ({material}, {material_id}, {ehull}, {nsites}), where:
                material : str : the formula of the material
                material_id : str : the ID of the material in MP
                ehull : float : energy above hull of the material
                nsites : int : Number of sites in a primitive cell

    Returns:
    -------
        ordered_material_list : list of tuples
            A list of the tuples contained in  materials_list sorted in increasing order by ehull.
    """
    ordered_material_list = sorted(materials_list, key=lambda x: x[2])
    return ordered_material_list

def _print_materials_list(materials_list): # Internal use only
    """
    Print a table with the materials´s info from materials_list, where materials_list is a list like the given from get_materials_and_cif function. No return is given.

    Parameters:
    ----------
        materials_list : list
            The list of materials. Each element is a tuple-like object of the form ({material}, {material_id}, {ehull}, {nsites}), where:
                material : str : the formula of the material
                material_id : str : the ID of the material in MP
                ehull : float : energy above hull of the material
                nsites : int : Number of sites in a primitive cell.
    """
    print("Index\tMaterial\tMaterial ID\tEnergy above hull\tNumber of sites in primitive cell")
    for index, mat in enumerate(materials_list):
        print(f"{index}\t{mat[0]:10}\t{mat[1]:10}\t{mat[2]:6.5f}\t\t\t{mat[3]}")
    print(f"\nNumber of materials: {len(materials_list)}")

def get_structure_from_cif(cif_id, path="./cif/", primitive=True):
    """
    Parse and return the crystal structure from a CIF file.

    This function loads a CIF file corresponding to the given material ID, parses it, and returns the crystal structure.

    Parameters:
    ----------
        cif_id : str
            The material ID from which the CIF file is to be loaded and parsed.
        path : str, default="./cif/"
            The path of the directory containing the .cif files
        primitive : bool, default=True
            Whether to return primitive unit cells

    Returns:
    -------
        standardized_structure : pymatgen.core.structure.Structure
            The parsed crystal structure from the CIF file.
    """    
    # Construct the full path to the CIF file using the material ID
    cif_file = path + f"{cif_id}.cif"
    
    # Parse the CIF file to extract the crystal structure
    parser = CifParser(cif_file)
    structure = parser.parse_structures(primitive=primitive)[0]

    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    standardized_structure = sga.get_conventional_standard_structure()
    
    return standardized_structure

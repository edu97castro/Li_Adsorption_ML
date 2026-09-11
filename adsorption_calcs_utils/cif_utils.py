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
    cif_file = path + f"/{cif_id}.cif"
    
    # Parse the CIF file to extract the crystal structure
    parser = CifParser(cif_file)
    structure = parser.parse_structures(primitive=primitive)[0]

    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    standardized_structure = sga.get_conventional_standard_structure()
    
    return standardized_structure

__all__ = [
    'get_structure_from_cif'
]

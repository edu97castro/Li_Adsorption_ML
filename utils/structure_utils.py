# utils/structure_utils.py

import numpy as np

from scipy.spatial.distance import pdist
from pymatgen.core.surface import SlabGenerator

def get_layers_indices(surface, tolerance=0.85):
    """
    Detects the layers in a structure. Gives a list of sublists, each sublist corresponds to one of the layers, and its elements are the indexs of the atoms (in the structure) that belong to that layer. The first sublist corresponds to the first layer, and so on.

    Parameters:
    ----------
        surface: pymatgen.core.structure.Structure
            Structure object.
        tolerance: float, default=0.85
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.

    Returns:
    ----------
        layers: list of lists
            The i-th sublist have the indexs of the atoms in structure that belong to the (i+1)-th layer.
    """
    atomic_positions = surface.cart_coords

    remaining_index = range(len(atomic_positions))
    remaining_positions = atomic_positions[remaining_index]
    remaining_z_coords = [pos[-1] for pos in remaining_positions]

    layers = []

    while len(remaining_index)>0:
        # If there is just one index left, it´s a whole layer
        if len(remaining_index)==1:
            layers.append(remaining_index)
            break

        remaining_distances = pdist(remaining_positions)
        remaining_min_distance = np.min(remaining_distances)
        remaining_max_z = np.max(remaining_z_coords)

        lay = []

        for i in range(len(remaining_z_coords)):
            if np.abs(remaining_positions[i][-1] - remaining_max_z) < tolerance:
                lay.append(remaining_index[i])

        remaining_index = [index for index in remaining_index if index not in lay]
        remaining_positions = atomic_positions[remaining_index]
        remaining_z_coords = [pos[-1] for pos in remaining_positions]

        layers.append(lay)

    return layers


def count_layers_local(surface, tolerance=0.85):
    """
    Counts the number of layers of an pymatgen´s structure object along the c-axis.

    Parameters:
    -----------
        surface: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            Pymatgen´s structure object which layers count.
        tolerance: float, default=0.85
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.

    Returns:
    --------
        int, the number of layers of the surface
    """
    layers = get_layers_indices(surface, tolerance)
    return len(layers)

def first_surface_layers(surface, n, tolerance=0.85):
    """
    Detects the first n layers in a structure. Gives a list o sublists, each sublist corresponds to one of the n first layers, and its elements are the indexs of the atoms (in the structure) that belong to the layer. The first sublist corresponds to the first layer, and so on.

    Parameters:
    ----------
        surface: pymatgen.core.structure.Structure
            Structure object.
        n: int
            Number of layers to detect.
        tolerance: float, default=0.85
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.

    Returns:
    ----------
        layers: list of lists
            The i-th sublist have the indexs of the atoms in structure that belong to the (i+1)-th layer.
    """
    layers = get_layers_indices(surface, tolerance)

    if n <= len(layers):
        first_layers = layers[:n]
    else:
        first_layers = layers
        print(f"WARNING: The slab has only {len(layers)} layers. Returning the lists of indices of all the surface´s layers.")
    return first_layers

def create_slab(structure, miller_index, min_slab=10.0, min_vac=20.0, center_slab=True):
    """
    Generates and returns a surface slab for a given Miller index with specified vacuum and slab size parameters.

    This function generates a surface slab using the provided structure, Miller index, and dimensional constraints.

    Parameters:
    -----------
        structure : pymatgen.core.structure.Structure
            The atomic structure from which to generate the surface slab.
        miller_index : list of int
            The Miller index of the surface to be generated, provided as a list of three integers.
        min_slab : float
            The minimum thickness of the slab in Ångströms.
        min_vac : float
            The minimum thickness of the vacuum layer in Ångströms.
        center_slab: bool, optional, Default=True
            Whether to center the surface slab in the supercell. If set to False, the slab will be set in the bottom of the supercell.

    Returns:
    --------
        surface : pymatgen.core.surface.Slab
            The generated surface slab.
    """
    slabgen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=min_slab,  # Ajusta según necesidad
        min_vacuum_size=min_vac,
        lll_reduce=True,   # Ayuda a evitar escalones
        center_slab=True,   # Centra el slab en el vacío
    )
    slabs = slabgen.get_slabs()

    if len(slabs)==0:
        index_str = "".join([str(elemento) for elemento in miller_index])
        print(f"Advertencia: No se identificaron slabs con índices de Miller {index_str}. Devolviendo None.")
        return None
    else:
        surface = slabs[0]

    # Translate the slab to the bottom of the superpell if center_slab == False
    if center_slab == False:
        surface = translate_slab(surface, False)

    # Scale the slab in the x and y directions
    return surface

def translate_slab(surface, to_center):
    """
    Takes an pymatgen´s Slab or Structure object and returns a copy translated to the bottom or the center of the supercell, depending of the value of to_center.

    Parameters:
    -----------
        surface : pymatgen.core.surface.Slab | pymategen.core.structure.Structure
            Pymatgen´s object with the structure of the slab.
        to_center : bool
            Wheter to translate the slab to the center the supercell. If set to False, the slab is traslated to the bottom of the supercell.

    Returns:
    --------
        traslated_slab : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure:
            A copy of the original slab, but translated according of the value of to_center parameter.
    """
    if to_center:
        frac_coords = surface.frac_coords
        center_of_mass_frac_coord = np.mean(frac_coords, axis=0)
        translation = np.array([0.5, 0.5, 0.5]) - center_of_mass_frac_coord
    else:
        z_translation = -min(surface.frac_coords[:, 2])
        translation = [0,0,z_translation]

    translated_slab = surface.copy()
    translated_slab.translate_sites(range(len(surface)), translation, frac_coords=True)

    # Correct sites with c=1
    if not to_center:
        for site in translated_slab:
            if site.c == 1:
                site.c = 0

    return translated_slab

def _remove_layers(surface, layers): # Internal use only
    """
    Creates a copy of a pymatgen.core.surface.Slab or pymatgen.core.structure.Structure object and remove some layers of it.

    Parameters:
    -----------
        surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            Pymatgen´s slab object.
        layers : list of sublists of int
            Layers to remove. The list contains sublists, each one corresponding to one of the layers to remove. Each sublist contains the indices of the sites of the corresponding layer.

    Returns:
    --------
        cut_surface: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            A copy of surface with the specified layers removed of it.
    """
    cut_surface = surface.copy()

    layers_to_remove = []
    for i in range(len(layers)):
        layers_to_remove += layers[i]

    # Remove sites that are in the layers
    index_to_delete = []
    for i in range(len(cut_surface.frac_coords)):
        if i in layers_to_remove:
            index_to_delete.append(i)
    cut_surface.remove_sites(index_to_delete)

    return cut_surface

def slab_truncator(surface, n, tolerance=0.85, center_final_slab=True):
    """
    Creates a truncated copy of a pymatgen´s slab to its first n layers.

    Parameters:
    -----------
        surface: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            Pymatgen´s slab object.
        n: int
            number of layers to truncate the slab.
        tolerance: float, default=0.85
            Tolerance in Anstrong used to determine the number of layers.
            In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.
        center_final_slab: bool, default=True
            Whether to center the truncated slab in the supercell. If set to False, the slab is placed in the bottom of the supercell.

    Returns:
    --------
        truncated_surface : pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            A copy of surface, but truncated to its first n layers.
    """
    layers = get_layers_indices(surface, tolerance=tolerance)
    if len(layers) > n:
        layers_to_remove = layers[n:]
        truncated_surface = _remove_layers(surface, layers_to_remove)
    else:
        print(f"WARNING! The slab has {len(layers)} layers. No truncation is apply.")
        truncated_surface = surface

    # Translate slab according the value of center_final_slab
    truncated_surface = translate_slab(truncated_surface, center_final_slab)

    return truncated_surface

def cut_surface_n_first_layers(surface, n, tolerance=0.85, center_final_slab=True):
    """
    Creates a copy of surface with it first layers cut. Useful to create new surfaces when there are few for a given chemical elements combination.

    Parameters:
    -----------
    surface : pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
        Surface to be cut.
    n : int
        Number of layers to cut.
    center_final_slab : bool, default=True
        Whether to center the truncated slab in the supercell. If set to False, the slab is placed in the bottom of the supercell.
    tolerance : float
        Tolerance in Anstrong used to determine the number of layers.
        In each iteration, the atom with the biggest z coordinate that wasn´t assign to a previous layer is identified. Each other atom that wasn´t assign to a previous layer whose distance along the z-axis to this atom is less than the tolerance is considered to be in the same layer.

    Returns:
    -----------
        cut_surface : pymatgen.core.structure.Structure | pymatgen.core.surface.Slab
            A copy of surface with its first n layers removed.
    """
    layers = get_layers_indices(surface, tolerance=tolerance)
    if len(layers) > n:
        layers_to_remove = layers[:n]
        cut_surface = _remove_layers(surface, layers_to_remove)
        # Translate slab according the value of center_final_slab
        cut_surface = translate_slab(cut_surface, center_final_slab)
    else:
        print(f"WARNING! The slab has {len(layers)} layers. No layers are removed. Returning None")
        cut_surface = None

    return cut_surface


def get_max_and_min_z_coord(structure):
    """
    Returns the maximun and minimun z coordinate of the sites in an structure.

    Parameters:
    -----------
        structure: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The structure of the slab.

    Returns:
    --------
        max_z, min_z: float
            Maximun and minimun z coordinate in Anstrong of the sites in an structure.
    """
    atomic_positions = structure.cart_coords
    z_coords = [pos[-1] for pos in atomic_positions]
    max_z = max(z_coords)
    min_z = min(z_coords)
    return float(max_z), float(min_z)

def get_slab_size(structure):
    """
    Returns the high of an slab.

    Parameters:
    -----------
        structure: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The structure of the slab.

    Returns:
    --------
        high: float
            The high of the slab in Anstrong.
    """
    max_z, min_z = get_max_and_min_z_coord(structure)
    return max_z - min_z

def slab_supercell_height(slab):
    """
    Get the high of the supercell.

    Parameters:
    -----------
        slab: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The structure of the slab.

    Returns:
    --------
        float, the high of the supercell.
    """
    return slab.lattice.c

def get_vac_size(slab):
    """
    Get the high of the vacuum in the supercell.

    Parameters:
    -----------
        slab: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The structure of the slab.

    Returns:
    --------
        float, the high of the vacuum.
    """
    supercell_high = slab_supercell_height(slab)
    slab_high = get_slab_size(slab)
    return supercell_high - slab_high

def modify_vac_size(slab, vac_size, center_slab=True):
    """
    Create a copy of an pymargen slab with modified high of vacuum.

    Parameters:
    -----------
        structure: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
            The structure of the slab.
        vac_size: float
            The new high of the vacuum in Anstrong.
        center_slab: bool
            Whether to center the slab in the supercell. If set to False, put the slab in the bottom of the supercell.

    Returns:
    --------
    modified_surface: pymatgen.core.surface.Slab | pymatgen.core.structure.Structure
        A copy of structure with a new vacuum size.
    """
    # Create a copy of surface and set the slab in the bottom of the supercell to avoid errors when reducing vacumm size too much
    modified_surface = translate_slab(slab, False)

    # Get size and cartesian coordinates of the original slab
    slab_size = get_slab_size(modified_surface)
    original_cart_sites = [site.coords for site in modified_surface.sites]

    # Determine the new size of the c vector
    new_c = slab_size+vac_size

    # Create a new lattice from the new c vector
    matrix = modified_surface.lattice.matrix.copy() # Warning! Note that modified_surface.lattice.matrix is exactly the same object as slab.lattice.matrix - no deep copy of the lattice's matrix attribute is made.
    matrix[2][-1] = new_c
    modified_surface.lattice = matrix

    # Get the new fractional coordinates to the original cartesian coordinates, in order to preserve them
    new_fractional_coords = [modified_surface.lattice.get_fractional_coords(cart_coord) for cart_coord in original_cart_sites]

    # Set the new fractional coordinates in the sites
    for i, site in enumerate(modified_surface):
        site.a = new_fractional_coords[i][0]
        site.b = new_fractional_coords[i][1]
        site.c = new_fractional_coords[i][2]

    # Translate slab
    modified_surface = translate_slab(modified_surface, center_slab)

    return modified_surface

# utils/in_out_management_utils.py

import os

def get_relaxed(path):
    """
    Retrieve the names of relaxed output files in the given directory.

    This function lists all files in the specified directory, filters for files with the suffix 'relax.out', and returns their names.

    Parameters:
    ----------
        path : str
            The directory path where the files are located.

    Returns:
    -------
        archivos_out : list of str
            A list containing the names of the files that match the 'relax.out' suffix.
    """
    # List all files in the specified directory
    archivos = os.listdir(path)

    # Filter the list to include only files (excluding directories)
    archivos_out = [archivo for archivo in archivos if os.path.isfile(os.path.join(path, archivo))]

    # Further filter to include only files ending with 'relax.out'
    archivos_out = [f'{path}/{archivo}' for archivo in archivos_out if archivo.endswith('relax.out')]

    return archivos_out

def extract_relaxed_positions_from_out(file_out):
    """
    Extract the positions of the relaxed atmos from an .out file from pw.x of QE.

    Parameters:
    ----------
    file_out: str
        The path of the .out file

    Returns:
    -------
    A list of strings of the form "{atom_specie}   {x_pos}     {y_pos}     {z_pos}".
    """
    with open(file_out, "r", encoding="utf-8", errors="ignore") as out_file:
        out_info = out_file.readlines()

    atomic_positions = []
    in_section = False

    for line in out_info:
        if "Begin final coordinates" in line:
            # Start capturing lines after this marker
            in_section = True
            continue
        if in_section:
            # Check if the line corresponds to an atom's position
            if line.strip() != "" and not "ATOMIC_POSITIONS" in line and not "End final coordinates" in line:
                cleaned_line = line.replace("0   0   0", "").strip()
                atomic_positions.append(cleaned_line)
            # Exit the section if the line doesn't contain an atom
            if "End final coordinates" in line:
                break
    if in_section == False:
        print(f"Convergence hasn´t been archieved, or an error ocurred. Please check {file_out}.")
        output = None
    else:
        output = atomic_positions

    return output

def redefine_atomic_positions_in(file_in, new_atomic_coordinates):
    """
    Take the data from an .in file of QE pw.x and redefine the atomic positions. It return a list with the modified lines of the file.

    Parameters:
    ----------
        file_in: string
            The path of the file to be modificated
        new_atomic_coordinates: list
            A list of strings of the form "{atomic_specie}      {x_position}    {y_position}        {z_position}", with the positions in Anstrong.

    Returns:
    -------
        model : list of str
            A list with the lines of the input file with modified atomic positions.
    """
    with open(file_in, 'r') as in_file:
        model = in_file.readlines()

    in_section = False
    indice = 0

    for i, line in enumerate(model):
        # Verify if we are in atomic positions section
        if "ATOMIC_POSITIONS (Angstrom)" in line:
            in_section = True
            continue
        # Change the line of model
        if in_section:
            if line.strip() != "":
                model[i] = new_atomic_coordinates[indice]+"\n"
                indice += 1
            else:
                break

    return model

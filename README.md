# Li_Adsorption_ML

![logo_python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![logo_pandas](https://img.shields.io/badge/Pandas-2.2.2-red.svg)
![logo numpy](https://img.shields.io/badge/Numpy-2.2.6-red.svg)
![logo matplñotlib](https://img.shields.io/badge/Matplotlib-30.10.3-red.svg)
![logo scipy](https://img.shields.io/badge/SciPy-1.14.1-red.svg)
![logo joblib](https://img.shields.io/badge/Joblib-1.4.2-red.svg)
![logo ase](https://img.shields.io/badge/ASE-3.25.0-red.svg)
![logo pymatgen](https://img.shields.io/badge/PyMatGen-orange.svg)
![logo licencia MIT](https://img.shields.io/badge/License-MIT-green.svg)

An automated workflow to generate a database of adsorption energies and a Graph Neural Network (GNN) model to predict them are presented. [PyMatGen](https://pymatgen.org/) tools are used to generate surfaces from [Materials Project](https://next-gen.materialsproject.org/materials)’s crystalline structures and to identify adsorption sites on them. The GNN model, based on Xie and Grossman's CGCNN model (2018)$^1$, is implemented in PyTorch and designed to run on GPUs.

This project aims to help to identify materials suitable for be used as anode current collectors in Anode-Free Lithium Batteries (AFLB). Pande and Viswanathan showed that the adsorption energy of Lithium (Li) on a metallic crystal's surface can be used as a descriptor of the material's performance as a anode current collector $^2$.

Using the automatic workflow and the trained model, a database of 1507 adsorption energies was created. The considered substrates are made of pure transition metals, Li, Mg, Ca, B, Al, Ga, Si, Sn, Ge and Pb, as well as their alloys with Li. The simulations were run at the [Centro de Cómputos de Alto Desempeño](https://supercomputo.unc.edu.ar) -High-Performance Computing Center- (CCAD) of the Universidad Nacional de Córdoba (UNC).

$^1$ T. Xie y J. C. Grossman. **“Crystal Graph Convolutional Neural Networks for
Accurate and Interpretable Prediction of Material Properties”**. En: Physical
Review Letters 120 (2018). doi: 10.1103/PhysRevLett.120.145301.

$^2$ V. Pande, V. Viswanathan. **“Computational Screening of Current Collectors for Enabling Anode-Free Lithium Metal Batteries”**. In: ACS Energy Letters 4.1 (2019), pp. 2952-2959. doi: 10.1021/acsenergylett.9b02306.

## Content overview

The notebook `run_adsorption_calc.ipynb` shows all the process of computing adsorption energies from Materials Project data. This notebook uses the `adsorption_calcs_utils` module, which contains all the required functions and is mainly based on [PyMatGen  Library](https://pymatgen.org/). You can read more details in [Workflow for computing adsorption energies section](#workflow-for-computing-adsorption-energies).

The `samples` directory contains some examples to try the notebook's workflow. Its content includes:
- The `cif` directory with `CIF` files of MP's crystals.
- The `pseudos` directory with the `UPF` pseudopotential files (provided by [PSLibrary](https://dalcorso.github.io/pslibrary/)) used in DFT calculations.
- `materials_project_DB.csv`, that contains MP's crystals's data.
- `potentials.csv`, which lists the available pseudopotentials in `pseudos` directory and data of the atomic species.
- The `ads_energies_results` directory, with the results of the calculations for each MP's material. Its subdirectories have names like `[reduced_chemical_formula]_[MP_id]`. Each one of these contains the `.in` files for `QE` DFT calculations and the resulting `.out` files for all slabs and slab+adsorbate systems.
- The `slabs` directory, that contains `joblib` files which store PyMatGen structure objects of the slabs generated from the materials.
- `ads_energies_db.csv`, that stores all the adsorption energies computed.
- `calcs_history.csv`, which works as a calculations record. You can see a description of its columns in [calcs_history's and final_calcs_history's columns description section](#calcs_historys-and-final_calcs_historys-columns-description).

The `model.in` is a sample of a `QE` `.in` file used by `adsorption_calcs_utils` to write the slabs and slab+adsorbate systems's `.in` files.

The `results` directory contains the results obtained so far in this project. It includes two files:
- `adsorption_sites_database.csv`, with the obtained adsorption energies database. You can see a description of its columns in [adsorption_sites_database's columns description section](adsorption_sites_databases-columns-description).
- `final_calcs_history.csv`, which is a record of all the calculations performed. In the [calcs_history's and final_calcs_history's columns description section](#calcs_historys-and-final_calcs_historys-columns-description) you can see more details about its columns meanings.

## Workflow for computing adsorption energies

The `run_adsorption_calc.ipynb` shows step by step the whole process. It has five main sections, each one corresponding to each main stage of the process:

1. The process starts with the selection of a material from MP database. For a material to be selected, its data must be loaded at `materials_project_DB` file and its `CIF` file must be in the `cif` directory. You can explore materials by their composition, and see their energy above hull to look for stable materials and the number of atoms per primitive cell.
2. Once you choose a material, in the second section you can create the slabs of the (1,0,0), (1,1,0) and (1,1,1) surfaces. The slabs are generated as PyMatGen's`Structure` objects and saved as `joblib` files in `samples/slabs` directory.
3. In the third stage, the `.in` files for the QE slabs relaxation calculations are created. The `.in` files are stored in `samples/ads_calc_results/[materials_pretty_formula]_[material_identifier_in_MP]`. Also, some surfaces's data is stored in `samples/calcs_history_csv` file, like the material's formula, identifier in MP and energy above hull; surface's Miller index, number of layers, number of atoms in a primitive cell, height, supercell parameters and whether it is stepped. While the slabs are being created, the supercell is shown to the user, and the user must indicate whether the surface is stepped or not. For this section of the notebook to work properly, it must be run in Jupyter Notebook or JupyterLab.
4. After performing the slabs's DFT calculations, the resulting `.out` files must be saved in the same folder as the `.in` files, i.e., in the corresponding subdirectory of `samples/ads_calc_results`. Then, in this stage the adsorption sites are identified and plotted. New `.in` files are created for computing relaxation calculations over each slab+adsorbate system. The calculations record file (`calcs_history.csv` in this case) is updated with the following information: whether the slabs's calculations performed successfully, the number of sites of each type and their sum.
5. After the DFT calculations over the slab+adsorbate systems are completed, and their `.out` files are saved in the material's subdirectory in `samples/ads_calc_results`, the final data is stored in the calculations record file (`calcs_history.csv`) and in the adsorption energies database (`ads_energies_db.csv`). In the first one, for each surface, is stored the number of computations for each type of adsorption site, the number of successfully performed computations, and how many calculations failed due to convergence issues, lack of computing time or errors.

In the las two sections of the `run_adsorption_calc.ipynb` notebook, you can consult the `calcs_history.csv` and `ads_energies_db.csv` databases (check the notebook for details).

## calcs_history's and final_calcs_history's columns description

In this section, the meaning of each column in `calcs_history.csv` and `final_calcs_history.csv` files is detailed. Note that each row correspondes to one surface and saves information about it and the adsorption energies calculations on its adsorption sites.

- **material**: Material's pretty formula.
- **material-id**: Material's identifier in MP.
- **e_hull**: Energy above hull.
- **indice_de_miller**: Surface´s Miller index.
- **capas**: Number of layers, counted by an own algorithm.
- **nsites**: Number of sites or atoms in the slab's supercell.
- **altura_slab**: Slab's height in Å.
- **a, b, c, alpha, beta, gamma**: Supercell's parameters.
- **escalonada**: Whether the surface is stepped.
- **exito_limpia**: Whether the slab's relaxation calculation was successfully completed.
- **sitios_bridge**: Number of bridge sites identified.
- **sitios_hollow**: Number of hollow sites identified.
- **sitios_ontop**: Number of on-top sites identified.
- **sitios_totales**: Total number of adsorption sites identified.
- **bridge_enviados**: How many relaxation calculations were performed on bridge sites. It counts how many `.out` files corresponding to the surface and a bridge site there are in the material's directory.
- **hollow_enviados**: How many relaxation calculations were performed on hollow sites. It counts how many `.out` files corresponding to the surface and a hollow site there are in the material's directory.
- **ontop_enviados**: How many relaxation calculations were performed on on-top sites. It counts how many `.out` files corresponding to the surface and a on-top site there are in the material's directory.
- **bridge_exitosos**: How many of the bridge adsorption energies were successfully performed.
- **hollow_exitosos**: How many of the hollow adsorption energies were successfully performed.
- **ontop_exitosos**: How many of the on-top adsorption energies were successfully performed.
- **prob_convergencia**: How many calculations failed due to convergence issues.
- **prob_tiempo**: How many calculations failed due to lack of computing time.
- **errores**: How many calculations failed due to errors. 

## adsorption_sites_database's columns description

In this section, the columns of the `adsorption_sites_database.csv` file are described:
- **surface**: Contains surfaces's identifiers, which have the form `[material_pretty_formula]_[material_MP_identifier]_[Miller_index]`.
- **adsorption_site**: Identifies each surface´s adsorption sites. The values have the form `[adsorption_site_type]_[adsorption_site_number]`
- **ads initial x**, **ads initial y**, **ads initial z**: Adsorbate's initial coordenates in the slab's supercell, before relaxation calculations.
- **ads final x**, **ads final y**, **ads final z**: Adsorbate's final coordenates in the slabś supercell, after relaxation calculations.
- **adsorptin energy**: Computed adsorption energy.

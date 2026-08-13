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

An automated workflow to generate a database of adsorption energies and a Graph Neural Network (GNN) model to predict them are presented. PyMatGen tools are used to generate surfaces from Materials Project’s crystalline structures and to identify adsorption sites on them. The GNN model, based on Xie and Grossman's CGCNN model (2018), is implemented in PyTorch and designed to run on GPUs.

This project aims to help to identify materials suitable for use as anode current collectors in Anode-Free Lithium Batteries (AFLB). Pande and Viswanathan showed that the adsorption energy of Lithium (Li) on a metallic crystal's surface can be used as a descriptor of the material's performance as a current collector.[^1]

Using the automatic workflow and the trained model, a database of 1507 adsorption energies was created. The considered substrates are made of pure transition metals, Li, Mg, Ca, B, Al, Ga, Si, Sn, Ge and Pb, as well as their alloys with Li.

## Content overview

The notebook `run_adsorption_calc.ipynb` shows all the process of computing adsorption energies from Materials Project data. This notebook uses the `adsorption_calcs_utils` module, which contains all the required functions.

The `samples` directory contains some examples to try the notebook's workflow. Its content includes:
- The `cif` directory with the `CIF` files of MP's crystals.
. The `pseudos` directory with the `UPF` pseudopotential files (provided by [PSLibrary](https://dalcorso.github.io/pslibrary/)) used in DFT calculations.
- `materials_project_DB.csv`, that contains MP's crystals's data.
- `potentials.csv`, which list the available pseudopotentials in `pseudos` directory and data of the atomic species.

The `results` folders saves the results of the adsorption energies computations. It contains:
- The `ads_energies_results` directory, with the results of the calculations for each MP's material. Its subdirectories have names `[reduced_chemical_formula]_[MP_id]`.Each one of these contains the `.in` files for `QE` DFT calculations and the resulting `.out` files for all slabs and slab+adsorbate systems.
- The `slabs` directory, which contains `joblib` files which store PyMatGen structure objects of the slabs generated from the materials.
- `ads_energies_db.csv`, that stores all the adsorption energies computed.
-`calcs_history`, which works as a calculations record.

Finally, the `model.in` is a sample of a `QE` `.in` file used by `adsorption_calcs_utils` to write the slabs and slab+adsorbate systems `.in` files.


:
1. In the first section, the materials database is filtered by chemical system.

The `ads_energies_calculations` directory contains the complete workflow for computing adsorption energies. It includes a jupyter notebook responsable for:
- Selecting materials
- Creating slabs from them
- Creating the `.in` files for Quantum ESPRESSO (QE) to relax the surfaces and compute the adsorption energies
- Identifying the adsorption sites on surfaces
- Extracting the results from the QE's `.out` files.

[^1] V. Pande, V. Viswanathan. **“Computational Screening of Current Collectors for Enabling Anode-Free Lithium Metal Batteries”**. In: ACS Energy Letters 4.1 (2019), pp. 2952-2959. doi: 10.1021/acsenergylett.9b02306.


# adsorption_calcs_utils

The `adsorption_calcs_utils` module contains all functions and utilities needed for the process of computing adsorption energies. This module is used in [`run_adsorption_calc.ipynb` notebook](../run_adsorption_calc.ipynb), which is a consumer of this module. For more details of the workflow, refer to the [README.md file in the root directory](../README.md).

**Status**: This module is under active development. The API may change.

# Dependencies

Direct:
- `pymatgen==2024.9.17.1` — structure handling and Materials Project API
- `ase>=3.25.0` — structure visualization
- `numpy>=2.2.6`, `scipy>=1.14.1` — numerical routines
- `pandas>=2.2.2` — tabular results
- `matplotlib>=3.10.3` — plotting
- `joblib>=1.4.2` — object saving and loading.



## Module structure

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public API exports; import all user-facing functions |
| `cif_utils.py` | Reads Materials Project crystals data, reads and parses crystal structures from CIF files |
| `diff_calculator_utils.py` | Defines the EnergyDiffCalculator class, which computes and export adsorption energies |
| `in_creation_utils.py` | Creates Quantum Espresso (QE) `.in` files |
| `in_out_management_utils.py` | Reads data from and modifies QE `.in` and `.out` files |
| `run_calculations_utils.py` | Functions responsible of each stage of the process of creating slabs from bulk structures given by CIF files and computing adsorption energies |
| `_run_helpers.py` | Auxiliary private functions for run_calculations_utils.py`'s functions |
| `structure_utils.py` | Functions for manipulating pymatgen Structure or Slab objects |

## Usage

To use the proposed workflow, you only need to import the `run_calculations_utils.py` submodule by:

```python
import adsorption_calcs_utils.run_calculations_utils
```

or:

```python
from adsorption_calcs_utils.run_calculations_utils import *
```

Of course, you can use any public object in the submodules `cif_utils.py`, `diff_calculator_utils.py`, `in_creation_utils.py`, `in_out_management_utils.py`, `run_calculations_utils.py` and `structure_utils.py`:
```python
from adsorption_calcs_utils.cif_utils import *
from adsorption_calcs_utils.structure_utils import translate_slab, get_max_and_min_z_coord
```

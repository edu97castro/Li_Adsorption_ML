# utils/diff_calculator_utils.py

import os
import re

import pandas as pd

class EnergyDiffCalculator:
    def __init__(self, paths):
        """
        Initialize the class with a list of paths to the files.

        Parameters
        ----------
        paths : list of str
            A list containing the paths to the relaxation output files.
        """
        self.paths = paths
        self.adsorption_paths = [p for p in paths if any(keyword in p for keyword in ['hollow', 'ontop', 'bridge'])]
        self.non_adsorption_paths = [p for p in paths if not any(keyword in p for keyword in ['hollow', 'ontop', 'bridge'])]

    def get_relax_energy(self, path):
        """
        Extract the final relaxation energy from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the relaxation output.

        Returns
        -------
        float or None
            The final relaxation energy if found, otherwise None.
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Check if the calculation was successfully achieved
        for line in reversed(lines):
            if 'Final energy' in line:
                match = re.search(r'-?\d+\.?\d*', line)
                if match:
                    return float(match.group())
        return None

    def get_energy_diff(self):
        """
        Calculate the energy differences between adsorption and non-adsorption files.

        Returns
        -------
        dict
            A dictionary with keys being the file base names and values being the energy differences.
        """
        energy_diffs = {}

        # Iterate over the adsorption paths
        for ads_path in self.adsorption_paths:
            # Extract the Miller index and compare it with non-adsorption files
            base_name = self._get_base_name(ads_path)
            non_ads_path = self._find_matching_non_adsorption(base_name)

            if non_ads_path:
                ads_energy = self.get_relax_energy(ads_path)
                non_ads_energy = self.get_relax_energy(non_ads_path)

                if ads_energy is not None and non_ads_energy is not None:
                    e_final = (ads_energy - non_ads_energy) + 29.69059736 / 2
                    energy_diffs[os.path.basename(ads_path)] = e_final * 13.6057

        return energy_diffs

    def _get_base_name(self, path):
        """
        Helper function to extract the base name of a file without hollow/ontop/bridge parts.

        Parameters
        ----------
        path : str
            The full path to the file.

        Returns
        -------
        str
            The base name of the file without the adsorption-specific part.
        """
        base_name = os.path.basename(path)
        return re.sub(r'_(hollow|ontop|bridge)(_|\d+)+_relax.out', '_relax.out', base_name)

    def _find_matching_non_adsorption(self, base_name):
        """
        Helper function to find a non-adsorption file matching a given base name.

        Parameters
        ----------
        base_name : str
            The base name to search for.

        Returns
        -------
        str or None
            The path to the matching non-adsorption file, or None if not found.
        """
        for non_ads_path in self.non_adsorption_paths:
            if base_name in non_ads_path:
                return non_ads_path
        return None

    def to_dataframe(self):
        """
        Generate a pandas DataFrame from the energy differences.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the adsorption paths as index and the energy differences as values.
        """
        energy_diffs = self.get_energy_diff()
        df = pd.DataFrame(energy_diffs.items(), columns=['Adsorption Path', 'Adsorption energy'])
        return df.sort_values(by='Adsorption Path')

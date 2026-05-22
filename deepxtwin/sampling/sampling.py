'''
 # @ Author: AzG
 # @ Create Time: 2025-07-04 16:03:47
 # @ Modified by: AzG
 # @ Modified time: 2025-12-15 13:24:09
 # @ Description: Sampling class that generates sets of design variables for parametric finite element (FE) analyses using predefined distributions.
 '''

# Import necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np
import chaospy as cp
import seaborn as sns
import matplotlib.pyplot as plt
from deepxtwin.utils import _read_config


# Define SamplingModel class
class SamplingModel:
    """Sampling class that generates and visualizes sets of design variables using predefined distributions."""

    def __init__(self, configpath="config.toml", n_samples=500):
        """
        This class generates samples for design variables based on predefined distributions specified in the top-level configuration file.  
        
        :param configpath: Path to the configuration file.
        :param n_samples: Number of samples to generate.
        TODO: In future work it may be interesting to also include parameters determining the inclinometer readings
        """

        self.config = _read_config(configpath).get("Sampling")
        self.n_samples = n_samples

        # Fetch design variable space for training surrogate model from config file
        self.L_col_min = self.config["L_col_min"]           # Depth of lime cement column panels
        self.L_col_max = self.config["L_col_max"]

        self.L_rib_min = self.config["L_rib_min"]           # Length of lime cement column panels
        self.L_rib_max = self.config["L_rib_max"]

        self.c_rib_min = self.config["c_rib_min"]           # Clear distance between lime cement column panels
        self.c_rib_max = self.config["c_rib_max"]

        self.su_LC_min = self.config["su_LC_min"]           # Undrained shear strength of lime cement column panels
        self.su_LC_max = self.config["su_LC_max"]

        self.su_clay_mean = self.config["su_clay_mean"]     # Undrained shear strength of clay formations
        self.su_clay_cov = self.config["su_clay_cov"]

        # Fetch constant parameter values from config file
        self.d_rib = self.config["d_rib"]
        self.E_LC = self.config["E_LC"]
        self.E_clay = self.config["E_clay"]
        self.GURsUARatio_clay_lower = self.config["GURsUARatio_clay_lower"]
        self.GURsUARatio_clay_upper = self.config["GURsUARatio_clay_upper"]
        self.gammaFC_clay_lower = self.config["gammaFC_clay_lower"]
        self.gammaFC_clay_upper = self.config["gammaFC_clay_upper"]

    def generate_joint_propability_distribution(self):
        """
        Generates the joint probability distribution of the design variable marginals.
        :return: Joint probability distribution of design variables.
        TODO: This method may be moved to the pce_surrogate_builder class in future work to ensure consistency between sampling and surrogate training.
        """

        # Set random seed for reproducibility
        np.random.seed(42)

        # Perform transformation of normal into lognormal distribution
        self.su_clay_std = self.su_clay_mean * self.su_clay_cov

        # Convert to lognormal parameters
        # Formulas obtained from Zhang et al. (2023): Geotechnical Reliability Analysis, p. 21
        su_clay_sigma = np.sqrt(np.log(1 + (self.su_clay_std / self.su_clay_mean) ** 2))
        su_clay_mu = np.log(self.su_clay_mean) - 0.5 * su_clay_sigma**2

        # Generate marginal distributions of design variables
        L_col_dist = cp.Uniform(self.L_col_min, self.L_col_max)
        L_rib_dist = cp.Uniform(self.L_rib_min, self.L_rib_max)
        c_rib_dist = cp.Uniform(self.c_rib_min, self.c_rib_max)
        su_LC_dist = cp.Uniform(self.su_LC_min, self.su_LC_max)
        su_clay_dist = cp.LogNormal(su_clay_mu, su_clay_sigma)

        # Generate joint distribution
        self.joint_dist = cp.J(
            L_col_dist, L_rib_dist, c_rib_dist, su_LC_dist, su_clay_dist
        )

        return self.joint_dist

    def generate_samples(self, filename="samples.csv", save_samples=False):
        """
        Generates samples for the design variables based on joint probability distribution and saves them to a CSV file.
        
        :param filename: Name of the output CSV file.
        :param save_samples: Whether to save samples to a CSV file.
        TODO: Generalize structure to allow for different sets of design variables.
        """

        # Generate joint probability distribution
        if not hasattr(self, "joint_dist"):
            self.generate_joint_propability_distribution()

        # Draw samples
        samples = np.round(
            self.joint_dist.sample(self.n_samples, rule="L").T, 2
        )  # shape: (n_samples, n_design_vars); Latin Hypercube sampling

        # Build rows
        header1 = (
            ["ribs"] * 8 + ["Leire1"] * 2 + ["Leire2"] * 2 + ["Leire1"] + ["Leire2"]
        )
        header2 = [
            "su_clay",
            "su_LC",
            "E_clay",
            "E_LC",
            "L_col",
            "L_rib",
            "c_rib",
            "d_rib",
            "GURsUARatio",
            "gammaFC",
            "GURsUARatio",
            "gammaFC",
            "sUARef",
            "sUARef",
        ]

        # Reorder columns to match header2
        data_rows = []
        for sample in samples:
            L_col, L_rib, c_rib, su_LC, su_clay = sample
            row = [
                su_clay,
                su_LC,
                self.E_clay,
                self.E_LC,
                L_col,
                L_rib,
                c_rib,
                self.d_rib,
                self.GURsUARatio_clay_lower,
                self.gammaFC_clay_lower,
                self.GURsUARatio_clay_upper,
                self.gammaFC_clay_upper,
                su_clay,
                su_clay,
            ]
            data_rows.append(row)

        self.samples_df = pd.DataFrame(data_rows, columns=header2)

        # Save to CSV file
        if save_samples:
            with open(filename, "w", newline="") as f:
                f.write(",".join(header1) + "\n")
                f.write(",".join(header2) + "\n")
                self.samples_df.to_csv(f, index=False, header=False)

    def visualize_samples(self, var_idx_list: list = [0, 1, 4, 5, 6]):
        """
        Visualizes the generated samples using a pairplot.
        
        :param var_idx_list: Indices of design variables to include in the visualization.
        :type var_idx_list: list
        """

        sns.set(style="whitegrid")
        g = sns.PairGrid(
            self.samples_df.iloc[:, var_idx_list]
        )  # Select design variables that are relevant for visualization

        # Upper triangle: scatter plots
        g.map_upper(sns.scatterplot)

        # Lower triangle: KDE density plots
        g.map_lower(sns.kdeplot, fill=True, cmap="Blues")

        # Diagonal: histograms
        g.map_diag(plt.hist, bins=20, edgecolor="k")

        # Title and layout
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(
            f"Scatter Matrix of Sampled Design Variables (n={self.n_samples})"
        )
        plt.show()

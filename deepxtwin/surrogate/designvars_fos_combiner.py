'''
 # @ Author: AzG
 # @ Create Time: 2025-07-25 10:33:20
 # @ Modified by: AzG
 # @ Modified time: 2025-12-15 15:06:12
 # @ Description: Transformation class that combines design variables and computed factors of safety (FoS) from different files into unified format.
 '''

# Import necessary libraries
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px


# Define Designvars_FoS_Combiner class
class Designvars_FoS_Combiner:
    """Class to collect FoS from different files, combine with design variables and generate interpretation plots.
    
    Note: 
    This class was necessary as the parametric finite element study had to be carried out in batches due to the 
    large number of samples that caused systematic errors in the automatic parametric model setup."""

    def __init__(self, designvars_folder_path:str, fos_parent_folder_path:str):
        """Initialize Designvars_FoS_Combiner instance.

        Args:
            designvars_folder_path (str): Path to folder containing design variable sets (CSV file).
            fos_parent_folder_path (str): Path to parent folder containing subfolders with FoS records.
        """
        self.designvars_folder_path = designvars_folder_path
        self.fos_parent_folder_path = fos_parent_folder_path  # parent folder including multiple subfolders with fos records (0000_0038, 0039_00xx, ...)

    def load_designvars(self, designvars_file_name: str = "inference.csv"):
        """Load design variable sets from CSV file"""
        print(self.designvars_folder_path)
        file_path = Path(self.designvars_folder_path) / designvars_file_name
        self.design_vars_df = pd.read_csv(file_path, header=[0, 1])

    def load_fos_single(self, fos_child_folder_name: str) -> pd.DataFrame:
        """Load FoS data from all samples in single folder"""
        fos_data = []

        # Construct full path to child folder
        fos_child_folder_path = (
            Path(self.fos_parent_folder_path) / fos_child_folder_name
        )

        # Iterate over all FoS files and collect data as list of sublists [[FoS1, FoS2], ...]
        for filename in os.listdir(fos_child_folder_path):
            if filename.startswith("FOS_") and filename.endswith(".csv"):
                file_path = os.path.join(fos_child_folder_path, filename)
                df = pd.read_csv(file_path)

                # Check if the file contains a "did not converge" message
                if (
                    "message" in df.columns
                    and "Phase did not converge" in df["message"].values
                ):
                    fos_data.append([np.nan, np.nan])

                elif "SumMsf" in df.columns and len(df) >= 2:
                    interim = df.iloc[0]["SumMsf"]
                    end = df.iloc[1]["SumMsf"]

                    # If the first value is below 1, set the second to np.nan
                    if interim < 1:
                        end = np.nan
                    fos_data.append([interim, end])

                # Handle unexpected file formats
                else:
                    raise ValueError(f"Unexpected format in file: {filename}")

        return pd.DataFrame(fos_data, columns=["FoS interim", "FoS end"])

    def load_fos_all(self, fos_parent_folder_path, debug=False) -> pd.DataFrame:
        """Load FoS data from all child folders under the parent folder"""

        fos_parent = Path(fos_parent_folder_path)

        if not fos_parent.exists():
            raise FileNotFoundError(f"Parent folder not found: {fos_parent}")

        fos_container_list = []

        # Iterate over all subfolders
        for child in fos_parent.iterdir():
            if child.is_dir():
                fos_container_list.append(
                    self.load_fos_single(child.name)
                )

        if not fos_container_list:
            raise ValueError("No FoS data found in any child folder")
        
        if debug:
            print(fos_container_list)

        # Combine FoS data into dataframe
        self.fos_df = pd.concat(fos_container_list, ignore_index=True)
        return self.fos_df

    def hstack_designvars_fos(self):
        """Hstack design_vars and computed fos"""

        # Flatten multi-index columns if present (e.g., from multiple header rows)
        if isinstance(self.design_vars_df.columns, pd.MultiIndex):
            self.design_vars_df.columns = [
                "_".join(map(str, col)).strip() for col in self.design_vars_df.columns
            ]

        # Merge only rows with matching index
        combined_df = pd.merge(
            self.design_vars_df,
            self.fos_df,
            left_index=True,
            right_index=True,
            how="inner",
        )

        self.combined_design_vars_fos_df = combined_df

    def export_data(self, filename: str = "designvars_fos_data.csv"):
        """Export combined data comprising design_vars and computed fos"""
        self.combined_design_vars_fos_df.to_csv(filename, index=False)

    def analyse_corr(self, debug=False):
        """Provide correlation matrix w.r.t. Pearson and Spearman rank coefficient, and matrix scatter plot"""
        combined_df = self.combined_design_vars_fos_df.copy()

        # Remove columns with only one unique value (ignores NaNs and constant features, such as pile diameter)
        combined_df = combined_df.loc[:, combined_df.nunique() > 1]

        # Compute correlation matrices
        pearson_corr = combined_df.corr(method="pearson")
        spearman_corr = combined_df.corr(method="spearman")

        # Plot Pearson correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Pearson Correlation Heatmap")
        plt.show()

        # Plot Spearman correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Spearman Correlation Heatmap")
        plt.show()

        # Label rows as converged or non-converged based on the criteria (FoS < 1 or np.nan)
        combined_df["convergence_status"] = (
            combined_df[["FoS interim", "FoS end"]].ge(1).all(axis=1)
        )

        # Map the boolean values to status labels
        combined_df["convergence_status"] = combined_df["convergence_status"].map(
            {True: "converged", False: "non-converged"}
        )

        if debug:
            print("Overview of non-converged design variable sets:\n")
            # Filter and display all rows where status is non-converged
            non_converged_df = combined_df[
                combined_df["convergence_status"] == "non-converged"
            ]

            # Display the result (all features of non-converged lines)
            print(non_converged_df)

            print("===============================================")

        # Remove temporary column
        features_to_plot = combined_df.drop(columns=["convergence_status"])

        # Add non-metric column for coloring only
        features_to_plot["convergence_status"] = combined_df["convergence_status"]

        # Plot matrix scatter plot with color based on convergence
        sns.pairplot(
            features_to_plot,
            hue="convergence_status",
            palette={"converged": "blue", "non-converged": "red"},
            diag_kind="hist",
            plot_kws={"alpha": 0.5},  # reduce opacity
        )
        plt.suptitle("Matrix Scatter Plot", y=1.02)
        plt.show()

        if debug:
            # Copy and filter the data
            hover_df = self.combined_design_vars_fos_df.copy()

            # Drop constant columns (only one unique value)
            hover_df = hover_df.loc[:, hover_df.nunique() > 1]

            # Label convergence based on existing logic
            hover_df["convergence_status"] = (
                hover_df[["FoS interim", "FoS end"]].ge(1).all(axis=1)
            )
            hover_df["convergence_status"] = hover_df["convergence_status"].map(
                {True: "converged", False: "non-converged"}
            )
            # Choose only numeric columns for the matrix
            numeric_cols = hover_df.select_dtypes(include="number").columns.tolist()

            # Create interactive scatter matrix
            fig = px.scatter_matrix(
                hover_df,
                dimensions=numeric_cols,
                color="convergence_status",
                title="Interactive Scatter Matrix (with Hover)",
                height=900,
                opacity=0.6,
            )

            # Optional: customize hover + layout
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            fig.update_layout(dragmode="select", hovermode="closest")

            fig.show()

    def plot_scatter_matrix(self, feature_names: list[str] = None):
        """Custom scatter matrix with:
        - upper triangle: colored scatter (converged vs failed),
        - diagonal: colored histograms,
        - lower triangle: blue KDE (all data combined, no scatter),
        - no legend.
        """
        # Copy and clean dataset
        df = self.combined_design_vars_fos_df.copy()
        df = df.loc[:, df.nunique() > 1]

        # Label convergence
        df["convergence_status"] = df[["FoS interim", "FoS end"]].notna().all(axis=1)
        df["convergence_status"] = df["convergence_status"].map(
            {True: "converged", False: "failed"}
        )

        # Select features only
        exclude_cols = {
            "FoS interim",
            "FoS end",
            "convergence_status",
            "Leire1_sUARef",
            "Leire2_sUARef",
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        plot_df = df[feature_cols + ["convergence_status"]].copy()

        # Apply custom feature names
        if feature_names is not None:
            assert len(feature_names) == len(feature_cols), (
                f"Expected {len(feature_cols)} feature names, got {len(feature_names)}."
            )
            rename_dict = dict(zip(feature_cols, feature_names))
            plot_df.rename(columns=rename_dict, inplace=True)
            feature_cols = feature_names

        # Base plot with upper scatter and diagonal histograms
        g = sns.pairplot(
            plot_df,
            hue="convergence_status",
            palette={"converged": "blue", "failed": "red"},
            diag_kind="hist",
            corner=False,
            plot_kws={"alpha": 0.6, "s": 10},
            diag_kws={"common_norm": False, "alpha": 0.6},
        )

        # Lower triangle: remove dots and add KDE (blue only)
        for i in range(1, len(feature_cols)):
            for j in range(i):
                ax = g.axes[i, j]
                ax.clear()  # Clear scatter
                # Use all points, regardless of convergence
                sns.kdeplot(
                    data=plot_df[feature_cols],  # drop 'convergence_status'
                    x=feature_cols[j],
                    y=feature_cols[i],
                    ax=ax,
                    fill=True,
                    cmap="Blues",
                    alpha=0.4,
                    levels=12,
                    thresh=0.05,
                )

        # Style axes
        for ax in g.axes.flatten():
            if ax:
                ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                ax.tick_params(labelsize=12)

        # Remove legend completely
        if g._legend:
            g._legend.remove()

        plt.show()

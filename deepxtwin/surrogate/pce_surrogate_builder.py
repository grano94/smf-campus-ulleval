'''
 # @ Author: AzG
 # @ Create Time: 2025-07-10 15:25:15
 # @ Modified by: AzG
 # @ Modified time: 2025-12-16 13:40:48
 # @ Description: This class builds and trains a Polynomial Chaos Expansion (PCE) surrogate model.
 '''

# Import necessary libraries
import numpy as np
import pandas as pd
import chaospy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class PCE_Surrogate_Builder:
    """Class to train/test PCE surrogate model.
    TODO: Extend to multiple target targets of the surrogate model."""

    def __init__(self, design_vars_fos_file_path: str, fos_col_idx: int = 14):
        """Initialize the PCE surrogate model and provide design variables and target = FoS
        TODO: As of now, the fos_col_idx is hardcoded."""
        self.design_vars_fos_file_path = design_vars_fos_file_path
        if design_vars_fos_file_path:
            self.load_rawdata(self.design_vars_fos_file_path)
        self.fos_col_idx = fos_col_idx

    def load_rawdata(self, design_vars_fos_file_name: str):
        """Load design variables and FoS data from CSV file"""
        self.data = pd.read_csv(design_vars_fos_file_name)

    def inspect_rawdata(self):
        """Inspect the loaded data and detect rows with NaN or missing values."""
        print("Data shape:", self.data.shape)
        print("Columns:")
        for idx, col_name in enumerate(self.data.columns):
            print(f"{idx}: {col_name}")

        # Rows with NaN / None
        has_nan = self.data.isna().any(axis=1)

        # Rows with empty or whitespace-only strings (object columns only)
        obj_cols = self.data.select_dtypes(include="object")
        has_empty = obj_cols.apply(
            lambda col: col.str.strip().eq("").fillna(False)
        ).any(axis=1)

        suspicious_rows = (has_nan | has_empty).sum()

        print(f"Number of rows with at least one NaN or missing value: {suspicious_rows}")
        print("First few rows:")
        print(self.data.head())

    def clean_rawdata(self):
        """Clean the data by removing rows with NaNs in specific FoS columns.
        TODO: As of now, only one FoS column is considered during the removal. To ensure that the same rows are removed
        when training multiple surrogates for different FoS, the union of all rows with NaNs may be removed."""
        # Get the column name corresponding to the FoS of interest
        fos_col = self.data.columns[self.fos_col_idx]

        # Drop only rows that have NaNs in FoS column of interest
        self.data_cleaned = self.data.dropna(subset=[fos_col])

        print(f"Shape of cleaned dataset: {self.data_cleaned.shape}")

    def train_test_split(
        self,
        design_vars_idx: list = [4, 5, 6, 1, 0],
        train_fraction: float = 0.8,
        n_bins: int = 5,
        plot_bin_distribution: bool = True,
        return_splits: bool = False,
    ):
        """Split data into training and test sets employing stratified sampling based on critical FoS
        TODO: Prevent errors by explicitly selecting the design variable columns by name."""
        
        # Extract design variables and FoS from cleaned data (after NaN removal)
        self.design_vars_cleaned = self.data_cleaned.iloc[:, design_vars_idx].to_numpy()
        self.fos_cleaned = self.data_cleaned.iloc[:, self.fos_col_idx].to_numpy()
        self.train_fraction = train_fraction

        # Train/test split considering stratified sampling
        stratify_bins = pd.qcut(
            self.data_cleaned.iloc[:, self.fos_col_idx], q=n_bins, labels=False
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.strat_train,
            self.strat_test,
        ) = train_test_split(
            self.design_vars_cleaned,
            self.fos_cleaned,
            stratify_bins,
            train_size=self.train_fraction,
            random_state=42,
            stratify=stratify_bins,
        )

        if plot_bin_distribution:
            bin_edges = np.histogram_bin_edges(self.y_train, bins=n_bins)

            # Format bin labels with ranges
            bin_labels = [
                f"{bin_edges[i]:.2f}–{bin_edges[i + 1]:.2f}"
                for i in range(len(bin_edges) - 1)
            ]

            # Visualize the distribution of bins
            train_bin_counts = pd.Series(self.strat_train).value_counts().sort_index()
            test_bin_counts = pd.Series(self.strat_test).value_counts().sort_index()

            plt.figure(figsize=(8, 5))
            bar_width = 0.35
            bins = range(n_bins)

            plt.bar(
                [b - bar_width / 2 for b in bins],
                train_bin_counts,
                width=bar_width,
                label="Train",
            )
            plt.bar(
                [b + bar_width / 2 for b in bins],
                test_bin_counts,
                width=bar_width,
                label="Test",
            )

            plt.xlabel("Stratification Bin (FoS Range)")
            plt.ylabel("Sample Count")
            plt.title("Distribution of Stratification Bins in Train and Test Sets")
            plt.xticks(bins, bin_labels, rotation=30)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

        if return_splits:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def train_pce_model(
        self,
        joint_dist,
        max_degree: int = 4,
        n_splits: int = 5,
        return_best_model: bool = False,
        plot_hyperparameter_study: bool = True,
        plot_preds_vs_true: bool = True,
    ):
        """Train a Polynomial Chaos Expansion (PCE) model and store all degrees."""

        assert self.X_train.shape[0] == self.y_train.shape[0], "Sample count mismatch"

        if not hasattr(self, "joint_dist") or self.joint_dist is None:
            self.joint_dist = joint_dist

        kf = KFold(n_splits, shuffle=True, random_state=42)
        mean_r2_scores = []
        self.models_by_degree = {}  # Store (poly_expansion, coeffs) for each degree

        # Hyperparameter study over polynomial degrees
        for degree in range(1, max_degree + 1):
            r2_scores = []
            mae_scores = []
            nrmse_scores = []

            poly_expansion = cp.expansion.stieltjes(degree, self.joint_dist)

            for train_idx, val_idx in kf.split(self.X_train):
                X_train_fold = self.X_train[train_idx, :]
                X_val_fold = self.X_train[val_idx, :]
                y_train_fold = self.y_train[train_idx]
                y_val_fold = self.y_train[val_idx]

                # NOTE: The same polynomial expansion associated with the joint distribution underlying the LHS is used
                coeffs = cp.fit_regression(poly_expansion, X_train_fold.T, y_train_fold)
                y_val_pred = coeffs(*X_val_fold.T)

                y_val_true = y_val_fold.flatten()
                y_val_pred = y_val_pred.flatten()

                r2_fold = r2_score(y_val_true, y_val_pred)
                mae_fold = mean_absolute_error(y_val_true, y_val_pred)
                nrmse_fold = np.sqrt(mean_squared_error(y_val_true, y_val_pred)) / (
                    np.max(y_val_true) - np.min(y_val_true)
                )

                r2_scores.append(r2_fold)
                mae_scores.append(mae_fold)
                nrmse_scores.append(nrmse_fold)

            mean_r2 = np.mean(r2_scores)
            mean_mae = np.mean(mae_scores)
            mean_nrmse = np.mean(nrmse_scores)

            mean_r2_scores.append(mean_r2)

            print(
                f"Degree {degree}: Mean CV R² = {mean_r2:.4f}, MAE = {mean_mae:.4f}, NRMSE = {mean_nrmse:.4f}"
            )

            final_coeffs = cp.fit_regression(
                poly_expansion, self.X_train.T, self.y_train
            )
            self.models_by_degree[degree] = (poly_expansion, final_coeffs)

        # Determine best degree from CV
        best_degree = np.argmax(mean_r2_scores) + 1
        self.best_degree = best_degree
        print(f"Best polynomial degree: {best_degree}")

        if plot_hyperparameter_study:
            plt.figure()
            plt.plot(range(1, max_degree + 1), mean_r2_scores, marker="o")
            plt.xlabel("Polynomial Degree")
            plt.ylabel("Mean CV R²")
            plt.title("Polynomial Degree Selection via Cross-Validation")
            plt.grid(True)
            plt.show()

        # Retrieve and store best model
        self.best_poly_expansion, self.best_coeffs = self.models_by_degree[best_degree]

        # Plot predictions vs true (train set)
        if plot_preds_vs_true:
            y_pred_best = self.best_coeffs(*self.X_train.T)
            y_true_best = self.y_train

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=y_true_best,
                    y=y_pred_best,
                    mode="markers",
                    marker=dict(size=8, opacity=0.7),
                    text=[f"{yt:.6f}" for yt in y_true_best],
                    hovertemplate="Predicted: %{y:.6f}<br>True: %{text}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[y_true_best.min(), y_true_best.max()],
                    y=[y_true_best.min(), y_true_best.max()],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    showlegend=False,
                )
            )

            fig.update_layout(
                title="PCE Predictions vs. True Labels (Train Set, Best Degree)",
                xaxis_title="True Labels",
                yaxis_title="PCE Predictions",
                xaxis=dict(range=[1, 4.0]),
                yaxis=dict(range=[1, 4.0]),
                template="simple_white",
            )

            fig.show()

        if return_best_model:
            return self.best_poly_expansion, self.best_coeffs

    def get_model_by_degree(self, degree: int):
        """Retrieve the trained model for a specific polynomial degree."""
        if not hasattr(self, "models_by_degree") or degree not in self.models_by_degree:
            raise ValueError(f"No model trained for degree {degree}")
        return self.models_by_degree[degree]

    def predict(self, X):
        """Predict the output using the best trained PCE model."""
        if not hasattr(self, "best_coeffs"):
            raise RuntimeError("PCE model not trained. Call train_pce_model first.")

        return self.best_coeffs(*X.T)

'''
 # @ Author: AzG
 # @ Create Time: 2025-07-25 11:15:41
 # @ Modified by: AzG
 # @ Modified time: 2025-12-16 13:42:18
 # @ Description: This class evaluates and compares multiple PCE surrogate models across sample sizes.
 '''

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from deepxtwin.surrogate import PCE_Surrogate_Builder
import chaospy as cp


class PCE_Surrogate_Analyzer:
    """Class to evaluate and compare multiple PCE surrogate models across sample sizes."""

    def __init__(self, pce_models: list): 
        """
        Args:
            pce_models (list): List of trained PCE_Surrogate_Builder instances (with different targets).
        """
        for model in pce_models:
            if not hasattr(model, "data_cleaned"):
                raise ValueError("All PCE models must have 'data_cleaned' attribute.")
        self.pce_models = pce_models

    def evaluate_pce_model(self, pce_idx=0, plot_limits: list = [1.0, 3.0], sm="E"):
        """
        Evaluate the PCE model on the test set and plot results.
        
        :param pce_idx: Index of the PCE model to evaluate
        :param plot_limits: Limits for the FoS plot axes
        :type plot_limits: list
        :param sm: Uppercase string for the surrogate model type (e.g., 'E' for excavation) 
        """

        model = self.pce_models[pce_idx]

        # Predict on test set
        y_pred = model.best_coeffs(*model.X_test.T)
        y_true = model.y_test

        r2 = r2_score(y_true, y_pred)
        nrmse = sqrt(mean_squared_error(y_true, y_pred)) / (
            np.max(y_true) - np.min(y_true)
        )
        mae = np.mean(np.abs(y_true - y_pred))

        print(f"Test R²: {r2:.4f}")
        print(f"Test NRMSE: {nrmse:.4f}")
        print(f"Test MAE: {mae:.4f}")

        # Plot: True vs Predicted (Test set)
        plt.figure(figsize=(5, 5), dpi=350)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(plot_limits, plot_limits, "--", color="black")
        plt.tick_params(labelsize=12)
        plt.xlim(plot_limits)
        plt.ylim(plot_limits)
        plt.xlabel(r"$FOS^{(FEA)}$ (1)", fontsize=12)
        plt.ylabel(rf"$FOS^{{({sm})}}$ (1)", fontsize=12)
        plt.grid(True)

        metrics_text = f"R² = {r2:.3f}\nNRMSE = {nrmse:.3f}\nMAE = {mae:.3f}"
        plt.gca().text(
            0.05,
            0.95,
            metrics_text,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
            verticalalignment="top",
            horizontalalignment="left",
            transform=plt.gca().transAxes,
        )
        plt.tight_layout()
        plt.show()

    def evaluate_sample_size_effect(
        self,
        design_vars_fos_file: str,
        max_degree: int = 4,
        plot_config: dict = None,
    ):
        """
        Evaluate model performance (NRMSE & R²) across varying sample sizes
        for all surrogate models. Training samples are drawn from the training set
        only, while evaluation is performed on the fixed test set.
        Each sample size is repeated several times and averaged.

        Args:
            design_vars_fos_file (str): Path to CSV file with design variables and FoS.
            max_degree (int): Maximum polynomial degree to consider in model training.
            plot_config (dict): Configuration dictionary with:
                {
                    "sample_sizes": [int, ...],
                    "r2_ylim": [float, float],
                    "nrmse_ylim": [float, float],
                    "labels": [str, ...],
                    "n_repeats": int  # Number of repetitions per sample size
                }
        """
        if plot_config is None:
            raise ValueError(
                "plot_config must be provided with 'sample_sizes' at minimum."
            )

        sample_sizes = plot_config.get("sample_sizes")
        if not sample_sizes:
            raise ValueError(
                "plot_config must contain a non-empty 'sample_sizes' list."
            )

        n_repeats = plot_config.get("n_repeats", 5)  # default: repeat 5 times
        r2_ylim = plot_config.get("r2_ylim", [0.0, 1.0])
        nrmse_ylim = plot_config.get("nrmse_ylim", [0.0, 0.35])
        model_labels = plot_config.get(
            "labels", [f"Model {i + 1}" for i in range(len(self.pce_models))]
        )

        all_r2_scores = []
        all_nrmse_scores = []

        for model_idx, base_model in enumerate(self.pce_models):
            print(
                f"\n== Evaluating surrogate {model_idx + 1} "
                f"(target col idx: {base_model.fos_col_idx}) =="
            )

            avg_r2_scores = []
            avg_nrmse_scores = []

            X_train_full = base_model.X_train
            y_train_full = base_model.y_train
            X_test_fixed = base_model.X_test
            y_test_fixed = base_model.y_test
            total_train_samples = X_train_full.shape[0]

            for n_samples in sample_sizes:
                if n_samples > total_train_samples:
                    print(
                        f"  Skipping sample size {n_samples}: "
                        f"not enough training data ({total_train_samples} available)"
                    )
                    continue

                r2_runs = []
                nrmse_runs = []

                for repeat in range(n_repeats):
                    """Repeat with different indices, but consistent across different sample sizes."""
                    rng = np.random.default_rng(seed=42 + repeat)
                    train_indices = rng.choice(
                        total_train_samples, size=n_samples, replace=False
                    )

                    X_train_sample = X_train_full[train_indices]
                    y_train_sample = y_train_full[train_indices]

                    tmp_model = PCE_Surrogate_Builder(
                        design_vars_fos_file,
                        fos_col_idx=base_model.fos_col_idx,
                    )
                    tmp_model.X_train, tmp_model.y_train = (
                        X_train_sample,
                        y_train_sample,
                    )

                    # Use same test dataset for performance assessment
                    tmp_model.X_test, tmp_model.y_test = X_test_fixed, y_test_fixed
                    tmp_model.joint_dist = base_model.joint_dist

                    tmp_model.train_pce_model(
                        joint_dist=tmp_model.joint_dist,
                        max_degree=max_degree,
                        plot_hyperparameter_study=False,
                        plot_preds_vs_true=False,
                    )

                    y_pred = tmp_model.best_coeffs(*X_test_fixed.T)
                    r2 = r2_score(y_test_fixed, y_pred)
                    nrmse = np.sqrt(mean_squared_error(y_test_fixed, y_pred)) / (
                        np.max(y_test_fixed) - np.min(y_test_fixed)
                    )

                    r2_runs.append(r2)
                    nrmse_runs.append(nrmse)

                mean_r2 = np.mean(r2_runs)
                mean_nrmse = np.mean(nrmse_runs)
                avg_r2_scores.append(mean_r2)
                avg_nrmse_scores.append(mean_nrmse)

                print(
                    f"  Sample size: {n_samples}, "
                    f"R²: {mean_r2:.4f} ± {np.std(r2_runs):.4f}, "
                    f"NRMSE: {mean_nrmse:.4f} ± {np.std(nrmse_runs):.4f}"
                )

            all_r2_scores.append(avg_r2_scores)
            all_nrmse_scores.append(avg_nrmse_scores)

        # Plot R²
        plt.figure(figsize=(5, 3), dpi=300)
        for idx, scores in enumerate(all_r2_scores):
            plt.plot(
                sample_sizes[: len(scores)],
                scores,
                marker="o",
                markersize=5,
                label=model_labels[idx],
            )
        plt.xlabel("Sample Size (1)", fontsize=12)
        plt.ylabel("R² Score (1)", fontsize=12)
        plt.tick_params(labelsize=12)
        plt.ylim(r2_ylim)
        plt.xlim(0, max(sample_sizes) + 50)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot NRMSE
        plt.figure(figsize=(5, 3), dpi=300)
        for idx, scores in enumerate(all_nrmse_scores):
            plt.plot(
                sample_sizes[: len(scores)],
                scores,
                marker="o",
                markersize=5,
                label=model_labels[idx],
            )
        plt.xlabel("Sample Size (1)", fontsize=12)
        plt.ylabel("NRMSE (1)", fontsize=12)
        plt.tick_params(labelsize=12)
        plt.ylim(nrmse_ylim)
        plt.xlim(0, max(sample_sizes) + 50)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate_global_sensitivity(
        self,
        plot_config: dict = None,
        return_indices: bool = False,
    ):
        """
        Plot main and total Sobol indices for each model side-by-side,
        sorted by descending maximum Sobol index per variable.

        Args:
            plot_config (dict): Optional dict with:
                {
                    "model_labels": list[str],  # Labels for each model
                    "var_labels": list[str],    # Labels for x-axis variables
                    "ylim": tuple[float, float] # y-axis limits
                }
            return_indices (bool): If True, return (main, total) Sobol indices for each model.
        """
        # Default values
        n_models = len(self.pce_models)
        default_model_labels = [f"Model {i + 1}" for i in range(n_models)]

        # Config setup
        labels = (
            plot_config.get("model_labels", default_model_labels)
            if plot_config
            else default_model_labels
        )
        ylim = plot_config.get("ylim", (0.0, 1.0)) if plot_config else (0.0, 1.0)

        sobol_main_all = []
        sobol_total_all = []

        for idx, model in enumerate(self.pce_models):
            if not hasattr(model, "best_coeffs") or not hasattr(model, "joint_dist"):
                raise ValueError(
                    f"Model {idx + 1} is missing 'best_coeffs' or 'joint_dist'"
                )

            main_sobol = cp.Sens_m(model.best_coeffs, model.joint_dist)
            total_sobol = cp.Sens_t(model.best_coeffs, model.joint_dist)

            print(f"\n== Model {idx + 1} (FoS col idx: {model.fos_col_idx}) ==")
            print("Main  Sobol:", np.round(main_sobol, 3))
            print("Total Sobol:", np.round(total_sobol, 3))

            sobol_main_all.append(main_sobol)
            sobol_total_all.append(total_sobol)

        sobol_main_all = np.array(sobol_main_all)  # shape: (n_models, n_vars)
        sobol_total_all = np.array(sobol_total_all)

        n_models, n_vars = sobol_main_all.shape

        # Use custom variable labels if provided
        if plot_config and "var_labels" in plot_config:
            var_labels = plot_config["var_labels"]
            if len(var_labels) != n_vars:
                raise ValueError(
                    "Length of 'var_labels' must match number of variables."
                )
        else:
            var_labels = [f"$X_{{{i + 1}}}$" for i in range(n_vars)]

        # Sort variables by descending max total Sobol index across models
        max_vals = np.max(np.vstack([sobol_main_all, sobol_total_all]), axis=0)
        sorted_indices = np.argsort(-max_vals)

        sobol_main_all = sobol_main_all[:, sorted_indices]
        sobol_total_all = sobol_total_all[:, sorted_indices]
        var_labels = [var_labels[i] for i in sorted_indices]

        bar_width = 0.8 / n_models
        x = np.arange(n_vars)

        # Main indices plot
        plt.figure(figsize=(5, 3), dpi=300)
        for m in range(n_models):
            plt.bar(
                x + m * bar_width - (bar_width * (n_models - 1) / 2),
                sobol_main_all[m],
                width=bar_width,
                label=labels[m],
                zorder=2,
            )
        plt.xticks(x, var_labels, fontsize=12)
        plt.ylim(*ylim)
        plt.ylabel("Main Sobol' Indices", fontsize=12)
        plt.grid(axis="y", zorder=1)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Total indices plot
        plt.figure(figsize=(5, 3), dpi=300)
        for m in range(n_models):
            plt.bar(
                x + m * bar_width - (bar_width * (n_models - 1) / 2),
                sobol_total_all[m],
                width=bar_width,
                label=labels[m],
                zorder=2,
            )
        plt.xticks(x, var_labels, fontsize=12)
        plt.ylim(*ylim)
        plt.ylabel("Total Sobol' Indices", fontsize=12)
        plt.grid(axis="y", zorder=1)
        plt.tight_layout()
        plt.show()

        if return_indices:
            return [(sobol_main_all[i], sobol_total_all[i]) for i in range(n_models)]

    def evaluate_univariate_effects(self, plot_config: dict = None):
        """
        Plot univariate effects of each design variable on the target using all PCE surrogate models.
        Each subplot shows one design variable being varied, others held constant.

        All models are plotted together in each subplot.
        Layout: 2 rows × 3 columns (supports up to 6 design variables).

        Args:
            plot_config (dict): Optional plotting config:
                {
                    "n_points": int,
                    "x_ranges": list[tuple[float, float]],   # One (min, max) per variable
                    "var_labels": list[str],                 # Optional variable names
                    "constant_values": list[float],          # Optional constants for non-varied vars
                    "constant_strategy": str,                # 'mean' (default) or 'median'
                    "y_lim": tuple[float, float]             # Optional (min, max) for shared y-axis
                    "labels": list[str]                      # Optional model labels for legend
                    "last_subplot_points_2d": list[tuple[float, float, int]] # Optional (x, y, model_idx)
                    "last_subplot_points_3d": list[tuple[float, float, int]] # Optional (x, y, model_idx)
                    "legend_loc": str                        # Optional legend location (default: 'lower right')
                }
        """
        if plot_config is None:
            plot_config = {}

        n_points = plot_config.get("n_points", 50)
        constant_strategy = plot_config.get("constant_strategy", "mean")
        y_lim = plot_config.get("y_lim", None)

        # Use first model to infer n_vars
        if len(self.pce_models) == 0:
            raise ValueError("No PCE models provided.")

        n_vars = self.pce_models[0].X_train.shape[1]

        # Constant values
        if "constant_values" in plot_config:
            constant_values = np.array(plot_config["constant_values"])
            if len(constant_values) != n_vars:
                raise ValueError(
                    "Length of 'constant_values' must match number of design variables."
                )
        else:
            ref_model = self.pce_models[0]
            if constant_strategy == "median":
                constant_values = np.median(ref_model.X_train, axis=0)
            else:
                constant_values = np.mean(ref_model.X_train, axis=0)

        # Ranges
        if "x_ranges" in plot_config:
            x_ranges = plot_config["x_ranges"]
            if len(x_ranges) != n_vars:
                raise ValueError(
                    "Length of 'x_ranges' must match number of design variables."
                )
        else:
            x_min = np.min(ref_model.X_train, axis=0)
            x_max = np.max(ref_model.X_train, axis=0)
            x_ranges = [
                (l - 0.05 * abs(h - l), h + 0.05 * abs(h - l))
                for l, h in zip(x_min, x_max)
            ]

        # Labels
        var_labels = plot_config.get(
            "var_labels", [f"$X_{{{i + 1}}}$" for i in range(n_vars)]
        )
        legend_loc = plot_config.get("legend_loc", "lower right")
        points_2d = plot_config.get("last_subplot_points_2d", [])
        points_3d = plot_config.get("last_subplot_points_3d", [])

        # Plotting layout
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), dpi=300, sharey=True)
        axes = axes.flatten()
        colors = plt.cm.tab10.colors

        # Define labels for models
        if "labels" in plot_config:
            model_labels = plot_config["labels"]
        else:
            # Default model labels if not provided
            model_labels = [f"Model {i + 1}" for i in range(len(self.pce_models))]

        for i in range(min(n_vars, n_rows * n_cols)):
            x_vals = np.linspace(*x_ranges[i], n_points)

            for m_idx, model in enumerate(self.pce_models):
                X_input = np.tile(constant_values, (n_points, 1))
                X_input[:, i] = x_vals
                y_preds = model.best_coeffs(*X_input.T)
                axes[i].plot(
                    x_vals,
                    y_preds,
                    label=model_labels[m_idx],
                    color=colors[m_idx % len(colors)],
                )

            axes[i].set_xlabel(var_labels[i], fontsize=12)
            if i % n_cols == 0:
                axes[i].set_ylabel("FOS", fontsize=12)

            if y_lim is not None:
                axes[i].set_ylim(y_lim[0], y_lim[1])
                axes[i].set_autoscale_on(False)
                axes[i].set_yticks(np.linspace(y_lim[0], y_lim[1], num=6))


            # Plot site-specific values as vertical lines
            ymin, ymax = axes[i].get_ylim()

            axes[i].vlines(
                constant_values[i],
                ymin=ymin,
                ymax=ymax,
                color="black",
                linestyle="--",
                linewidth=1.0,
                )

            axes[i].tick_params(labelsize=12)
            axes[i].grid(True)


        # Overlay optional SRFEA reference points in the last active subplot only.
        last_active_idx = min(n_vars, n_rows * n_cols) - 1
        for p_idx, (x_pt, y_pt, model_idx) in enumerate(points_2d):
            if 0 <= model_idx < len(self.pce_models):
                axes[last_active_idx].scatter(
                    x_pt,
                    y_pt,
                    marker="o",
                    s=36,
                    color=colors[model_idx % len(colors)],
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"2D SRFEA ({model_labels[model_idx]})" if p_idx == model_idx else None,
                    zorder=4,
                )

        for p_idx, (x_pt, y_pt, model_idx) in enumerate(points_3d):
            if 0 <= model_idx < len(self.pce_models):
                axes[last_active_idx].scatter(
                    x_pt,
                    y_pt,
                    marker="D",
                    s=42,
                    color=colors[model_idx % len(colors)],
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"3D SRFEA ({model_labels[model_idx]})" if p_idx == model_idx else None,
                    zorder=5,
                )

        # Hide all unused subplots.
        total_axes = n_rows * n_cols
        for j in range(n_vars, total_axes):
            axes[j].axis("off")

        # Build proxy handles so the legend is complete on any axis.
        legend_handles = []
        for m_idx, lbl in enumerate(model_labels):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[m_idx % len(colors)],
                    linewidth=1.5,
                    label=lbl,
                )
            )

        legend_handles.append(
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.0, label="site-specific value"
            )
        )

        if len(points_2d) > 0:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=6,
                    label="2D SRFEA",
                )
            )

        if len(points_3d) > 0:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="None",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=6,
                    label="3D SRFEA",
                )
            )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Place legend inside the empty bottom-right panel (figure-level),
        # or fallback to the last active subplot when no panel is empty.
        if n_vars < total_axes:
            empty_idx = total_axes - 1
            bbox = axes[empty_idx].get_position()
            fig.legend(
                handles=legend_handles,
                fontsize=12,
                loc=legend_loc,
                bbox_to_anchor=(bbox.x0, bbox.y0, bbox.width, bbox.height),
                bbox_transform=fig.transFigure,
            )
        else:
            axes[last_active_idx].legend(handles=legend_handles, fontsize=10, loc=legend_loc)

        plt.show()

    def evaluate_response_surface(
        self,
        pce_idx: int,
        var1_idx: int,
        var2_idx: int,
        degree: int = None,
        plot_config: dict = None,
        zlim: [float, float] = [0.9, 3.5],
    ):
        """
        Plot 3D response surface for a selected PCE model over two input variables,
        with all other variables fixed at values provided in plot_config.
        """
        if plot_config is None:
            raise ValueError("plot_config must be provided")

        # Extract selected model
        model = self.pce_models[pce_idx]

        if degree is None:
            coeffs = model.best_coeffs
            X_train = model.X_train
            label_suffix = f"(Best Degree = {model.best_degree})"
        else:
            if (
                not hasattr(model, "models_by_degree")
                or degree not in model.models_by_degree
            ):
                raise ValueError(
                    f"No model of degree {degree} found for PCE index {pce_idx}"
                )
            _, coeffs = model.models_by_degree[degree]
            X_train = model.X_train
            label_suffix = f"(Degree = {degree})"

        n_vars = X_train.shape[1]

        # Read plot config 
        n_points = plot_config.get("n_points", 50)
        x_ranges = plot_config.get("x_ranges", None)
        const_vals = plot_config.get("constant_values", None)
        var_labels = plot_config.get("var_labels", [f"X{i + 1}" for i in range(n_vars)])
        output_label = plot_config.get("output_label", r"$FOS^{(E)}$ (1)")
        plot_labels = plot_config.get(
            "plot_labels", [var_labels[var1_idx], var_labels[var2_idx]]
        )
        show_train = plot_config.get("show_training_points", True)
        show_test = plot_config.get("show_test_points", True)
        zlim = plot_config.get("zlim", [0.9, 3.5])

        # Validate config 
        if x_ranges is None or const_vals is None:
            raise ValueError(
                "plot_config must contain both 'x_ranges' and 'constant_values'"
            )
        if len(x_ranges) != n_vars or len(const_vals) != n_vars:
            raise ValueError(
                "Length of 'x_ranges' and 'constant_values' must match number of variables"
            )

        # Generate grid for var1 and var2 
        x1_vals = np.linspace(*x_ranges[var1_idx], n_points)
        x2_vals = np.linspace(*x_ranges[var2_idx], n_points)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        # Build input matrix with fixed constants, override var1 and var2 
        X_pred = np.tile(const_vals, (n_points * n_points, 1))
        print(X_pred)
        X_pred[:, var1_idx] = X1.ravel()
        X_pred[:, var2_idx] = X2.ravel()

        # Predict on grid 
        y_pred = coeffs(*X_pred.T).reshape((n_points, n_points))

        # Plot 3D surface 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X1,
            X2,
            y_pred,
            cmap="viridis",
            alpha=0.2,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            vmin=zlim[0],
            vmax=zlim[1],
        )

        # Overlay training predictions 
        if show_train:
            x1_train = model.X_train[:, var1_idx]
            x2_train = model.X_train[:, var2_idx]
            y_train_pred = model.best_coeffs(*model.X_train.T)
            min_idx = np.argmin(x2_train)
            print(f"Index of min train set value of var2_idx: {min_idx}")
            print(f"y_test_pred at min var2_idx: {y_train_pred[min_idx]}")

            ax.scatter(
                x1_train,
                x2_train,
                y_train_pred,
                color="black",
                s=15,
                label=f"Surrogate model prediction at ({plot_labels[0]}, {plot_labels[1]}) of training set",
                alpha=0.7,
            )

        # Overlay test predictions 
        if show_test:
            x1_test = model.X_test[:, var1_idx]
            x2_test = model.X_test[:, var2_idx]
            y_test_pred = model.best_coeffs(*model.X_test.T)
            min_idx = np.argmin(x2_test)
            print(f"Index of min test set value of var2_idx: {min_idx}")
            print(f"y_test_pred at min var2_idx: {y_test_pred[min_idx]}")

            ax.scatter(
                x1_test,
                x2_test,
                y_test_pred,
                color="red",
                s=15,
                label=f"Surrogate model prediction at ({plot_labels[0]}, {plot_labels[1]}) of test set",
                alpha=0.7,
            )

        # Final plot formatting 
        ax.set_xlabel(var_labels[var1_idx], fontsize=14)
        ax.set_ylabel(var_labels[var2_idx], fontsize=14)
        ax.set_zlim(zlim)

        # Increase label padding
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10

        # Optional manual ticks from plot_config
        x_ticks = plot_config.get("x_ticks", None)
        y_ticks = plot_config.get("y_ticks", None)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        # Set font size for tick labels
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
        for label in ax.get_zticklabels():
            label.set_fontsize(14)

        cbar = fig.colorbar(surf, shrink=0.5, aspect=12, label=output_label)
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=14)

        if show_train or show_test:
            ax.legend(
                fontsize=14,
                loc="lower left",
                bbox_to_anchor=(0.0, -0.20),
                frameon=False,
            )

        plt.tight_layout()
        plt.show()

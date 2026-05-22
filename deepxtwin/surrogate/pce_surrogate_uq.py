'''
 # @ Author: AzG
 # @ Create Time: 2025-07-29 12:42:53
 # @ Modified by: AzG
 # @ Modified time: 2025-12-16 15:10:10
 # @ Description: Classes for UQ analysis using PCE surrogate models.
 '''

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import pandas as pd
import chaospy as cp


class InputDistributionFactory:
    """Stores input marginals for MC sampling: 1 lognormal + 4 fixed variables using chaospy."""

    def __init__(self, mean, cov, fixed_values: list):
        self.mean = mean
        self.cov = cov
        self.fixed_values = fixed_values
        self.var_labels = [
            "$d_{LC}$ (m)",
            "$L_{LC}$ (m)",
            "$c_{LC}$ (m)",
            "$s_{u,LC}$ (kPa)",
            "$s_{u,clay}$ (kPa)",
        ]

    def sample(self, nsamples:int, debug=False):
        """MC sampling class.

        Args:
            nsamples (int): MC samples to generate
            debug (bool, optional): Statistical validation of generated MC samples. Defaults to False.

        Returns:
            np.array: samples array
        """
        # Set random seed for reproducibility
        np.random.seed(42)

        # Convert to lognormal parameters (Zhang et al. 2023)
        std = self.mean * self.cov
        sigma_ln = np.sqrt(np.log(1 + (std / self.mean) ** 2))
        mu_ln = np.log(self.mean) - 0.5 * sigma_ln**2

        # Build the chaospy lognormal distribution
        lognormal_dist = cp.LogNormal(mu_ln, sigma_ln)

        # Sample from the lognormal (random = monte carlo sampling)
        lognormal_samples = lognormal_dist.sample(nsamples, rule="random").reshape(
            -1, 1
        )

        # Build the fixed array
        fixed_array = np.array(self.fixed_values).reshape(1, -1)
        fixed_samples = np.repeat(fixed_array, nsamples, axis=0)

        # Combine fixed + variable
        samples = np.hstack((fixed_samples, lognormal_samples))

        if debug:
            df = pd.DataFrame(samples, columns=self.var_labels)
            print("Sample Summary:")
            print(df.describe())

        return samples


class UQRunner:
    """Performs Monte Carlo sampling of design variables and computes FoS predictions, beta and pf."""

    def __init__(self, pce_model, fs_limit=None):
        self.model = pce_model
        self.fs_limit = fs_limit
        self.outputs = None

    def run(self, X):
        self.outputs = self.model.predict(X)
        return self.outputs

    def compute_pf_beta(self):
        if self.fs_limit is None:
            raise ValueError("fs_limit must be provided to compute Pf and beta.")

        log_outputs = np.log(self.outputs)
        log_fs_limit = np.log(self.fs_limit)
        g = log_outputs - log_fs_limit
        pf = np.mean(
            g < 0
        )  # NOTE: To account for lognormal distribution in calculation of reliability index lognormal vals are used in limit state function
        beta = -norm.ppf(pf) if pf > 0 else np.inf
        return pf, beta


class UQAnalysis:
    """Runs UQ workflows (mean/CI, KDE, Pf/Beta)
    NOTE: Assumes only one variable is uncertain (lognormal), rest are fixed."""

    def __init__(self, pce_model, mean_input, fixed_inputs):
        self.model = pce_model
        self.mean_input = mean_input
        self.fixed_inputs = fixed_inputs

    def analyze_mean_ci_vs_samples(
        self, cov=0.3, sample_sizes=[100, 1000, 5000, 10000], debug=False
    ):
        """Evaluate influence of samples size on mean and 95% confidence interval of FoS predictions."""
        stats_list = []  # to collect mean and CI for each sample size

        for n in sample_sizes:
            X = InputDistributionFactory(
                self.mean_input, cov, self.fixed_inputs
            ).sample(n)

            if debug:
                X_df = pd.DataFrame(
                    X,
                    columns=[
                        "d_LC (m)",
                        "L_LC (m)",
                        "c_LC (m)",
                        "s_u_LC (kPa)",
                        "s_u_clay (kPa)",
                    ],
                )
                print(f"\nInput Sample Summary (n={n}):")
                print(X_df.describe())

            # Generate PCE predictions and sample stats
            runner = UQRunner(self.model)
            y = runner.run(X)

            mu = y.mean()
            std = y.std(ddof=1)
            ci95 = (
                1.96 * std / np.sqrt(n)
            )  # NOTE: Z-value for confidence niveau 95 %, valid for mean despite lognormal stribution of one design variable (not valid for median, percentiles though)
            p5 = np.percentile(y, 5)

            stats_list.append(
                {
                    "Sample Size": n,
                    "Mean": mu,
                    "95% CI Lower": mu - ci95,
                    "95% CI Upper": mu + ci95,
                    "5% Percentile": p5,
                }
            )

        summary_df = pd.DataFrame(stats_list)
        print("\n=== Mean and 95% Confidence Interval Summary ===")
        print(summary_df.to_string(index=False))

        # Plot mean and 95% CI vs sample size
        print("\n=== Plotting ===")
        plt.figure()
        plt.fill_between(
            summary_df["Sample Size"],
            summary_df["95% CI Lower"],
            summary_df["95% CI Upper"],
            alpha=0.3,
            label="95% CI",
        )
        plt.plot(
            summary_df["Sample Size"], summary_df["Mean"], marker="o", label="Mean"
        )
        plt.xlabel("Sample size")
        plt.ylabel("Output mean ± 95% CI")
        plt.title("Mean and 95% CI vs. Sample Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_kde_vs_cov(
        self, cov_list=[0.1, 0.15, 0.2, 0.25, 0.3], nsamples=10000, debug=False
    ):
        """Evaluate influence of input COV on output FoS distribution using KDE plots."""
        stats_list = []

        plt.figure(figsize=(10, 5))
        for cov in cov_list:
            X = InputDistributionFactory(
                self.mean_input, cov, self.fixed_inputs
            ).sample(nsamples)

            if debug:
                X_df = pd.DataFrame(
                    X,
                    columns=[
                        "d_LC (m)",
                        "L_LC (m)",
                        "c_LC (m)",
                        "s_u_LC (kPa)",
                        "s_u_clay (kPa)",
                    ],
                )
                print(f"\nInput Sample Summary (COV={cov:.2f}):")
                print(X_df.describe())

            runner = UQRunner(self.model)
            y = runner.run(X)

            mu = y.mean()
            std = y.std(ddof=1)
            ci95 = 1.96 * std / np.sqrt(nsamples)
            p5 = np.percentile(y, 5)

            stats_list.append(
                {
                    "COV": cov,
                    "Sample Size": nsamples,
                    "Mean": mu,
                    "95% CI Lower": mu - ci95,
                    "95% CI Upper": mu + ci95,
                    "5% Percentile": p5,
                }
            )

            kde = gaussian_kde(y)
            x_vals = np.linspace(min(y), max(y), 10000)
            plt.plot(x_vals, kde(x_vals), label=f"COV = {cov:.2f}")

        # Print summary table of means and 95% CIs
        summary_df = pd.DataFrame(stats_list)
        print("\n=== Mean and 95% Confidence Interval Summary by COV ===")
        print(summary_df.to_string(index=False, float_format="%.4f"))

        print("\n=== Plotting ===")
        plt.xlabel(r"$FOS^{(E)}$ (1)", fontsize=16)
        plt.ylabel("PDF (1)", fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([0, 4.0])
        plt.xlim([0, 3.5])
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def analyze_pf_beta_vs_fs_cov(
        self,
        cov_list=[0.1, 0.15, 0.2, 0.25, 0.3],
        fs_limits=[1.2, 1.4, 1.6, 1.8],
        nsamples=10000,
    ):
        """Evaluate influence of COV of design variable and FoS limit criterion on Pf and beta."""
        pf_matrix = np.zeros((len(cov_list), len(fs_limits)))
        beta_matrix = np.zeros_like(pf_matrix)
        results = []

        for i, cov in enumerate(cov_list):
            X = InputDistributionFactory(
                self.mean_input, cov, self.fixed_inputs
            ).sample(nsamples)
            for j, fs in enumerate(fs_limits):
                runner = UQRunner(self.model, fs_limit=fs)
                runner.run(X)
                pf, beta = runner.compute_pf_beta()
                pf_matrix[i, j] = pf
                beta_matrix[i, j] = beta
                results.append(
                    {
                        "COV": cov,
                        "FS Limit": fs,
                        "Sample Size": nsamples,
                        "P_f": pf,
                        "β": beta,
                    }
                )

        # Convert to DataFrame for pretty printing
        result_df = pd.DataFrame(results)
        print("\n=== Probability of Failure and Reliability Index Summary ===")
        print(result_df.to_string(index=False))

        print("\n=== Plotting ===")
        # Plotting P_f vs FS
        plt.figure(figsize=(10, 5))
        for i, cov in enumerate(cov_list):
            plt.plot(fs_limits, pf_matrix[i], marker="o", label=f"COV = {cov:.2f}")
        plt.xlabel("FS Limit", fontsize=12)
        plt.ylabel("Probability of Failure (P_f)", fontsize=12)
        plt.yscale("log")
        plt.title("Failure Probability vs. FS Limit", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plotting β vs FS
        plt.figure(figsize=(10, 5))
        for i, cov in enumerate(cov_list):
            plt.plot(fs_limits, beta_matrix[i], marker="s", label=f"COV = {cov:.2f}")
        plt.xlabel("FS Limit", fontsize=14)
        plt.ylabel("Reliability Index (β)", fontsize=14)
        plt.title("Reliability Index vs. FS Limit", fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

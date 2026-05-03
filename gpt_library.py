import arviz as az
import pandas as pd
import numpy as np
import global_config as cfg
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
from model_design import GPTConfig


class GPTLibrary:
    def __init__(self, config_dict=None, epochs=1):
        """initializes the gpt library to track model metrics and time series data."""
        self.summary_df = pd.DataFrame()

        if config_dict is not None:
            # handle single gptconfig objects
            if isinstance(config_dict, GPTConfig):
                config_dict = {config_dict.name: config_dict}

            # convert config objects to dicts
            rows = [cfg.to_dict() for cfg in config_dict.values()]

            self.summary_df = pd.DataFrame(rows)
            self.summary_df['epochs'] = epochs

            if 'name' in self.summary_df.columns:
                self.summary_df.rename(columns={'name': 'Model'}, inplace=True)
                self.summary_df.set_index('Model', inplace=True)

        self.time_series_df = pd.DataFrame(columns=['val_curve', 'loss_curve', 'tok_by_step'])
        self.time_series_df.index.name = 'Model'

    def log_summary(self, model_name, column_name, val):
        """
        safely updates or adds new metrics.
        creates a new row if the model doesn't exist.
        """

        self.summary_df.at[model_name, column_name] = val

    def log_series(self, model, stats, val_results=None):
        """
        explodes stats into the time series dataframe.
        pads validation results to match the loss curve length.
        """
        steps = len(stats['loss_curve'])

        padded_val = [None] * steps
        val_results = val_results.tolist()

        # pad validation data to match step count
        if val_results is not None:
            if isinstance(val_results, (float, int)):
                padded_val[-1] = val_results
            else:
                n = min(len(val_results), steps)
                for i in range(n):
                    padded_val[i] = val_results[i]

        new_series = pd.DataFrame({
            'loss_curve': stats['loss_curve'],
            'tok_by_step': stats['tokens_by_step'],
            'val_curve': padded_val
        })

        new_series.index = [model] * steps
        new_series.index.name = 'model'

        self.time_series_df = pd.concat([self.time_series_df, new_series])

    def get_df(self, time_series=False):
        """returns either the summary or the time series dataframe."""
        if time_series:
            return self.time_series_df
        return self.summary_df

    def print_mod_stats(self, name):
        """prints stats for a specific model."""
        print(self.summary_df.loc[name])

    def get_col(self, col, time_series=False):
        """fetches a specific column from the active dataframe."""
        if time_series:
            return self.time_series_df[col]
        return self.summary_df[col]

    # --- analysis methods ---

    def plt_loss_curves(self, mup=0, filepref=None):
        """plots training loss curves for models, separated by mup usage."""
        df = self.time_series_df.copy()

        # filter for mup models based on the summary
        mup_models = self.summary_df[self.summary_df["use_mup"] == mup].index
        df = df[df.index.isin(mup_models)]

        model_names = df.index.unique()

        palette = sns.color_palette("bright", n_colors=len(model_names))

        fig, ax = plt.subplots(figsize=(10, 6))

        # plot a curve for each model
        for i, m in enumerate(model_names):
            model_df = df.loc[m]

            tokens_by_step = np.array(model_df["tok_by_step"])
            loss_curve = np.array(model_df["loss_curve"])

            ax.plot(
                tokens_by_step,
                loss_curve,
                label=m,
                color=palette[i],
                alpha=0.3,
                lw=2
            )

        # format the plot
        ax.set_title(f"Loss Curves: ")
        ax.set_xlabel("Tokens Processed")
        ax.set_ylabel("Loss")

        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_plot(f"{filepref}loss_curves.png")

        plt.show()

    def display_training_stats(self, filepref=None):
        """generates and saves a clean visual table of the training stats."""

        display_map = {
            "params": "Params",
            "tok_per_sec": "Tokens/Sec",
            "train_time": "Wall Time (s)",
            "mem_mb": "GPU Mem (MB)",
            "mean_tr_loss": "Avg Train Loss",
            "mean_val_loss": "Final Val Loss"
        }
        if "mean_tr_loss" not in self.summary_df.columns:
            print("No training data available to display.")
            return

        filtered_df = self.summary_df.dropna(subset=["mean_tr_loss"]).copy()

        available_cols = [c for c in display_map.keys() if c in filtered_df.columns]
        plot_df = filtered_df[available_cols].copy()

        # rename columns for the table header
        plot_df.columns = [display_map[c] for c in available_cols]

        # move model name from index to a standard column
        plot_df = plot_df.reset_index().rename(columns={"model": "Model Name"})

        # helper to format numbers and handle missing values safely
        def format_val(val, fmt):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "-"
            try:
                return fmt.format(val)
            except:
                return str(val)

        # apply visual formatting to table data
        plot_df["Params"] = plot_df["Params"].apply(lambda x: format_val(x, "{:,.0f}"))
        plot_df["Tokens/Sec"] = plot_df["Tokens/Sec"].apply(lambda x: format_val(x, "{:,.0f}"))
        plot_df["Wall Time (s)"] = plot_df["Wall Time (s)"].apply(lambda x: format_val(x, "{:.2f}"))
        plot_df["GPU Mem (MB)"] = plot_df["GPU Mem (MB)"].apply(lambda x: format_val(x, "{:.1f}"))
        plot_df["Avg Train Loss"] = plot_df["Avg Train Loss"].apply(lambda x: format_val(x, "{:.4f}"))
        plot_df["Final Val Loss"] = plot_df["Final Val Loss"].apply(lambda x: format_val(x, "{:.4f}"))

        # create and style the visual table
        fig, ax = plt.subplots(figsize=(13, 2 + len(plot_df) * 0.6))
        ax.axis('off')

        tbl = ax.table(
            cellText=plot_df.values,
            colLabels=plot_df.columns,
            cellLoc='center',
            loc='center',
            colColours=["#40466e"] * len(plot_df.columns)
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.5)  # extra vertical scale for clarity

        # stripe the table rows for readability
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
            elif row > 0 and row % 2 == 0:
                cell.set_facecolor('#f2f2f2')

        plt.title("Model Training Stats", fontsize=16, pad=30)
        save_plot(f"{filepref}stat_table.png")
        plt.show()

    def fit_scaling_law_manual(self, mup=0):
        """manually fits a simple log-log regression to parameter and loss data."""
        mask = (self.summary_df["use_mup"] == mup)
        df = self.summary_df[mask].copy()
        N = df['params'].values.astype(float)
        L = df['mean_val_loss'].values.astype(float)
        log_N = np.log(N)
        log_L = np.log(L)
        slope, intercept = np.polyfit(log_N, log_L, 1)

        alpha_manual = -slope
        a_manual = np.exp(intercept)

        print(f"--- Scaling Law Results ---")
        print(f"Fitted Alpha (α): {alpha_manual:.4f}")
        print(f"Fitted Scale (a): {a_manual:.4f}")

        return alpha_manual, a_manual

    def plot_scaling_law_full(self, idata, alpha_man, a_man, mup=0, n_points=200, filepref=None):
        """plots the full scaling law comparing manual fits vs pymc posteriors."""
        colors = sns.color_palette("viridis", 5)
        sns.set_style("white")
        mask = (self.summary_df["use_mup"] == mup)
        df = self.summary_df[mask].dropna(subset=["params", "params_theory", "mean_val_loss"])

        N_obs = df["params"].values.astype(float)
        N_theo = df["params_theory"].values.astype(float)
        L_obs = df["mean_val_loss"].values.astype(float)

        # sort observed data by parameter count
        s_idx = np.argsort(N_obs)
        N_obs, N_theo, L_obs = N_obs[s_idx], N_theo[s_idx], L_obs[s_idx]

        # generate a smooth grid for the curve
        N_grid = np.geomspace(N_theo.min() * 0.5, N_obs.max() * 5, n_points)

        L_manual = a_man * N_grid ** (-alpha_man)

        # extract posterior samples
        post = idata.posterior
        a_s = post["a"].values.flatten()
        alpha_s = post["alpha"].values.flatten()
        c_s = post["c"].values.flatten()

        mu_samples = np.array([
            a * N_grid ** (-alpha) + c
            for a, alpha, c in zip(a_s, alpha_s, c_s)
        ])

        fig, ax = plt.subplots(figsize=(10, 6))

        # plot observed vs theoretical
        ax.scatter(N_obs, L_obs, color=colors[0], s=70, alpha=0.5, label="Total Params")
        ax.scatter(N_theo, L_obs, color='red', marker='x', s=70, label="Theoretical Params")
        for i in range(len(N_obs)):
            ax.plot([N_obs[i], N_theo[i]], [L_obs[i], L_obs[i]], color='gray', linestyle=':', alpha=0.4)

        # overlay pymc posterior confidence intervals
        az.plot_hdi(N_grid, mu_samples, hdi_prob=0.89, ax=ax,
                    fill_kwargs={"alpha": 0.3, "color": colors[2], "label": "PyMC 89% Mean HDI"})

        ax.plot(N_grid, mu_samples.mean(axis=0), color=colors[2], lw=3, label="PyMC Posterior Mean")

        # add manual fit line
        ax.plot(N_grid, L_manual, "--", color='orange', lw=2, label=f"Manual Fit (α={alpha_man})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (N)")
        ax.set_ylabel("Loss (L)")
        ax.set_title(f"Scaling Law")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9)
        save_plot(f"{filepref}scaling law plot.png")
        sns.despine()
        plt.show()

    def fit_scaling_law_mcmc_series(self, draws=1000, tune=1000, mup=0, real_param_count=False):
        """fits a scaling law using pymc on time series data."""
        df = self.time_series_df.copy()
        loss_col = 'val_curve'
        df[loss_col] = pd.to_numeric(df[loss_col], errors='coerce')
        df['use_mup'] = df.index.get_level_values(0).map(self.summary_df['use_mup'])

        param_source = 'params'
        df['N'] = df.index.get_level_values(0).map(self.summary_df[param_source])

        mask = (df['use_mup'] == mup)
        clean_df = df[mask].dropna(subset=[loss_col, 'N'])

        losses = clean_df[loss_col].values.astype(float)
        params = clean_df['N'].values.astype(float)

        import pymc as pm
        with pm.Model() as model:
            N_data = pm.Data("N", params)
            loss = pm.Data("loss", losses)

            alpha = pm.Normal("alpha", mu=0.2, sigma=0.1)
            a = pm.Exponential("a", 1.0)
            c = pm.Uniform("c", lower=0, upper=losses.min() * 0.99)

            mu = pm.Deterministic("mu", a * N_data ** (-alpha) + c)
            sigma = pm.HalfNormal("sigma", 0.2)

            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=loss)
            idata = pm.sample(draws=draws, tune=tune, target_accept=0.9, chains=2,
                              idata_kwargs={"log_likelihood": True})

        # return the model and inference data
        return model, idata

    def compute_theoretical_params(self):
        """
        calculates theoretical parameter counts for models.
        adds a params_theory column to the summary dataframe.
        """

        def get_row_theory(row):
            # mapping from your dataframe columns
            d_model = row['n_embd']
            n_layer = row['n_layer']
            d_ff = row['d_ff']
            vocab = row['vocab_size']

            # attention blocks: weights, biases, and projections
            att = (4 * d_model ** 2) + (4 * d_model)

            # feed forward blocks
            ffw = (2 * d_model * d_ff) + d_ff + d_model

            # layer norms and final output head
            ln_and_head = (n_layer * 4 * d_model) + (2 * d_model) + (d_model * vocab)

            return n_layer * (att + ffw) + ln_and_head

        self.summary_df['params_theory'] = self.summary_df.apply(get_row_theory, axis=1)

        print("Theoretical parameters calculated and added to summary_df.")
        return self.summary_df[['params', 'params_theory']].head()

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pymc as pm
    import arviz as az

    def plot_scaling_law_extrapolation(self, model, idata, alpha_man, a_man, mup=1, ns=1000, filepref=None):
        """plots extrapolation curves based on the scaling law models."""

        # setup plotting styles
        colors = sns.color_palette("husl", 6)
        c_idx = 1
        sns.set_style("white")

        # isolate observed data
        mask = (self.summary_df["use_mup"] == mup)
        df = self.summary_df[mask].dropna(subset=["params", "mean_val_loss"])
        N_obs = df["params"].values.astype(float)
        L_obs = df['mean_val_loss'].values.astype(float)
        N_xl = N_obs.max()

        # generate an extended grid for extrapolation
        N_grid = np.linspace(0, N_xl * 15, ns)

        fig, ax = plt.subplots(figsize=(12, 8))

        with model:

            pm.set_data({
                "N": N_grid,
                'loss': np.zeros(ns)
            })

            # draw samples from posterior predictive distribution
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=["mu", "obs"],
                predictions=True
            )

        # extract xarrays natively to avoid conversion issues
        mu_samples = az.extract(ppc, group="predictions")["mu"]
        pred_samples = az.extract(ppc, group="predictions")["obs"]

        # plot true mean vals
        ax.scatter(N_obs, L_obs, color='green', alpha=0.8, s=80, zorder=10, label="Model Parameters")

        # plot the total predictive uncertainty
        az.plot_hdi(N_grid, pred_samples.T, hdi_prob=0.89, ax=ax,
                    fill_kwargs={"alpha": 0.1, "color": colors[3], "label": "89% pred hdi"}
                    )

        # plot the mean uncertainty
        az.plot_hdi(N_grid, mu_samples.T, hdi_prob=0.89, ax=ax,
                    fill_kwargs={"alpha": 0.4, "color": colors[2], "label": "89% Mean HDI"}
                    )

        # mean line
        ax.plot(N_grid, mu_samples.mean("sample"), color=colors[5],
                lw=2.5, label="Posterior Mean Line"
                )

        # naive manual fit
        manual_mu = a_man * N_grid ** (-alpha_man)
        ax.plot(N_grid, manual_mu, color='red', linestyle='--', lw=2, zorder=12, label="Manual Theoretical Fit")

        # calculate and plot point extrapolations at multiple scales
        for s in [2, 5, 10]:
            N_t = N_xl * s

            # calculate guess
            dist_t = idata.posterior["a"] * (N_t ** -idata.posterior["alpha"]) + idata.posterior["c"]
            m_t = dist_t.mean().item()
            ax.axvline(N_t, color='red', linestyle='--', alpha=0.3)
            ax.scatter(N_t, m_t, color='red', marker='D', s=100, zorder=11)
            ax.annotate(f'{s}x XL\n{m_t:.4f}', xy=(N_t, m_t), xytext=(0, 12),
                        textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

        # force log scale for axes
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylim(L_obs.min() * .8, L_obs.max() * 1.1)
        ax.set_xlim(N_obs.min() * .9, N_grid.max())

        ax.set_xlabel("Parameters (N)", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title("Scaling Law Extrapolation", fontsize=15)

        # prevent legend duplication
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), ncol=2, fontsize=9, loc='lower left')

        sns.despine()
        plt.tight_layout()
        if filepref is not None:
            save_plot(filename=f'{filepref}_scale_extrap')

        plt.show()


def save_plot(filename, folder="visualizations", dark_mode=True):
    """saves a plot to the specified folder with forced white background."""
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)
    if not path.endswith('.png'):
        path += '.png'

    with plt.style.context('default'):
        fig = plt.gcf()

        # re-apply facecolor to avoid transparent background bugs
        fig.set_facecolor('white')

        plt.savefig(
            path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            transparent=False
        )
    return
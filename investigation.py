# The purpose of the project is to create a causal graph based on data retrieved from the server
# Data files live in the `data/` folder and are CSVs with an index column

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class CausalityAnalyzer:
    """
    A class to load multiple datasets with labels, compute summary statistics,
    and visualize pairwise relationships and statistics for causal analysis.
    """
    def __init__(self):
        self.datasets = {}

    def add_data(self, label: str, file_path: str, index_col=0):
        """Load a CSV file into the analyzer under a given label."""
        df = pd.read_csv(file_path, index_col=index_col)
        self.datasets[label] = df

    def combine_data(self) -> pd.DataFrame:
        """Concatenate all labeled datasets, adding a 'label' column."""
        frames = []
        for label, df in self.datasets.items():
            temp = df.copy()
            temp['label'] = label
            frames.append(temp)
        return pd.concat(frames, axis=0) if frames else pd.DataFrame()

    def compute_statistics(self) -> pd.DataFrame:
        """Compute mean, variance, and noise estimates for each variable per dataset."""
        stats_list = []
        for label, df in self.datasets.items():
            stats = pd.DataFrame({
                'mean': df.mean(),
                'variance': df.var()
            })
            noise_diff = df.diff().std()
            rolling_mean = df.rolling(window=10, center=True).mean()
            noise_rolling = (df - rolling_mean).std()
            stats['noise_std_diff'] = noise_diff
            stats['noise_std_rolling'] = noise_rolling
            stats['label'] = label
            stats_list.append(stats)
        if not stats_list:
            return pd.DataFrame()
        # reshape for easy plotting
        stats_df = pd.concat(stats_list)
        stats_df = stats_df.reset_index().rename(columns={'index': 'variable'})
        return stats_df

    def list_labels(self) -> list:
        """Return current dataset labels."""
        return list(self.datasets.keys())

    def get_palette(self, labels=None):
        """Return a consistent color palette for the given labels."""
        if labels is None:
            labels = self.list_labels()
        palette = sns.color_palette("tab10", n_colors=len(labels))
        return dict(zip(labels, palette))

    def conditional_correlation(self, x: str, y: str, given: str, label: str = None, method: str = 'pearson'):
        """
        Compute the partial (conditional) correlation between x and y given 'given' variable.
        Optionally restrict to a specific dataset label.
        method: 'pearson', 'spearman', or 'kendall'
        """
        if label is not None:
            if label not in self.datasets:
                raise ValueError(f"Label {label} not found in datasets.")
            df = self.datasets[label]
        else:
            df = self.combine_data()
        df = df[[x, y, given]].dropna()
        if df.empty:
            return np.nan
        # Regress x and y on 'given', then correlate residuals
        from scipy.stats import pearsonr, spearmanr, kendalltau
        for v in [x, y, given]:
            if v not in df.columns:
                raise ValueError(f"Variable {v} not found in data.")
        # Remove linear effect of 'given' from x and y
        from sklearn.linear_model import LinearRegression
        X_given = df[[given]].values
        x_resid = df[x] - LinearRegression().fit(X_given, df[x]).predict(X_given)
        y_resid = df[y] - LinearRegression().fit(X_given, df[y]).predict(X_given)
        if method == 'pearson':
            corr, _ = pearsonr(x_resid, y_resid)
        elif method == 'spearman':
            corr, _ = spearmanr(x_resid, y_resid)
        elif method == 'kendall':
            corr, _ = kendalltau(x_resid, y_resid)
        else:
            raise ValueError(f"Unknown method: {method}")
        return corr

    def robust_correlation(self, x, y, data, method='pearson'):
        """
        Compute correlation between x and y in data using the specified method.
        """
        from scipy.stats import pearsonr, spearmanr, kendalltau
        if method == 'pearson':
            return pearsonr(data[x], data[y])[0]
        elif method == 'spearman':
            return spearmanr(data[x], data[y])[0]
        elif method == 'kendall':
            return kendalltau(data[x], data[y])[0]
        else:
            raise ValueError(f"Unknown method: {method}")

    def plot_pairplot(self, labels: list = None, kind='scatter', corr_method='pearson'):
        """
        Plot a Seaborn pairplot for loaded datasets filtered by optional labels, with correlation coefficients.
        corr_method: 'pearson', 'spearman', or 'kendall'
        """
        combined = self.combine_data()
        if combined.empty:
            print("No data to plot. Please add datasets first.")
            return
        if labels is not None:
            combined = combined[combined['label'].isin(labels)]
            if combined.empty:
                print(f"No data matches labels {labels}.")
                return
        else:
            labels = self.list_labels()
        palette = self.get_palette(labels)

        g = sns.pairplot(
            combined,
            hue='label',
            kind=kind,
            diag_kind='kde',
            diag_kws={'common_norm': False},
            palette=palette
        )
        # Add correlation to lower triangle, place in lower right
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            ax = g.axes[i, j]
            x_var = g.x_vars[j]
            y_var = g.y_vars[i]
            if x_var in combined.columns and y_var in combined.columns:
                r = self.robust_correlation(x_var, y_var, combined, method=corr_method)
                ax.annotate(
                    f"{corr_method[0].upper()} = {r:.2f}",
                    xy=(0.95, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
                )

        plt.tight_layout()
        plt.show(block=False)

    def plot_statistics(self, labels: list = None):
        """Plot bar charts of summary statistics across datasets, optionally filtered by labels, all in one figure."""
        stats = self.compute_statistics()
        if stats.empty:
            print("No statistics to plot. Please add datasets first.")
            return
        if labels is not None:
            stats = stats[stats['label'].isin(labels)]
            if stats.empty:
                print(f"No statistics matches labels {labels}.")
                return
        else:
            labels = self.list_labels()
        palette = self.get_palette(labels)
        metrics = ['mean', 'variance', 'noise_std_diff', 'noise_std_rolling']
        # arrange metrics in a grid of subplots
        n = len(metrics)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten()
        for idx, metric in enumerate(metrics):
            pivot = stats.pivot(index='variable', columns='label', values=metric)
            # Use the palette to set colors for each label
            label_colors = [palette[label] for label in pivot.columns]
            pivot.plot(kind='bar', ax=axes[idx], color=label_colors)
            axes[idx].set_title(f"{metric.replace('_', ' ').title()} by Dataset")
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
        # hide unused subplots if any
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show(block=False)

    def plot_conditional_correlation_heatmaps(self, label=None, method='pearson', annot=True, figsize=(12, 4)):
        """
        Plot a grid of heatmaps showing conditional correlations for all variable pairs,
        conditioning on each other variable in turn.
        Each heatmap corresponds to conditioning on a different variable.
        """
        if label is not None:
            if label not in self.datasets:
                raise ValueError(f"Label {label} not found in datasets.")
            df = self.datasets[label]
        else:
            df = self.combine_data()
        variables = [v for v in df.columns if v != 'label']
        n = len(variables)
        if n < 3:
            print("Need at least 3 variables for conditional correlation heatmaps.")
            return
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1]*nrows))
        axes = np.array(axes).reshape(-1)
        for idx, cond_var in enumerate(variables):
            heat = np.zeros((n, n))
            for i, x in enumerate(variables):
                for j, y in enumerate(variables):
                    if x == y or x == cond_var or y == cond_var:
                        heat[i, j] = np.nan
                    else:
                        heat[i, j] = self.conditional_correlation(x, y, cond_var, label=label, method=method)
            ax = axes[idx]
            sns.heatmap(heat, xticklabels=variables, yticklabels=variables, annot=annot, fmt=".2f", center=0, cmap="coolwarm", ax=ax, cbar=idx==0)
            ax.set_title(f"Cond. on {cond_var}")
        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show(block=False)


if __name__ == '__main__':
    # Example usage: adjust paths as needed
    analyzer = CausalityAnalyzer()
    analyzer.add_data('without_intervention', 'data/data_1938.csv')
    # analyzer.add_data('intervention_D_-2', 'data/data_1998.csv')
    analyzer.add_data('intervention_B_2', 'data/data_1999.csv')
    # Plot pairwise relationships
    analyzer.plot_pairplot()
    # Plot summary statistics
    analyzer.plot_statistics()
    
    # analyzer.plot_conditional_correlation_heatmaps()
    plt.show()
# The purpose of the project is to create a causal graph based on data retrieved from the server
# Data files live in the `data/` folder and are CSVs with an index column

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    def plot_pairplot(self, labels: list = None, kind='scatter'):
        """Plot a Seaborn pairplot for loaded datasets filtered by optional labels."""
        combined = self.combine_data()
        if combined.empty:
            print("No data to plot. Please add datasets first.")
            return
        if labels is not None:
            combined = combined[combined['label'].isin(labels)]
            if combined.empty:
                print(f"No data matches labels {labels}.")
                return
        sns.pairplot(
            combined,
            hue='label',
            kind=kind,
            diag_kind='kde',
            diag_kws={'common_norm': False}
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
        metrics = ['mean', 'variance', 'noise_std_diff', 'noise_std_rolling']
        # arrange metrics in a grid of subplots
        n = len(metrics)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten()
        for idx, metric in enumerate(metrics):
            pivot = stats.pivot(index='variable', columns='label', values=metric)
            pivot.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f"{metric.replace('_', ' ').title()} by Dataset")
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
        # hide unused subplots if any
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show(block=False)


if __name__ == '__main__':
    # Example usage: adjust paths as needed
    analyzer = CausalityAnalyzer()
    analyzer.add_data('without_intervention', 'data/data_1269.csv')
    analyzer.add_data('with_intervention_A_0', 'data/data_1292.csv')
    # Plot pairwise relationships
    analyzer.plot_pairplot()
    # Plot summary statistics
    analyzer.plot_statistics()
    plt.show()
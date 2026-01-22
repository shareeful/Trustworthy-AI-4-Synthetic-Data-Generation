"""
Visualization utilities for results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Plot training curves for multiple metrics.
    
    Args:
        metrics_history: Dictionary of {metric_name: [values]}
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    metric_names = list(metrics_history.keys())[:4]  # Plot first 4 metrics
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = metrics_history[metric_name]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} over Training', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for idx in range(len(metric_names), 4):
        fig.delaxes(axes[idx])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_weight_evolution(
    weights_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Loss Weight Evolution"
):
    """
    Plot evolution of loss weights during closed-loop adjustment.
    
    Args:
        weights_history: Dictionary of {weight_name: [values]}
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for weight_name, values in weights_history.items():
        cycles = range(len(values))
        ax.plot(cycles, values, linewidth=2, marker='o', label=f'λ_{weight_name}')
    
    ax.set_xlabel('Adjustment Cycle', fontweight='bold')
    ax.set_ylabel('Weight Value', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved weight evolution to {save_path}")
    
    plt.show()


def plot_compliance_comparison(
    baseline_metrics: Dict[str, float],
    ours_metrics: Dict[str, float],
    thresholds: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Compliance Comparison"
):
    """
    Plot comparison of metrics between baseline and our method.
    
    Args:
        baseline_metrics: Baseline method metrics
        ours_metrics: Our method metrics
        thresholds: Compliance thresholds
        save_path: Path to save figure
        title: Plot title
    """
    metrics_names = list(baseline_metrics.keys())
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    baseline_values = [baseline_metrics[m] for m in metrics_names]
    ours_values = [ours_metrics[m] for m in metrics_names]
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, ours_values, width, label='Our Method', alpha=0.8)
    
    # Plot thresholds as horizontal lines
    for idx, metric in enumerate(metrics_names):
        if metric in thresholds:
            threshold = thresholds[metric]
            ax.hlines(threshold, idx - 0.5, idx + 0.5, colors='red', 
                     linestyles='dashed', label='Threshold' if idx == 0 else '')
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Values', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved compliance comparison to {save_path}")
    
    plt.show()


def generate_results_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate formatted results table for paper.
    
    Args:
        results: Dictionary of {method_name: {metric: value}}
        save_path: Path to save CSV
        
    Returns:
        DataFrame with formatted results
    """
    df = pd.DataFrame(results).T
    
    # Format values
    for col in df.columns:
        if 'accuracy' in col.lower() or 'rate' in col.lower():
            df[col] = df[col].apply(lambda x: f"{x:.1%}")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
    
    if save_path:
        df.to_csv(save_path)
        logger.info(f"Saved results table to {save_path}")
    
    return df

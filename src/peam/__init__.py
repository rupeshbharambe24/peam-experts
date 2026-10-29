"""PEAM: Pattern-Expert Additive Model with log-odds reason codes."""
from .model import PatternExpertAdditiveModel
from .metrics import all_metrics, brier, ece, pr_auc, roc_auc

__all__ = ["PatternExpertAdditiveModel", "all_metrics", "roc_auc", "pr_auc", "brier", "ece"]
__version__ = "0.1.0"

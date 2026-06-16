export const MODEL_COLORS: Record<string, string> = {
  XGBoost: '#3498db',
  GCN: '#e74c3c',
  GraphSAGE: '#2ecc71',
  GAT: '#f39c12',
  Ensemble: '#9b59b6',
};

export const METRICS = ['f1', 'roc_auc', 'precision', 'recall'] as const;

export const METRIC_LABELS: Record<string, string> = {
  f1: 'F1 Score',
  roc_auc: 'ROC-AUC',
  precision: 'Precision',
  recall: 'Recall',
  accuracy: 'Accuracy',
};

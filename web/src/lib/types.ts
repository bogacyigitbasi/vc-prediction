export interface GraphNode {
  id: string;
  type: 'investor' | 'company';
  label: string;
  degree: number;
  cluster: number;
  category?: string;
  country?: string;
  fundingTotal?: number;
  hasLabel?: boolean;
  prediction?: number;
  portfolioSize?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    totalNodes: number;
    totalEdges: number;
    investors: number;
    companies: number;
  };
}

export interface ModelResult {
  name: string;
  shortName: string;
  type: 'tabular' | 'gnn' | 'ensemble';
  f1: number;
  roc_auc: number;
  precision: number;
  recall: number;
  accuracy: number | null;
  description: string;
  color: string;
}

export interface ImprovementStep {
  step: string;
  f1: number;
  auc: number;
  delta: string | null;
}

export interface AblationResult {
  name: string;
  setting: string;
  f1: number;
  roc_auc: number;
  whatsRemoved: string;
}

export interface Company {
  name: string;
  permalink: string;
  category: string;
  country: string;
  city: string;
  triggerRound: string;
  triggerDate: string;
  triggerAmount: number;
  label: number;
  split: string;
  status: string;
  fundingTotal: number;
  investors: string[];
}

export interface DatasetStats {
  totalCompanies: number;
  totalInvestors: number;
  totalEdges: { train: number; val: number; test: number };
  labeledCompanies: number;
  positiveRate: number;
  splits: Record<string, { count: number; positiveRate: number }>;
  companyFeatureDims: number;
  investorFeatureDims: number;
}

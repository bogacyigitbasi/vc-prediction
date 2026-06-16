import type { GraphData, ModelResult, ImprovementStep, AblationResult, Company, DatasetStats } from './types';

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return res.json();
}

export async function getGraphData(): Promise<GraphData> {
  return fetchJson('/data/graph_overview.json');
}

export async function getModelResults(): Promise<{ models: ModelResult[]; improvementJourney: ImprovementStep[] }> {
  return fetchJson('/data/model_results.json');
}

export async function getAblationResults(): Promise<{ ablations: AblationResult[]; insights: string[] }> {
  return fetchJson('/data/ablation_results.json');
}

export async function getCompanies(): Promise<Company[]> {
  return fetchJson('/data/companies.json');
}

export async function getDatasetStats(): Promise<DatasetStats> {
  return fetchJson('/data/dataset_stats.json');
}

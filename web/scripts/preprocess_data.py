"""Convert ML project data into static JSON files for the web app."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED = DATA_DIR / 'processed'
INTERIM = DATA_DIR / 'interim'
RAW_CB = DATA_DIR / 'raw' / 'crunchbase_2015'
OUT_DIR = Path(__file__).parent.parent / 'public' / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def generate_dataset_stats():
    print("Generating dataset_stats.json...")
    splits = pd.read_csv(INTERIM / 'splits.csv')
    edges_train = pd.read_csv(PROCESSED / 'edges_train.csv')
    edges_val = pd.read_csv(PROCESSED / 'edges_val.csv')
    edges_test = pd.read_csv(PROCESSED / 'edges_test.csv')
    companies = pd.read_csv(PROCESSED / 'company_index.csv')
    investors = pd.read_csv(PROCESSED / 'investor_index.csv')

    stats = {
        "totalCompanies": len(companies),
        "totalInvestors": len(investors),
        "totalEdges": {
            "train": len(edges_train),
            "val": len(edges_val),
            "test": len(edges_test),
        },
        "labeledCompanies": len(splits),
        "positiveRate": round(splits['label'].mean(), 3),
        "splits": {},
        "companyFeatureDims": 58,
        "investorFeatureDims": 7,
    }

    for split_name in ['train', 'val', 'test']:
        s = splits[splits['split'] == split_name]
        stats["splits"][split_name] = {
            "count": len(s),
            "positiveRate": round(s['label'].mean(), 3),
        }

    with open(OUT_DIR / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  -> {OUT_DIR / 'dataset_stats.json'}")


def generate_model_results():
    print("Generating model_results.json...")
    data = {
        "models": [
            {
                "name": "XGBoost (baseline)",
                "shortName": "XGBoost",
                "type": "tabular",
                "f1": 0.5577, "roc_auc": 0.7114,
                "precision": 0.5956, "recall": 0.5244, "accuracy": 0.6666,
                "description": "Gradient-boosted decision trees on tabular company features only. No graph structure. The gold standard baseline for structured data.",
                "color": "#3498db",
            },
            {
                "name": "GCN (GraphConv)",
                "shortName": "GCN",
                "type": "gnn",
                "f1": 0.5435, "roc_auc": 0.6113,
                "precision": 0.4522, "recall": 0.6810, "accuracy": 0.5413,
                "description": "Graph Convolutional Network with symmetric aggregation. Failed to train stably on bipartite graph — loss spikes and degenerate predictions due to learning rate sensitivity.",
                "color": "#e74c3c",
            },
            {
                "name": "GraphSAGE v2",
                "shortName": "GraphSAGE",
                "type": "gnn",
                "f1": 0.6277, "roc_auc": 0.7406,
                "precision": 0.5115, "recall": 0.8122, "accuracy": 0.6138,
                "description": "Sample-and-aggregate GNN with mean aggregation. Best single model. Inductive by design — handles unseen nodes and sparse bipartite structure robustly.",
                "color": "#2ecc71",
            },
            {
                "name": "GAT (4 heads)",
                "shortName": "GAT",
                "type": "gnn",
                "f1": 0.5648, "roc_auc": 0.6581,
                "precision": 0.5194, "recall": 0.6189, "accuracy": 0.6176,
                "description": "Graph Attention Network with 4 attention heads. Attention mechanism needs more data/edges to learn which neighbors matter. Below XGBoost baseline.",
                "color": "#f39c12",
            },
            {
                "name": "Ensemble (XGB + GraphSAGE)",
                "shortName": "Ensemble",
                "type": "ensemble",
                "f1": 0.6377, "roc_auc": 0.7448,
                "precision": 0.5160, "recall": 0.8347, "accuracy": None,
                "description": "Weighted average of XGBoost and GraphSAGE v2. Combines complementary signals: tabular features + graph structure. Best overall F1 and recall.",
                "color": "#9b59b6",
            },
        ],
        "improvementJourney": [
            {"step": "XGBoost baseline", "f1": 0.558, "auc": 0.711, "delta": None},
            {"step": "+ Graph (SAGEv1)", "f1": 0.625, "auc": 0.737, "delta": "+12.0%"},
            {"step": "+ Text + Weights", "f1": 0.628, "auc": 0.741, "delta": "+0.5%"},
            {"step": "+ Ensemble", "f1": 0.638, "auc": 0.745, "delta": "+1.6%"},
        ],
    }

    with open(OUT_DIR / 'model_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  -> {OUT_DIR / 'model_results.json'}")


def generate_ablation_results():
    print("Generating ablation_results.json...")
    data = {
        "ablations": [
            {"name": "Full Model (GraphSAGE v2)", "setting": "All features + temporal edges", "f1": 0.6277, "roc_auc": 0.7406, "whatsRemoved": "Nothing (baseline)"},
            {"name": "Topology-Only", "setting": "Random features", "f1": 0.5723, "roc_auc": 0.5442, "whatsRemoved": "All node features replaced with random vectors"},
            {"name": "No Temporal Filter", "setting": "Leaked future edges", "f1": 0.6202, "roc_auc": 0.7302, "whatsRemoved": "Temporal edge filtering removed"},
            {"name": "XGBoost (no graph)", "setting": "Tabular only", "f1": 0.5577, "roc_auc": 0.7114, "whatsRemoved": "All graph structure removed"},
        ],
        "insights": [
            "Node features are the primary signal source — AUC drops 20 percentage points without them (0.74 → 0.54)",
            "Graph structure provides a complementary +3pp AUC boost over tabular-only (0.71 → 0.74)",
            "Temporal edge filtering is methodologically sound but doesn't inflate results — the VC network is structurally stable 2012-2014",
            "Graph topology alone (random features) is barely above random chance (AUC = 0.54)",
            "The combination of features + graph is greater than either alone",
        ],
    }

    with open(OUT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  -> {OUT_DIR / 'ablation_results.json'}")


def generate_graph_overview():
    print("Generating graph_overview.json...")
    edges = pd.read_csv(PROCESSED / 'edges_test.csv')
    comp_idx = pd.read_csv(PROCESSED / 'company_index.csv')
    inv_idx = pd.read_csv(PROCESSED / 'investor_index.csv')
    comp_texts = pd.read_csv(PROCESSED / 'company_texts.csv')
    splits = pd.read_csv(INTERIM / 'splits.csv')

    raw_companies = pd.read_csv(RAW_CB / 'companies.csv', low_memory=False)
    raw_investments = pd.read_csv(RAW_CB / 'investments.csv', low_memory=False)

    idx_to_comp_permalink = dict(zip(comp_idx['idx'], comp_idx['permalink']))
    idx_to_inv_permalink = dict(zip(inv_idx['idx'], inv_idx['permalink']))

    comp_permalink_to_name = dict(zip(raw_companies['permalink'], raw_companies['name']))
    comp_permalink_to_country = dict(zip(raw_companies['permalink'], raw_companies['country_code']))
    comp_permalink_to_funding = dict(zip(raw_companies['permalink'], raw_companies['funding_total_usd']))

    inv_permalink_to_name = {}
    for _, row in raw_investments.drop_duplicates('investor_permalink').iterrows():
        inv_permalink_to_name[row['investor_permalink']] = row['investor_name']

    comp_text_map = dict(zip(comp_texts['idx'], comp_texts['text']))
    splits_map = dict(zip(splits['company_permalink'], splits['label']))

    inv_degree = defaultdict(int)
    comp_degree = defaultdict(int)
    for _, row in edges.iterrows():
        inv_degree[int(row['investor_idx'])] += 1
        comp_degree[int(row['company_idx'])] += 1

    top_investors = sorted(inv_degree.keys(), key=lambda x: inv_degree[x], reverse=True)[:250]
    top_companies = sorted(comp_degree.keys(), key=lambda x: comp_degree[x], reverse=True)[:250]

    top_inv_set = set(top_investors)
    top_comp_set = set(top_companies)

    nodes = []
    for inv in top_investors:
        permalink = idx_to_inv_permalink.get(inv, '')
        name = inv_permalink_to_name.get(permalink, permalink.split('/')[-1] if permalink else f'Investor {inv}')
        nodes.append({
            "id": f"i_{inv}",
            "type": "investor",
            "label": name,
            "degree": inv_degree[inv],
            "cluster": 0,
            "portfolioSize": inv_degree[inv],
        })

    for comp in top_companies:
        permalink = idx_to_comp_permalink.get(comp, '')
        name = comp_permalink_to_name.get(permalink, permalink.split('/')[-1] if permalink else f'Company {comp}')
        country = comp_permalink_to_country.get(permalink, '')
        funding = comp_permalink_to_funding.get(permalink, 0)
        category = comp_text_map.get(comp, '')
        label = splits_map.get(permalink)
        nodes.append({
            "id": f"c_{comp}",
            "type": "company",
            "label": name,
            "degree": comp_degree[comp],
            "cluster": 0,
            "category": category if isinstance(category, str) else '',
            "country": country if isinstance(country, str) else '',
            "fundingTotal": int(float(funding)) if pd.notna(funding) and str(funding).replace('.','').replace('-','').isdigit() else 0,
            "hasLabel": label is not None,
            "prediction": int(label) if label is not None else None,
        })

    graph_edges = []
    for _, row in edges.iterrows():
        inv = int(row['investor_idx'])
        comp = int(row['company_idx'])
        if inv in top_inv_set and comp in top_comp_set:
            graph_edges.append({"source": f"i_{inv}", "target": f"c_{comp}"})

    # Simple cluster assignment based on connected components via top investor
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    adjacency = defaultdict(set)
    for e in graph_edges:
        adjacency[e['source']].add(e['target'])
        adjacency[e['target']].add(e['source'])

    visited = set()
    cluster_id = 0
    for n in sorted(nodes, key=lambda x: x['degree'], reverse=True):
        if n['id'] in visited:
            continue
        queue = [n['id']]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in node_id_to_idx:
                nodes[node_id_to_idx[current]]['cluster'] = cluster_id
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        cluster_id += 1

    result = {
        "nodes": nodes,
        "edges": graph_edges,
        "stats": {
            "totalNodes": len(nodes),
            "totalEdges": len(graph_edges),
            "investors": len(top_investors),
            "companies": len(top_companies),
        },
    }

    with open(OUT_DIR / 'graph_overview.json', 'w') as f:
        json.dump(result, f)
    print(f"  -> {OUT_DIR / 'graph_overview.json'} ({len(nodes)} nodes, {len(graph_edges)} edges)")


def generate_companies():
    print("Generating companies.json...")
    splits = pd.read_csv(INTERIM / 'splits.csv')
    raw_companies = pd.read_csv(RAW_CB / 'companies.csv', low_memory=False)
    raw_investments = pd.read_csv(RAW_CB / 'investments.csv', low_memory=False)

    comp_meta = raw_companies.set_index('permalink')[['name', 'category_list', 'country_code', 'city', 'status', 'funding_total_usd']].to_dict('index')

    inv_by_company = defaultdict(list)
    for _, row in raw_investments.iterrows():
        if pd.notna(row['investor_name']):
            inv_by_company[row['company_permalink']].append(row['investor_name'])

    companies = []
    for _, row in splits.iterrows():
        permalink = row['company_permalink']
        meta = comp_meta.get(permalink, {})
        name = row.get('company_name', meta.get('name', permalink.split('/')[-1]))
        companies.append({
            "name": name if isinstance(name, str) else permalink.split('/')[-1],
            "permalink": permalink,
            "category": meta.get('category_list', '') if isinstance(meta.get('category_list'), str) else '',
            "country": meta.get('country_code', '') if isinstance(meta.get('country_code'), str) else '',
            "city": meta.get('city', '') if isinstance(meta.get('city'), str) else '',
            "triggerRound": row['trigger_round_type'],
            "triggerDate": str(row['trigger_funded_at']),
            "triggerAmount": safe_float(row['trigger_amount_usd']),
            "label": int(row['label']),
            "split": row['split'],
            "status": meta.get('status', '') if isinstance(meta.get('status'), str) else '',
            "fundingTotal": safe_float(meta.get('funding_total_usd', 0)),
            "investors": list(set(inv_by_company.get(permalink, [])))[:10],
        })

    with open(OUT_DIR / 'companies.json', 'w') as f:
        json.dump(companies, f)
    print(f"  -> {OUT_DIR / 'companies.json'} ({len(companies)} companies)")


def generate_features():
    print("Generating features.json...")
    splits = pd.read_csv(INTERIM / 'splits.csv')
    raw_companies = pd.read_csv(RAW_CB / 'companies.csv', low_memory=False)
    raw_investments = pd.read_csv(RAW_CB / 'investments.csv', low_memory=False)

    raw_co = raw_companies.set_index('permalink')

    # Pre-compute per-company aggregates from investments
    inv_count_per_co = raw_investments.groupby('company_permalink')['investor_permalink'].nunique().to_dict()
    max_round_per_co = raw_investments.groupby('company_permalink')['raised_amount_usd'].max().to_dict()
    last_round_per_co = raw_investments.groupby('company_permalink')['funded_at'].max().to_dict()
    round_count_per_co = raw_investments.groupby('company_permalink')['funding_round_permalink'].nunique().to_dict()

    # Company features from raw data
    company_rows = []
    for _, row in splits.iterrows():
        permalink = row['company_permalink']
        name = row.get('company_name', permalink.split('/')[-1])
        if not isinstance(name, str):
            name = permalink.split('/')[-1]

        meta = raw_co.loc[permalink] if permalink in raw_co.index else pd.Series()

        founded_at = meta.get('founded_at', '')
        founded_year = int(str(founded_at)[:4]) if isinstance(founded_at, str) and len(str(founded_at)) >= 4 and str(founded_at)[:4].isdigit() else None

        trigger_date = row.get('trigger_funded_at', '')
        days_since_founded = None
        if founded_year and isinstance(trigger_date, str) and len(trigger_date) >= 10:
            try:
                fd = pd.Timestamp(founded_at)
                td = pd.Timestamp(trigger_date)
                days_since_founded = (td - fd).days
            except Exception:
                pass

        funding_total = safe_float(meta.get('funding_total_usd', 0))
        rounds = int(round_count_per_co.get(permalink, safe_float(meta.get('funding_rounds', 0))))
        investors = int(inv_count_per_co.get(permalink, 0))
        max_amount = safe_float(max_round_per_co.get(permalink, 0))
        country = meta.get('country_code', '')
        category = meta.get('category_list', '')

        last_round_date = last_round_per_co.get(permalink, '')
        days_since_last = None
        if isinstance(trigger_date, str) and isinstance(last_round_date, str) and len(trigger_date) >= 10 and len(str(last_round_date)) >= 10:
            try:
                td = pd.Timestamp(trigger_date)
                ld = pd.Timestamp(last_round_date)
                days_since_last = (td - ld).days
            except Exception:
                pass

        company_rows.append({
            'name': name,
            'label': int(row['label']),
            'split': row['split'],
            'founded_year': founded_year,
            'days_since_founded': days_since_founded,
            'funding_total_usd': int(funding_total) if funding_total else 0,
            'round_count': rounds,
            'investor_count': investors,
            'max_round_usd': int(max_amount) if max_amount else 0,
            'days_since_last_round': days_since_last,
            'country': country if isinstance(country, str) else '',
            'category': category if isinstance(category, str) else '',
            'trigger_round': row['trigger_round_type'],
            'trigger_amount_usd': int(safe_float(row.get('trigger_amount_usd', 0))),
        })

    # Investor features from raw data
    inv_agg = raw_investments.groupby('investor_permalink').agg(
        name=('investor_name', 'first'),
        total_investments=('funding_round_permalink', 'nunique'),
        portfolio_size=('company_permalink', 'nunique'),
        categories=('company_category_list', lambda x: x.dropna().nunique()),
        median_amount=('raised_amount_usd', 'median'),
        first_date=('funded_at', 'min'),
        last_date=('funded_at', 'max'),
        countries=('company_country_code', lambda x: x.dropna().nunique()),
    ).reset_index()
    inv_agg = inv_agg.sort_values('total_investments', ascending=False).head(500)

    investor_rows = []
    for _, r in inv_agg.iterrows():
        name = r['name']
        if not isinstance(name, str) or pd.isna(name):
            name = str(r['investor_permalink']).split('/')[-1]
        years = 0
        try:
            fd = pd.Timestamp(r['first_date'])
            ld = pd.Timestamp(r['last_date'])
            years = round((ld - fd).days / 365.25, 1)
        except Exception:
            pass
        investor_rows.append({
            'name': name,
            'total_investments': int(r['total_investments']),
            'portfolio_size': int(r['portfolio_size']),
            'category_diversity': int(r['categories']),
            'median_amount_usd': int(safe_float(r['median_amount'])),
            'years_active': years,
            'geo_diversity': int(r['countries']),
        })

    comp_cols = ['name', 'label', 'split', 'founded_year', 'funding_total_usd', 'round_count',
                 'investor_count', 'max_round_usd', 'trigger_round', 'trigger_amount_usd',
                 'days_since_founded', 'days_since_last_round', 'country', 'category']
    inv_cols = ['name', 'total_investments', 'portfolio_size', 'category_diversity',
                'median_amount_usd', 'years_active', 'geo_diversity']

    result = {
        'companyFeatures': company_rows[:5000],
        'investorFeatures': investor_rows,
        'companyColumns': comp_cols,
        'investorColumns': inv_cols,
    }

    with open(OUT_DIR / 'features.json', 'w') as f:
        json.dump(result, f)
    print(f"  -> {OUT_DIR / 'features.json'} ({len(company_rows[:5000])} companies, {len(investor_rows)} investors)")


if __name__ == '__main__':
    generate_dataset_stats()
    generate_model_results()
    generate_ablation_results()
    generate_graph_overview()
    generate_companies()
    generate_features()
    print("\nDone! All JSON files generated in", OUT_DIR)

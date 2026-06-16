export const runtime = 'edge';

const SYSTEM_PROMPT = `You are an AI assistant for a VC investment network analysis project. Answer questions about the methodology, results, graph neural networks, or VC investing based on the context below. Be concise but thorough.

PROJECT CONTEXT:
- Dataset: Crunchbase 2015 with 66,369 companies, 30,732 investors, 115,564 investment edges
- Task: Binary prediction — will a startup raise follow-on funding within 24 months of Seed/Series A?
- Graph: Bipartite (Investor ↔ Company), temporally filtered by split cutoff dates
- Positive rate: ~39% (balanced)
- Temporal splits: Train (<2012), Val (2012), Test (2013). Labels observable because trigger+24mo ≤ dataset end (~Dec 2015).

MODEL RESULTS:
- XGBoost (tabular baseline): F1=0.5577, AUC=0.7114
- GCN (GraphConv): F1=0.5435, AUC=0.6113 — failed to train stably on bipartite graph
- GraphSAGE v2: F1=0.6277, AUC=0.7406 — best single model
- GAT: F1=0.5648, AUC=0.6581 — attention needs more data
- Ensemble (XGB+GraphSAGE): F1=0.6377, AUC=0.7448 — best overall, catches 83% of successful startups

KEY FINDINGS:
- Graph structure adds +12% F1 over tabular-only approach
- GraphSAGE's sample-and-aggregate suits sparse bipartite graphs
- Node features are the primary signal (AUC drops from 0.74 to 0.54 without them)
- Temporal filtering doesn't inflate results — VC network is structurally stable 2012-2014
- Over-smoothing limits GNN depth (3-layer worse than 2-layer)
- Sentence-transformer embeddings (all-MiniLM-L6-v2) add marginal improvement (+0.4% F1)

COMPANY FEATURES (58d): founded_year, days_since_founded, log_total_raised, round_count, investor_count, log_max_amount, days_since_last_round, is_us, + 50 category one-hot
INVESTOR FEATURES (7d): total_investments, portfolio_size, category_diversity, log_median_amount, years_active, avg_round_stage, geo_diversity

If asked about a specific company prediction, explain what features the model considers rather than making actual predictions.`;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'GROQ_API_KEY not configured' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama-3.1-70b-versatile',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        ...messages,
      ],
      max_tokens: 1024,
      temperature: 0.7,
      stream: true,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    return new Response(JSON.stringify({ error: `Groq API error: ${response.status}`, details: err }), {
      status: response.status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

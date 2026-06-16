import Link from 'next/link';
import StatCard from '@/components/layout/StatCard';

const navCards = [
  { href: '/graph', title: 'Explore the Network', desc: 'Interactive bipartite graph of 97K nodes with clustering and drill-down', icon: '◎' },
  { href: '/models', title: 'Compare Models', desc: 'XGBoost vs GCN vs GraphSAGE vs GAT vs Ensemble — metrics, ROC curves, training', icon: '▣' },
  { href: '/explorer', title: 'Browse Dataset', desc: '23,740 labeled companies with filters by sector, geography, funding, and outcome', icon: '⊞' },
  { href: '/ablation', title: 'Ablation Studies', desc: 'Feature importance, topology-only, and temporal leakage experiments', icon: '⊟' },
  { href: '/chat', title: 'Ask the AI', desc: 'Chat with Llama 3.1 about the methodology, results, or VC investing', icon: '◈' },
];

export default function Home() {
  return (
    <div className="max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Predicting Startup Follow-on Funding</h1>
        <p className="text-[var(--text-secondary)] mt-2 text-lg">
          Temporal Graph Neural Networks on Venture Capital Investment Networks
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard label="Companies" value="66,369" subtitle="In the network" />
        <StatCard label="Investors" value="30,732" subtitle="Unique VCs & angels" />
        <StatCard label="Investment Edges" value="115,564" subtitle="Test graph" color="var(--company)" />
        <StatCard label="Models Compared" value="5" subtitle="XGB + 3 GNNs + Ensemble" color="var(--ensemble)" />
      </div>

      <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-6 mb-8">
        <h2 className="text-lg font-semibold mb-2">Key Result</h2>
        <p className="text-[var(--text-secondary)]">
          Our best model — a weighted ensemble of XGBoost and GraphSAGE — achieves{' '}
          <span className="text-[var(--success)] font-bold">F1 = 0.638</span> and{' '}
          <span className="text-[var(--success)] font-bold">ROC-AUC = 0.745</span>,
          catching <span className="text-[var(--accent)] font-bold">83% of companies</span> that
          raise follow-on funding. This is a <span className="font-bold">14.3% F1 improvement</span> over
          tabular-only baselines, proving that VC network structure carries real predictive signal.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {navCards.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5 hover:border-[var(--accent)] hover:bg-[var(--surface-hover)] transition-colors group"
          >
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xl">{card.icon}</span>
              <h3 className="font-semibold group-hover:text-[var(--accent)] transition-colors">{card.title}</h3>
            </div>
            <p className="text-sm text-[var(--text-secondary)]">{card.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

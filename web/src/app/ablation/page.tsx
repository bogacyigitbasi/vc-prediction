'use client';

import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import type { AblationResult } from '@/lib/types';

const COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db'];

export default function AblationPage() {
  const [data, setData] = useState<{ ablations: AblationResult[]; insights: string[] } | null>(null);

  useEffect(() => {
    fetch('/data/ablation_results.json')
      .then(r => r.json())
      .then(setData);
  }, []);

  if (!data) return <div className="text-[var(--text-secondary)]">Loading...</div>;

  const chartData = data.ablations.map(a => ({
    name: a.name,
    F1: a.f1,
    'ROC-AUC': a.roc_auc,
  }));

  const fullModel = data.ablations[0];

  return (
    <div className="max-w-5xl">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Ablation Studies</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          Systematically removing components to measure their contribution
        </p>
      </div>

      <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 mb-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[var(--text-secondary)] border-b border-[var(--border)]">
              <th className="pb-2 pr-4">Setting</th>
              <th className="pb-2 pr-4">F1</th>
              <th className="pb-2 pr-4">ROC-AUC</th>
              <th className="pb-2 pr-4">Delta F1</th>
              <th className="pb-2 pr-4">Delta AUC</th>
              <th className="pb-2">What&apos;s Removed</th>
            </tr>
          </thead>
          <tbody>
            {data.ablations.map((a, i) => {
              const deltaF1 = a.f1 - fullModel.f1;
              const deltaAuc = a.roc_auc - fullModel.roc_auc;
              return (
                <tr key={i} className="border-b border-[var(--border)]/30 hover:bg-[var(--surface-hover)]">
                  <td className="py-2.5 pr-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i] }} />
                    {a.name}
                  </td>
                  <td className="py-2.5 pr-4 font-mono">{a.f1.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 font-mono">{a.roc_auc.toFixed(4)}</td>
                  <td className={`py-2.5 pr-4 font-mono ${deltaF1 < 0 ? 'text-[var(--investor)]' : deltaF1 > 0 ? 'text-[var(--success)]' : 'text-[var(--text-secondary)]'}`}>
                    {i === 0 ? '—' : `${deltaF1 > 0 ? '+' : ''}${deltaF1.toFixed(4)}`}
                  </td>
                  <td className={`py-2.5 pr-4 font-mono ${deltaAuc < 0 ? 'text-[var(--investor)]' : deltaAuc > 0 ? 'text-[var(--success)]' : 'text-[var(--text-secondary)]'}`}>
                    {i === 0 ? '—' : `${deltaAuc > 0 ? '+' : ''}${deltaAuc.toFixed(4)}`}
                  </td>
                  <td className="py-2.5 text-[var(--text-secondary)] text-xs">{a.whatsRemoved}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4">
          <h3 className="text-sm font-semibold mb-3">F1 Score by Setting</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 9 }} />
              <YAxis domain={[0, 0.8]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }} />
              <Bar dataKey="F1" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4">
          <h3 className="text-sm font-semibold mb-3">ROC-AUC by Setting</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 9 }} />
              <YAxis domain={[0, 0.8]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }} />
              <Bar dataKey="ROC-AUC" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5 mb-6">
        <h3 className="text-sm font-semibold mb-3">Key Insights</h3>
        <ul className="space-y-2">
          {data.insights.map((insight, i) => (
            <li key={i} className="text-sm text-[var(--text-secondary)] flex gap-2">
              <span className="text-[var(--accent)] mt-0.5">•</span>
              {insight}
            </li>
          ))}
        </ul>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5">
          <h3 className="text-sm font-semibold mb-2 text-[var(--investor)]">Ablation 1: Topology-Only</h3>
          <p className="text-xs text-[var(--text-secondary)]">
            Replace all node features (442 dimensions) with random Gaussian vectors. Keep graph edges intact. Tests whether graph structure alone carries predictive signal.
          </p>
          <p className="text-xs text-[var(--text-secondary)] mt-2">
            <strong>Result:</strong> AUC collapses from 0.74 to 0.54 (near random). Features are essential.
          </p>
        </div>
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5">
          <h3 className="text-sm font-semibold mb-2 text-[var(--warning)]">Ablation 2: No Temporal Filter</h3>
          <p className="text-xs text-[var(--text-secondary)]">
            Use all edges (including future investments) for every split. Tests whether temporal integrity enforcement matters.
          </p>
          <p className="text-xs text-[var(--text-secondary)] mt-2">
            <strong>Result:</strong> Metrics barely change (F1 delta &lt; 0.01). The VC network is structurally stable 2012-2014.
          </p>
        </div>
      </div>
    </div>
  );
}

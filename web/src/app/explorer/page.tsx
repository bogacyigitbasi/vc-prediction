'use client';

import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import type { Company, GraphData } from '@/lib/types';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

const PAGE_SIZE = 50;

interface EgoGraph {
  nodes: { id: string; type: string; label: string; degree: number; isRoot: boolean; hop: number }[];
  links: { source: string; target: string }[];
  companyName: string;
}

interface FeaturesData {
  companyFeatures: Record<string, unknown>[];
  investorFeatures: Record<string, unknown>[];
  companyColumns: string[];
  investorColumns: string[];
}

type Tab = 'companies' | 'companyFeatures' | 'investorFeatures';

export default function ExplorerPage() {
  const [tab, setTab] = useState<Tab>('companies');
  const [companies, setCompanies] = useState<Company[]>([]);
  const [features, setFeatures] = useState<FeaturesData | null>(null);
  const [search, setSearch] = useState('');
  const [filterLabel, setFilterLabel] = useState<'all' | '1' | '0'>('all');
  const [filterSplit, setFilterSplit] = useState<string>('all');
  const [filterCountry, setFilterCountry] = useState<string>('all');
  const [sortBy, setSortBy] = useState<keyof Company>('triggerAmount');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [featSort, setFeatSort] = useState<{ col: string; dir: 'asc' | 'desc' }>({ col: '', dir: 'desc' });
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Company | null>(null);
  const [graphCache, setGraphCache] = useState<GraphData | null>(null);
  const [egoGraph, setEgoGraph] = useState<EgoGraph | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const egoRef = useRef<any>(null);

  useEffect(() => {
    fetch('/data/companies.json').then(r => r.json()).then(setCompanies);
  }, []);

  useEffect(() => {
    if (tab !== 'companies' && !features) {
      fetch('/data/features.json').then(r => r.json()).then(setFeatures);
    }
  }, [tab, features]);

  const loadGraphData = useCallback(async () => {
    if (graphCache) return graphCache;
    const res = await fetch('/data/graph_overview.json');
    const data: GraphData = await res.json();
    setGraphCache(data);
    return data;
  }, [graphCache]);

  const buildEgoGraph = useCallback(async (company: Company) => {
    const graph = await loadGraphData();
    const investorNames = new Set(company.investors.map(n => n.toLowerCase()));
    const adjacency = new Map<string, Set<string>>();
    for (const e of graph.edges) {
      if (!adjacency.has(e.source)) adjacency.set(e.source, new Set());
      if (!adjacency.has(e.target)) adjacency.set(e.target, new Set());
      adjacency.get(e.source)!.add(e.target);
      adjacency.get(e.target)!.add(e.source);
    }
    const nodeMap = new Map(graph.nodes.map(n => [n.id, n]));
    const seedInvestors = graph.nodes.filter(n => n.type === 'investor' && investorNames.has(n.label.toLowerCase()));
    const rootId = `root_${company.permalink}`;
    const egoNodes: EgoGraph['nodes'] = [{ id: rootId, type: 'company', label: company.name, degree: company.investors.length, isRoot: true, hop: 0 }];
    const egoLinks: EgoGraph['links'] = [];
    const addedIds = new Set<string>([rootId]);
    for (const invName of company.investors) {
      const matched = seedInvestors.find(n => n.label.toLowerCase() === invName.toLowerCase());
      if (matched) {
        if (!addedIds.has(matched.id)) {
          egoNodes.push({ id: matched.id, type: 'investor', label: matched.label, degree: matched.degree, isRoot: false, hop: 1 });
          addedIds.add(matched.id);
        }
        egoLinks.push({ source: rootId, target: matched.id });
      } else {
        const fakeId = `inv_${invName}`;
        if (!addedIds.has(fakeId)) {
          egoNodes.push({ id: fakeId, type: 'investor', label: invName, degree: 1, isRoot: false, hop: 1 });
          addedIds.add(fakeId);
        }
        egoLinks.push({ source: rootId, target: fakeId });
      }
    }
    for (const inv of seedInvestors) {
      const neighbors = adjacency.get(inv.id);
      if (!neighbors) continue;
      for (const nid of neighbors) {
        if (addedIds.has(nid)) continue;
        const n = nodeMap.get(nid);
        if (!n) continue;
        egoNodes.push({ id: n.id, type: n.type, label: n.label, degree: n.degree, isRoot: false, hop: 2 });
        addedIds.add(n.id);
      }
      for (const nid of neighbors) {
        if (addedIds.has(nid)) egoLinks.push({ source: inv.id, target: nid });
      }
    }
    setEgoGraph({ nodes: egoNodes, links: egoLinks, companyName: company.name });
    setTimeout(() => egoRef.current?.zoomToFit(400, 40), 300);
  }, [loadGraphData]);

  // --- Companies tab filtering ---
  const filtered = useMemo(() => {
    let result = companies;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(c => c.name.toLowerCase().includes(q) || c.category.toLowerCase().includes(q));
    }
    if (filterLabel !== 'all') result = result.filter(c => c.label === Number(filterLabel));
    if (filterSplit !== 'all') result = result.filter(c => c.split === filterSplit);
    if (filterCountry !== 'all') result = result.filter(c => c.country === filterCountry);
    result.sort((a, b) => {
      const av = a[sortBy], bv = b[sortBy];
      if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av - bv : bv - av;
      return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return result;
  }, [companies, search, filterLabel, filterSplit, filterCountry, sortBy, sortDir]);

  // --- Features tab filtering/sorting ---
  const featRows = useMemo(() => {
    if (!features) return [];
    const rows = tab === 'companyFeatures' ? features.companyFeatures : features.investorFeatures;
    let result = rows;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(r => (r.name as string).toLowerCase().includes(q));
    }
    if (featSort.col) {
      result = [...result].sort((a, b) => {
        const av = a[featSort.col], bv = b[featSort.col];
        if (typeof av === 'number' && typeof bv === 'number') return featSort.dir === 'asc' ? av - bv : bv - av;
        return featSort.dir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
    }
    return result;
  }, [features, tab, search, featSort]);

  const featCols = features
    ? (tab === 'companyFeatures' ? features.companyColumns : features.investorColumns)
    : [];

  const paged = tab === 'companies'
    ? filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
    : featRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil((tab === 'companies' ? filtered.length : featRows.length) / PAGE_SIZE);

  const handleSort = (col: keyof Company) => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('desc'); }
    setPage(0);
  };

  const handleFeatSort = (col: string) => {
    if (featSort.col === col) setFeatSort({ col, dir: featSort.dir === 'asc' ? 'desc' : 'asc' });
    else setFeatSort({ col, dir: 'desc' });
    setPage(0);
  };

  const countries = useMemo(() => {
    const counts: Record<string, number> = {};
    companies.forEach(c => { counts[c.country] = (counts[c.country] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 15);
  }, [companies]);

  const USD_COLS = new Set(['funding_total_usd', 'max_round_usd', 'trigger_amount_usd', 'median_amount_usd']);
  const formatVal = (col: string, v: any) => {
    if (v === null || v === undefined) return '—';
    if (USD_COLS.has(col) && typeof v === 'number') {
      if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
      if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
      if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
      return v > 0 ? `$${v.toLocaleString()}` : '—';
    }
    if (typeof v === 'number') return v % 1 === 0 ? v.toLocaleString() : v.toFixed(1);
    return String(v);
  };

  if (!companies.length) return <div className="text-[var(--text-secondary)]">Loading companies...</div>;

  return (
    <div className="max-w-[95vw]">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">Dataset Explorer</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          Browse companies, investor networks, and the raw features fed into the models
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-[var(--surface)] rounded-lg border border-[var(--border)] p-1 w-fit">
        {[
          { id: 'companies' as Tab, label: 'Companies', count: filtered.length },
          { id: 'companyFeatures' as Tab, label: 'Company Features', count: features?.companyFeatures.length },
          { id: 'investorFeatures' as Tab, label: 'Investor Features', count: features?.investorFeatures.length },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => { setTab(t.id); setPage(0); setSearch(''); }}
            className={`px-4 py-1.5 rounded text-sm transition-colors ${
              tab === t.id
                ? 'bg-[var(--accent)] text-white'
                : 'text-[var(--text-secondary)] hover:text-white'
            }`}
          >
            {t.label} {t.count !== undefined && <span className="text-xs opacity-70">({t.count?.toLocaleString()})</span>}
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <input
          type="text"
          placeholder={tab === 'companies' ? 'Search by name or category...' : 'Search by name...'}
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0); }}
          className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm w-64"
        />
        {tab === 'companies' && (
          <>
            <select value={filterLabel} onChange={e => { setFilterLabel(e.target.value as typeof filterLabel); setPage(0); }}
              className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm">
              <option value="all">All Labels</option>
              <option value="1">Raised Follow-on</option>
              <option value="0">No Follow-on</option>
            </select>
            <select value={filterSplit} onChange={e => { setFilterSplit(e.target.value); setPage(0); }}
              className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm">
              <option value="all">All Splits</option>
              <option value="train">Train</option>
              <option value="val">Validation</option>
              <option value="test">Test</option>
            </select>
            <select value={filterCountry} onChange={e => { setFilterCountry(e.target.value); setPage(0); }}
              className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm">
              <option value="all">All Countries</option>
              {countries.map(([code, count]) => (
                <option key={code} value={code}>{code} ({count})</option>
              ))}
            </select>
          </>
        )}
      </div>

      <div className="flex gap-4">
        {/* === Companies Tab === */}
        {tab === 'companies' && (
          <>
            <div className="flex-1 bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[var(--text-secondary)] border-b border-[var(--border)] text-xs">
                      {[
                        { key: 'name', label: 'Company' },
                        { key: 'category', label: 'Category' },
                        { key: 'country', label: 'Country' },
                        { key: 'triggerRound', label: 'Round' },
                        { key: 'triggerAmount', label: 'Amount' },
                        { key: 'label', label: 'Outcome' },
                      ].map(col => (
                        <th key={col.key} className="px-3 py-2 cursor-pointer hover:text-white" onClick={() => handleSort(col.key as keyof Company)}>
                          {col.label} {sortBy === col.key && (sortDir === 'asc' ? '▲' : '▼')}
                        </th>
                      ))}
                      <th className="px-3 py-2">Investors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(paged as Company[]).map((c, i) => (
                      <tr key={i} className="border-b border-[var(--border)]/30 hover:bg-[var(--surface-hover)] cursor-pointer" onClick={() => setSelected(c)}>
                        <td className="px-3 py-2 font-medium max-w-[180px] truncate">{c.name}</td>
                        <td className="px-3 py-2 text-[var(--text-secondary)] max-w-[120px] truncate">{c.category}</td>
                        <td className="px-3 py-2">{c.country}</td>
                        <td className="px-3 py-2">{c.triggerRound}</td>
                        <td className="px-3 py-2 font-mono">{c.triggerAmount ? `$${(c.triggerAmount / 1e6).toFixed(1)}M` : '—'}</td>
                        <td className="px-3 py-2">
                          <span className={`text-xs px-2 py-0.5 rounded-full ${c.label === 1 ? 'bg-[var(--success)]/20 text-[var(--success)]' : 'bg-[var(--investor)]/20 text-[var(--investor)]'}`}>
                            {c.label === 1 ? 'Raised' : 'No'}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-[var(--text-secondary)] max-w-[250px]">
                          <div className="flex flex-wrap gap-1">
                            {c.investors.slice(0, 3).map((inv, j) => (
                              <span key={j} className="text-xs bg-[var(--surface-hover)] rounded px-1.5 py-0.5 truncate max-w-[120px]">{inv}</span>
                            ))}
                            {c.investors.length > 3 && <span className="text-xs text-[var(--accent)]">+{c.investors.length - 3}</span>}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex justify-between items-center px-4 py-3 border-t border-[var(--border)] text-xs text-[var(--text-secondary)]">
                <span>Page {page + 1} of {totalPages} · {filtered.length.toLocaleString()} companies</span>
                <div className="flex gap-2">
                  <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0} className="px-2 py-1 rounded bg-[var(--surface-hover)] disabled:opacity-30">Prev</button>
                  <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} className="px-2 py-1 rounded bg-[var(--surface-hover)] disabled:opacity-30">Next</button>
                </div>
              </div>
            </div>

            {selected && (
              <div className="w-80 bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 h-fit flex-shrink-0">
                <div className="flex justify-between items-start mb-3">
                  <h3 className="font-semibold">{selected.name}</h3>
                  <button onClick={() => setSelected(null)} className="text-[var(--text-secondary)] hover:text-white text-sm">✕</button>
                </div>
                <div className="space-y-3 text-sm">
                  <div><span className="text-[var(--text-secondary)] text-xs">Category</span><p className="text-xs">{selected.category}</p></div>
                  <div className="flex justify-between"><span className="text-[var(--text-secondary)]">Country</span><span>{selected.country}</span></div>
                  <div className="flex justify-between"><span className="text-[var(--text-secondary)]">Status</span><span>{selected.status}</span></div>
                  <div className="flex justify-between"><span className="text-[var(--text-secondary)]">Total Funding</span><span className="font-mono">${(selected.fundingTotal / 1e6).toFixed(1)}M</span></div>
                  <div className="flex justify-between"><span className="text-[var(--text-secondary)]">Trigger Round</span><span>{selected.triggerRound}</span></div>
                  <div className="flex justify-between"><span className="text-[var(--text-secondary)]">Trigger Date</span><span>{selected.triggerDate?.split('T')[0]}</span></div>
                  <div className="flex justify-between">
                    <span className="text-[var(--text-secondary)]">Outcome</span>
                    <span className={selected.label === 1 ? 'text-[var(--success)]' : 'text-[var(--investor)]'}>
                      {selected.label === 1 ? 'Raised follow-on' : 'No follow-on'}
                    </span>
                  </div>
                  {selected.investors.length > 0 && (
                    <div>
                      <span className="text-[var(--text-secondary)] text-xs">Investors ({selected.investors.length})</span>
                      <div className="mt-1 max-h-40 overflow-y-auto">
                        {selected.investors.map((inv, i) => <p key={i} className="text-xs py-0.5">{inv}</p>)}
                      </div>
                    </div>
                  )}
                  <button onClick={() => buildEgoGraph(selected)} className="w-full mt-2 bg-[var(--accent)] text-white rounded-lg px-3 py-2 text-xs font-medium hover:opacity-90 transition-opacity">
                    Show Investor Network
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {/* === Features Tabs (Company & Investor) === */}
        {tab !== 'companies' && (
          <div className="flex-1 bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden">
            {!features ? (
              <div className="flex items-center justify-center h-40">
                <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
              </div>
            ) : (
              <>
                <div className="px-4 py-3 border-b border-[var(--border)] text-xs text-[var(--text-secondary)]">
                  {tab === 'companyFeatures'
                    ? 'Raw feature values used to build the 58-dim company vectors fed into XGBoost and GNN models. Numeric features are standardized (z-score) before training.'
                    : 'Raw investor features aggregated from Crunchbase investment records. Top 500 investors by deal count.'}
                </div>
                <div className="overflow-x-auto">
                  <table className="text-sm" style={{ minWidth: tab === 'companyFeatures' ? '900px' : '700px' }}>
                    <thead>
                      <tr className="text-left text-[var(--text-secondary)] border-b border-[var(--border)] text-xs">
                        {featCols.map(col => (
                          <th
                            key={col}
                            className="px-3 py-2 cursor-pointer hover:text-white whitespace-nowrap sticky top-0 bg-[var(--surface)]"
                            onClick={() => handleFeatSort(col)}
                          >
                            {col} {featSort.col === col && (featSort.dir === 'asc' ? '▲' : '▼')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(paged as Record<string, any>[]).map((row, i) => (
                        <tr key={i} className="border-b border-[var(--border)]/30 hover:bg-[var(--surface-hover)]">
                          {featCols.map(col => (
                            <td key={col} className={`px-3 py-1.5 text-xs whitespace-nowrap ${
                              col === 'name' ? 'font-medium max-w-[200px] truncate' :
                              col === 'label' ? (row[col] === 1 ? 'text-emerald-400' : 'text-rose-400') :
                              col === 'split' || col === 'country' || col === 'category' || col === 'trigger_round' ? 'text-[var(--text-secondary)]' :
                              'font-mono'
                            }`}>
                              {col === 'label' ? (row[col] === 1 ? 'Yes' : 'No') : formatVal(col, row[col])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex justify-between items-center px-4 py-3 border-t border-[var(--border)] text-xs text-[var(--text-secondary)]">
                  <span>Page {page + 1} of {totalPages} · {featRows.length.toLocaleString()} rows</span>
                  <div className="flex gap-2">
                    <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0} className="px-2 py-1 rounded bg-[var(--surface-hover)] disabled:opacity-30">Prev</button>
                    <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} className="px-2 py-1 rounded bg-[var(--surface-hover)] disabled:opacity-30">Next</button>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Ego graph modal */}
      {egoGraph && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setEgoGraph(null)}>
          <div className="bg-[var(--background)] rounded-xl border border-[var(--border)] w-[80vw] h-[75vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center px-5 py-3 border-b border-[var(--border)]">
              <div>
                <h3 className="font-semibold text-sm">{egoGraph.companyName} — Investor Network (2-hop)</h3>
                <p className="text-xs text-[var(--text-secondary)]">
                  {egoGraph.nodes.length} nodes · {egoGraph.links.length} edges
                  <span className="ml-3">
                    <span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />Root
                    <span className="inline-block w-2 h-2 rounded-full bg-rose-500 mr-1 ml-3" />Investor
                    <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1 ml-3" />Company
                  </span>
                </p>
              </div>
              <button onClick={() => setEgoGraph(null)} className="text-[var(--text-secondary)] hover:text-white text-lg px-2">✕</button>
            </div>
            <div className="flex-1" style={{ background: '#0a0f1a' }}>
              <ForceGraph2D
                ref={egoRef}
                graphData={{ nodes: egoGraph.nodes, links: egoGraph.links }}
                nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                  if (!isFinite(node.x) || !isFinite(node.y)) return;
                  const r = node.isRoot ? 10 : node.hop === 1 ? 6 : 3;
                  const color = node.isRoot ? '#fbbf24' : node.type === 'investor' ? '#f43f5e' : '#22d3ee';
                  if (node.isRoot) {
                    const glow = ctx.createRadialGradient(node.x, node.y, r, node.x, node.y, r * 3);
                    glow.addColorStop(0, 'rgba(251,191,36,0.4)');
                    glow.addColorStop(1, 'transparent');
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, r * 3, 0, Math.PI * 2);
                    ctx.fillStyle = glow;
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, r + 2, 0, Math.PI * 2);
                    ctx.strokeStyle = '#fbbf24';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                  }
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
                  ctx.fillStyle = color;
                  ctx.fill();
                  if ((node.isRoot || node.hop === 1) && globalScale > 0.5) {
                    const fontSize = Math.max(10 / globalScale, 3);
                    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    const label = node.label.length > 22 ? node.label.slice(0, 20) + '...' : node.label;
                    ctx.fillStyle = 'rgba(0,0,0,0.7)';
                    ctx.fillText(label, node.x + 0.5, node.y + r + 3.5);
                    ctx.fillStyle = node.isRoot ? '#fbbf24' : '#e2e8f0';
                    ctx.fillText(label, node.x, node.y + r + 3);
                  }
                }}
                nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
                  if (!isFinite(node.x) || !isFinite(node.y)) return;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, node.isRoot ? 12 : 8, 0, Math.PI * 2);
                  ctx.fillStyle = color;
                  ctx.fill();
                }}
                nodeLabel={(node: any) => `${node.label} (${node.type}${node.isRoot ? ', root' : ''}, hop ${node.hop})`}
                nodeRelSize={4}
                linkColor={() => 'rgba(255,255,255,0.2)'}
                linkWidth={0.5}
                backgroundColor="#0a0f1a"
                warmupTicks={50}
                cooldownTicks={150}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

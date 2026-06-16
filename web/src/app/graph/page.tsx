'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import type { GraphData } from '@/lib/types';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

export default function GraphPage() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] } | null>(null);
  const [fullData, setFullData] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphData['nodes'][0] | null>(null);
  const [colorMode, setColorMode] = useState<'type' | 'cluster'>('type');
  const [nodeCount, setNodeCount] = useState(100);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

  const clusterColors = ['#6366f1', '#f43f5e', '#22d3ee', '#f59e0b', '#a855f7', '#10b981', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6'];

  useEffect(() => {
    fetch('/data/graph_overview.json')
      .then(r => r.json())
      .then((data: GraphData) => setFullData(data))
      .catch(() => setFullData(null));
  }, []);

  useEffect(() => {
    if (!fullData) return;
    const half = Math.floor(nodeCount / 2);
    const investors = fullData.nodes.filter(n => n.type === 'investor').sort((a, b) => b.degree - a.degree).slice(0, half);
    const companies = fullData.nodes.filter(n => n.type === 'company').sort((a, b) => b.degree - a.degree).slice(0, half);
    const topNodes = [...investors, ...companies];
    const nodeIds = new Set(topNodes.map(n => n.id));
    const edges = fullData.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    setGraphData({ nodes: topNodes, links: edges });
    setSelectedNode(null);

    setTimeout(() => {
      fgRef.current?.zoomToFit(400, 40);
    }, 500);
  }, [fullData, nodeCount]);

  const neighborMap = useMemo(() => {
    if (!graphData) return new Map<string, Set<string>>();
    const map = new Map<string, Set<string>>();
    for (const link of graphData.links) {
      const sid = typeof link.source === 'object' ? link.source.id : link.source;
      const tid = typeof link.target === 'object' ? link.target.id : link.target;
      if (!map.has(sid)) map.set(sid, new Set());
      if (!map.has(tid)) map.set(tid, new Set());
      map.get(sid)!.add(tid);
      map.get(tid)!.add(sid);
    }
    return map;
  }, [graphData]);

  const highlightedNodes = useMemo(() => {
    if (!selectedNode) return new Set<string>();
    const set = new Set<string>();
    set.add(selectedNode.id);
    const neighbors = neighborMap.get(selectedNode.id);
    if (neighbors) neighbors.forEach(n => set.add(n));
    return set;
  }, [selectedNode, neighborMap]);

  if (!fullData) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">VC Investment Network</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          Top nodes by degree — investors (rose) + companies (cyan). Click a node to highlight neighbors.
        </p>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div className="bg-[var(--surface)] rounded-lg border border-[var(--border)] px-4 py-2 flex items-center gap-3">
          <label className="text-xs text-[var(--text-secondary)]">Color</label>
          <select
            value={colorMode}
            onChange={e => setColorMode(e.target.value as 'type' | 'cluster')}
            className="bg-[var(--surface-hover)] text-sm rounded px-2 py-1 border border-[var(--border)]"
          >
            <option value="type">Node Type</option>
            <option value="cluster">Cluster</option>
          </select>
        </div>
        <div className="bg-[var(--surface)] rounded-lg border border-[var(--border)] px-4 py-2 flex items-center gap-3">
          <label className="text-xs text-[var(--text-secondary)]">Nodes: {nodeCount}</label>
          <input
            type="range"
            min={30}
            max={300}
            step={10}
            value={nodeCount}
            onChange={e => setNodeCount(Number(e.target.value))}
            className="w-32 accent-[var(--accent)]"
          />
        </div>
        <div className="bg-[var(--surface)] rounded-lg border border-[var(--border)] px-4 py-2 text-xs text-[var(--text-secondary)]">
          {graphData?.nodes.length ?? 0} nodes · {graphData?.links.length ?? 0} edges
        </div>
        <div className="bg-[var(--surface)] rounded-lg border border-[var(--border)] px-4 py-2 flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            Investor
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
            Company
          </span>
        </div>
        <button
          onClick={() => fgRef.current?.zoomToFit(400, 40)}
          className="bg-[var(--surface)] rounded-lg border border-[var(--border)] px-4 py-2 text-xs text-[var(--text-secondary)] hover:text-white hover:border-[var(--accent)] transition-colors"
        >
          Fit view
        </button>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 rounded-xl border border-[var(--border)] overflow-hidden" style={{ height: '72vh', background: '#0a0f1a' }}>
          {graphData && (
            <ForceGraph2D
              ref={fgRef}
              graphData={graphData}
              nodeColor={(node: any) => {
                if (selectedNode && !highlightedNodes.has(node.id)) return '#334155';
                if (colorMode === 'type') return node.type === 'investor' ? '#f43f5e' : '#22d3ee';
                return clusterColors[node.cluster % clusterColors.length];
              }}
              nodeVal={(node: any) => {
                if (selectedNode?.id === node.id) return Math.sqrt(node.degree) * 3;
                if (selectedNode && highlightedNodes.has(node.id)) return Math.sqrt(node.degree) * 2;
                return Math.sqrt(node.degree) * 1.5;
              }}
              nodeLabel={(node: any) => `${node.label} (${node.type}, degree: ${node.degree})`}
              nodeRelSize={4}
              linkColor={(link: any) => {
                if (!selectedNode) return 'rgba(255,255,255,0.2)';
                const sid = typeof link.source === 'object' ? link.source.id : link.source;
                const tid = typeof link.target === 'object' ? link.target.id : link.target;
                if (sid === selectedNode.id || tid === selectedNode.id) return '#fbbf24';
                return 'rgba(255,255,255,0.04)';
              }}
              linkWidth={(link: any) => {
                if (!selectedNode) return 0.5;
                const sid = typeof link.source === 'object' ? link.source.id : link.source;
                const tid = typeof link.target === 'object' ? link.target.id : link.target;
                if (sid === selectedNode.id || tid === selectedNode.id) return 2;
                return 0.2;
              }}
              onNodeClick={(node: any) => {
                setSelectedNode(prev => prev?.id === node.id ? null : node);
              }}
              onBackgroundClick={() => setSelectedNode(null)}
              backgroundColor="#0a0f1a"
              warmupTicks={50}
              cooldownTicks={150}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
              enableNodeDrag={true}
            />
          )}
        </div>

        {selectedNode && (
          <div className="w-72 bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 h-fit shrink-0">
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-semibold text-sm leading-tight">{selectedNode.label}</h3>
              <button onClick={() => setSelectedNode(null)} className="text-[var(--text-secondary)] hover:text-white text-xs ml-2">✕</button>
            </div>
            <span className={`inline-block text-xs px-2 py-0.5 rounded-full ${
              selectedNode.type === 'investor' ? 'bg-rose-500/20 text-rose-400' : 'bg-cyan-500/20 text-cyan-400'
            }`}>
              {selectedNode.type}
            </span>
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Degree</span>
                <span className="font-mono">{selectedNode.degree}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Connections shown</span>
                <span className="font-mono">{highlightedNodes.size - 1}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Cluster</span>
                <span className="font-mono">{selectedNode.cluster}</span>
              </div>
              {selectedNode.category && (
                <div>
                  <span className="text-[var(--text-secondary)] text-xs">Category</span>
                  <p className="text-xs mt-0.5">{selectedNode.category}</p>
                </div>
              )}
              {selectedNode.country && (
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Country</span>
                  <span>{selectedNode.country}</span>
                </div>
              )}
              {selectedNode.hasLabel !== undefined && (
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Raised follow-on</span>
                  <span className={selectedNode.prediction === 1 ? 'text-emerald-400' : 'text-rose-400'}>
                    {selectedNode.prediction === 1 ? 'Yes' : 'No'}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

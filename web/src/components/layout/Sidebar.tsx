'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  { href: '/', label: 'Overview', icon: '◉' },
  { href: '/graph', label: 'Network Graph', icon: '◎' },
  { href: '/models', label: 'Model Comparison', icon: '▣' },
  { href: '/explorer', label: 'Dataset Explorer', icon: '⊞' },
  { href: '/ablation', label: 'Ablation Studies', icon: '⊟' },
  { href: '/chat', label: 'Ask AI', icon: '◈' },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-56 bg-[var(--surface)] border-r border-[var(--border)] flex flex-col z-50">
      <div className="p-4 border-b border-[var(--border)]">
        <h1 className="text-sm font-bold text-[var(--accent)] tracking-wide">VC NETWORK</h1>
        <p className="text-xs text-[var(--text-secondary)] mt-0.5">GNN Research Explorer</p>
      </div>
      <nav className="flex-1 py-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-r-2 border-[var(--accent)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]'
              }`}
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-[var(--border)] text-xs text-[var(--text-secondary)]">
        Crunchbase 2015 Dataset
      </div>
    </aside>
  );
}

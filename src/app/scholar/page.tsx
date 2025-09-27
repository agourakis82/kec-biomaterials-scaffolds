'use client';
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

type SearchMode = 'doi' | 'arxiv' | 'terms';

interface ResultItem {
  title: string;
  authors?: string[];
  year?: number;
  doi?: string;
  arxivId?: string;
  citations?: number;
  abstract?: string;
}

export default function Page() {
  const [mode, setMode] = useState<SearchMode>('doi');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [analysis, setAnalysis] = useState<any | null>(null);

  async function search() {
    setErr(null);
    setLoading(true);
    setAnalysis(null);
    try {
      if (!query.trim()) throw new Error('Informe um termo/DOI/consulta.');
      if (mode === 'doi') {
        const item = await fetchCrossrefByDOI(query.trim());
        setResults(item ? [item] : []);
      } else if (mode === 'arxiv') {
        const items = await fetchArxiv(query.trim());
        setResults(items);
      } else {
        const items = await fetchCrossrefByTerms(query.trim());
        setResults(items);
      }
    } catch (e:any) {
      setErr(e.message || 'Falha na busca');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function analyze(item: ResultItem) {
    setErr(null);
    setLoading(true);
    try {
      const doc = `\\title{${item.title}}
\\begin{abstract}
${item.abstract || 'Resumo não disponível. Texto sintético para análise.'}
\\end{abstract}`;
      const res = await fetch(`${apiBase}/q1-scholar/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document: doc, include_ai_analysis: true }),
      });
      if (!res.ok) throw new Error(`Backend ${res.status}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (e:any) {
      setErr(e.message || 'Falha na análise');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }

  function toBibTeX(item: ResultItem) {
    const key = (item.authors?.[0]?.split(' ').slice(-1)[0] || 'key') + (item.year || 'xxxx');
    const author = item.authors?.join(' and ') || 'Unknown';
    return `@article{${key},
  title={${item.title}},
  author={${author}},
  year={${item.year || ''}},
  doi={${item.doi || ''}},
  eprint={${item.arxivId || ''}}
}`;
  }

  async function copyBibtex(item: ResultItem) {
    try {
      await navigator.clipboard.writeText(toBibTeX(item));
    } catch {}
  }

  return (
    <div className="p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold">Q1 Scholar</h1>
        <p className="text-sm text-gray-500">Busca DOI/ArXiv/termos • BibTeX • Análise Q1 • API: {apiBase}</p>
      </header>

      <section className="flex items-center gap-3">
        <select
          className="border rounded px-2 py-1"
          value={mode}
          onChange={(e) => setMode(e.target.value as SearchMode)}
        >
          <option value="doi">DOI</option>
          <option value="arxiv">ArXiv</option>
          <option value="terms">Termos</option>
        </select>
        <input
          className="flex-1 border rounded px-3 py-2"
          placeholder={mode === 'doi' ? '10.xxxx/xxxx' : mode === 'arxiv' ? 'graphene biomaterials' : 'biomaterials porosity'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="bg-black text-white px-4 py-2 rounded disabled:opacity-60"
          onClick={search}
          disabled={loading}
        >
          {loading ? 'Buscando...' : 'Buscar'}
        </button>
      </section>

      {err && <div className="text-red-600 text-sm">{err}</div>}

      <section className="grid gap-4">
        {results.map((r, idx) => (
          <article key={idx} className="border rounded p-4 space-y-2">
            <div className="font-medium">{r.title}</div>
            <div className="text-sm text-gray-600">
              {(r.authors || []).join(', ')} {r.year ? `• ${r.year}` : ''} {r.doi ? `• DOI ${r.doi}` : ''} {r.arxivId ? `• arXiv:${r.arxivId}` : ''}
            </div>
            <div className="flex gap-2">
              <button className="border px-3 py-1 rounded" onClick={() => copyBibtex(r)}>Copiar BibTeX</button>
              <button className="border px-3 py-1 rounded" onClick={() => analyze(r)} disabled={loading}>Analisar no Backend</button>
            </div>
            <div className="h-24">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[{ name: 'Citações', value: r.citations || 0 }]}>
                  <XAxis dataKey="name" hide />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
        ))}
        {!loading && results.length === 0 && <div className="text-sm text-gray-500">Nenhum resultado.</div>}
      </section>

      {analysis && (
        <section className="border rounded p-4">
          <h2 className="font-semibold mb-2">Relatório de Análise</h2>
          <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(analysis, null, 2)}</pre>
        </section>
      )}
    </div>
  );
}

async function fetchCrossrefByDOI(doi: string): Promise<ResultItem | null> {
  const url = `https://api.crossref.org/works/${encodeURIComponent(doi)}`;
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) return null;
  const j = await res.json();
  const m = j?.message;
  return {
    title: (m?.title && m.title[0]) || 'Sem título',
    authors: (m?.author || []).map((a:any) => [a.given, a.family].filter(Boolean).join(' ')),
    year: m?.issued?.['date-parts']?.[0]?.[0],
    doi: m?.DOI,
    citations: m?.['is-referenced-by-count'] || 0,
    abstract: m?.abstract ? stripTags(m.abstract) : undefined,
  };
}

async function fetchCrossrefByTerms(q: string): Promise<ResultItem[]> {
  const url = `https://api.crossref.org/works?query=${encodeURIComponent(q)}&rows=10`;
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) return [];
  const j = await res.json();
  const items = j?.message?.items || [];
  return items.map((m:any) => ({
    title: (m?.title && m.title[0]) || 'Sem título',
    authors: (m?.author || []).map((a:any) => [a.given, a.family].filter(Boolean).join(' ')),
    year: m?.issued?.['date-parts']?.[0]?.[0],
    doi: m?.DOI,
    citations: m?.['is-referenced-by-count'] || 0,
    abstract: m?.abstract ? stripTags(m.abstract) : undefined,
  }));
}

async function fetchArxiv(q: string): Promise<ResultItem[]> {
  const url = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(q)}&start=0&max_results=10`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const xml = await res.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'application/xml');
  const entries = Array.from(doc.getElementsByTagName('entry'));
  return entries.map((e) => {
    const title = e.getElementsByTagName('title')[0]?.textContent?.trim() || 'Sem título';
    const authors = Array.from(e.getElementsByTagName('author')).map((a) => a.getElementsByTagName('name')[0]?.textContent || '').filter(Boolean);
    const id = e.getElementsByTagName('id')[0]?.textContent || '';
    const arxivId = id.split('/abs/')[1] || id;
    const summary = e.getElementsByTagName('summary')[0]?.textContent || undefined;
    const year = (() => {
      const published = e.getElementsByTagName('published')[0]?.textContent || '';
      return published ? new Date(published).getFullYear() : undefined;
    })();
    return { title, authors, year, arxivId, abstract: summary };
  });
}

function stripTags(s: string) {
  return s.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
}

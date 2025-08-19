import React, { useEffect, useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

type Mapping = {
  desc_norm: string
  account_name: string
  is_travel: boolean
  rationale: string
}

type MatchRow = {
  desc_norm: string
  proposed_account_name: string
  is_travel: boolean
  rationale: string
  match_rank: number
  match_score: number
  coa_account_name: string
  coa_type?: string
  coa_description?: string
}

type TxnRow = {
  Desc_norm: string
  desc_norm_normed?: string | null
  best_account_name?: string | null
  is_travel?: boolean | null
  rationale?: string | null
  not_sure?: boolean | null
  guessed_store?: string | null
}

export function App() {
  const [loading, setLoading] = useState(false)
  const [loadingLLM, setLoadingLLM] = useState(false)
  const [loadingRec, setLoadingRec] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [limit, setLimit] = useState<number>(20)
  const [descriptions, setDescriptions] = useState<string[]>([])
  const [mappings, setMappings] = useState<Mapping[]>([])
  const [txns, setTxns] = useState<TxnRow[]>([])
  const [recCounts, setRecCounts] = useState<{ matched: number; missing_in_gl: number; missing_in_bank: number } | null>(null)
  // Part 1 datasets (full, not just previews)
  const [matchedData, setMatchedData] = useState<any[]>([])
  const [missingInGlData, setMissingInGlData] = useState<any[]>([])
  const [missingInBankData, setMissingInBankData] = useState<any[]>([])
  // Active dataset tab for Part 1 panel
  const [recActiveTab, setRecActiveTab] = useState<'missing_in_gl' | 'matched' | 'missing_in_bank'>('missing_in_gl')
  // Sliding window state
  const [windowStart, setWindowStart] = useState<number>(0)
  const [windowSize, setWindowSize] = useState<number>(50)

  const runReconcile = async () => {
    setLoadingRec(true)
    setError(null)
    try {
      const rresp = await fetch(`${API_BASE}/reconcile`, { method: 'POST' })
      if (!rresp.ok) throw new Error(`HTTP ${rresp.status}`)
      const rdata = await rresp.json()
      setRecCounts(rdata.counts)
      setMatchedData(rdata.matched ?? [])
      setMissingInGlData(rdata.missing_in_gl ?? [])
      setMissingInBankData(rdata.missing_in_bank ?? [])
      setRecActiveTab('missing_in_gl')
      setWindowStart(0)
    } catch (e: any) {
      setError(e?.message ?? 'Failed to reconcile')
    } finally {
      setLoadingRec(false)
    }
  }

  const runCategorization = async () => {
    setLoading(true)
    setLoadingLLM(false)
    setError(null)
    // Clear previous results so the UI reflects a fresh run
    setMappings([])
    setTxns([])
    try {
      // Phase 1: fetch descriptions quickly
      const dresp = await fetch(`${API_BASE}/descriptions?limit_keywords=${encodeURIComponent(String(limit))}`)
      if (!dresp.ok) throw new Error(`HTTP ${dresp.status}`)
      const ddata = await dresp.json()
      setDescriptions(ddata.descriptions ?? [])

      // Phase 2: call the categorization endpoint (LLM involved)
      setLoadingLLM(true)
      const resp = await fetch(`${API_BASE}/categorize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit_keywords: limit })
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setMappings(data.llm_mappings ?? [])
      setTxns(data.transactions_with_account ?? [])
    } catch (e: any) {
      setError(e?.message ?? 'Failed to fetch')
    } finally {
      setLoading(false)
      setLoadingLLM(false)
    }
  }

  const txnsCount = txns.length

  // Clamp window when active dataset or size changes
  useEffect(() => {
    setWindowStart(0)
  }, [recActiveTab])

  const activeDatasetLabel = useMemo(() => {
    if (recActiveTab === 'missing_in_gl') return 'Missing in GL'
    if (recActiveTab === 'matched') return 'Matched'
    return 'Missing in Bank'
  }, [recActiveTab])

  // Normalize rows for display into Date_norm, Desc_norm, Amount_norm
  const activeDisplayRows = useMemo(() => {
    let rows: any[] = []
    if (recActiveTab === 'missing_in_gl') rows = missingInGlData
    else if (recActiveTab === 'missing_in_bank') rows = missingInBankData
    else rows = matchedData

    if (recActiveTab === 'matched') {
      return rows.map((r: any) => {
        const parts = String(r.match_key ?? '').split('|')
        const [dateNorm, descNorm, amountNorm] = [parts[0] ?? '', parts[1] ?? '', parts[2] ?? '']
        return { Date_norm: dateNorm, Desc_norm: descNorm, Amount_norm: amountNorm }
      })
    }
    return rows.map((r: any) => ({
      Date_norm: String(r.Date_norm ?? ''),
      Desc_norm: String(r.Desc_norm ?? ''),
      Amount_norm: String(r.Amount_norm ?? '')
    }))
  }, [recActiveTab, matchedData, missingInBankData, missingInGlData])

  const activeTotal = activeDisplayRows.length
  const visibleRows = useMemo(() => activeDisplayRows.slice(windowStart, windowStart + windowSize), [activeDisplayRows, windowStart, windowSize])

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h1>Mesh Categorization</h1>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
        <label>
          Keyword limit:
          <input type="number" value={limit} onChange={e => setLimit(parseInt(e.target.value || '0', 10))} style={{ marginLeft: 8, width: 100 }} />
        </label>
        <button onClick={runReconcile} disabled={loadingRec || loading || loadingLLM}>
          {loadingRec ? 'Reconciling…' : 'Run reconciliation'}
        </button>
        <button onClick={runCategorization} disabled={loading || loadingLLM}>
          {loadingLLM ? 'Categorizing…' : (loading ? 'Preparing…' : 'Run categorization')}
        </button>
        {error && <span style={{ color: 'crimson' }}>{error}</span>}
      </div>

      {recCounts && (
        <section style={{ marginBottom: 24 }}>
          <h2>Part 1 — Reconciliation</h2>
          <div style={{ display: 'flex', gap: 24, marginBottom: 12 }}>
            <div><strong>Matched</strong>: {recCounts.matched}</div>
            <div><strong>Missing in GL (Bank-only)</strong>: {recCounts.missing_in_gl}</div>
            <div><strong>Missing in Bank (GL-only)</strong>: {recCounts.missing_in_bank}</div>
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            {[
              { key: 'missing_in_gl', label: 'Missing in GL' },
              { key: 'matched', label: 'Matched' },
              { key: 'missing_in_bank', label: 'Missing in Bank' },
            ].map((tab: any) => (
              <button
                key={tab.key}
                onClick={() => setRecActiveTab(tab.key)}
                style={{
                  padding: '6px 10px',
                  borderRadius: 6,
                  border: recActiveTab === tab.key ? '2px solid #444' : '1px solid #ccc',
                  background: recActiveTab === tab.key ? '#f2f2f2' : '#fff',
                  cursor: 'pointer'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
            <strong>{activeDatasetLabel}</strong>
            <button onClick={() => setWindowStart(Math.max(0, windowStart - windowSize))} disabled={windowStart === 0}>Prev</button>
            <span>
              {activeTotal === 0 ? '0–0 of 0' : `${windowStart + 1}–${Math.min(activeTotal, windowStart + windowSize)} of ${activeTotal}`}
            </span>
            <button onClick={() => setWindowStart(Math.min(Math.max(0, activeTotal - windowSize), windowStart + windowSize))} disabled={windowStart + windowSize >= activeTotal}>Next</button>
            <label style={{ marginLeft: 12 }}>
              Window size:
              <input
                type="number"
                value={windowSize}
                min={1}
                onChange={e => setWindowSize(Math.max(1, parseInt(e.target.value || '1', 10)))}
                style={{ marginLeft: 6, width: 80 }}
              />
            </label>
          </div>
          <div style={{ overflowX: 'auto', border: '1px solid #eee', borderRadius: 8, padding: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>Date_norm</th>
                  <th>Desc_norm</th>
                  <th>Amount_norm</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((r: any, i: number) => (
                  <tr key={i}>
                    <td>{r.Date_norm}</td>
                    <td>{r.Desc_norm}</td>
                    <td>{r.Amount_norm}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section style={{ marginBottom: 24 }}>
        <h2>Descriptions ({descriptions.length})</h2>
        {loading && !loadingLLM && <div style={{ marginBottom: 8 }}>Loading descriptions…</div>}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {descriptions.slice(0, 50).map((d, i) => (
            <span key={i} style={{ border: '1px solid #ddd', borderRadius: 6, padding: '2px 8px' }}>{d}</span>
          ))}
          {descriptions.length > 50 && <span>…</span>}
        </div>
      </section>

      

      <section>
        <h2>Transactions (all matches) — {txnsCount}</h2>
        {loadingLLM && <div style={{ marginBottom: 8 }}>Loading categorization…</div>}
        <div style={{ overflowX: 'auto', border: '1px solid #eee', borderRadius: 8, padding: 8 }}>
          <table>
            <thead>
              <tr>
                <th>Desc_norm</th>
                <th>desc_norm_normed</th>
                <th>best_account_name</th>
                <th>is_travel</th>
                <th>not_sure</th>
                <th>guessed_store</th>
                <th>rationale</th>
              </tr>
            </thead>
            <tbody>
              {txns.map((t, i) => (
                <tr key={i}>
                  <td>{t.Desc_norm}</td>
                  <td>{t.desc_norm_normed ?? ''}</td>
                  <td>{t.best_account_name ?? ''}</td>
                  <td>{t.is_travel === undefined || t.is_travel === null ? '' : (t.is_travel ? 'Yes' : 'No')}</td>
                  <td>{t.not_sure === undefined || t.not_sure === null ? '' : (t.not_sure ? 'Yes' : 'No')}</td>
                  <td>{t.guessed_store ?? ''}</td>
                  <td style={{ maxWidth: 520 }}>{t.rationale ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}



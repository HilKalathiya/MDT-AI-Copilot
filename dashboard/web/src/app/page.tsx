"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend, ReferenceLine, ScatterChart, Scatter, ZAxis
} from "recharts";

// ── Types ──────────────────────────────────────────────────────────────────

interface Stats {
  ue_samples: number;
  mdt_reports: number;
  avg_rsrp_dbm: number;
  unique_ues: number;
  data_source: string;
}

interface Sample {
  id: number;
  ue_rrc_id: number;
  rsrp_dbm: number;
  delta_rsrp_db: number;
  reason: string;
  is_injected_anomaly: number;
  logged_at: string;
  nid_cell: number;
}

interface AnomalyResult {
  total_samples: number;
  anomaly_count: number;
  true_positives: number;
  false_positives: number;
  anomaly_rate: number;
  anomalies: Array<{
    ue_rrc_id: number;
    rsrp_dbm: number;
    z_score: number;
    reason: string;
    is_injected_anomaly: number;
    logged_at: string;
  }>;
}

interface Cell {
  cell_id: number;
  sample_count: number;
  unique_ues: number;
  avg_rsrp_dbm: number;
  min_rsrp_dbm: number;
  max_rsrp_dbm: number;
  avg_delta_db: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

// ── API helpers ────────────────────────────────────────────────────────────

const API_BASE = "/api";

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${await res.text()}`);
  return res.json();
}

async function apiPost<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${await res.text()}`);
  return res.json();
}

// ── Helpers ────────────────────────────────────────────────────────────────

function getRsrpHealth(rsrp: number): { label: string; cls: string } {
  if (rsrp > -90) return { label: "Good", cls: "badge-green" };
  if (rsrp > -100) return { label: "Fair", cls: "badge-amber" };
  return { label: "Poor", cls: "badge-red" };
}

function formatRsrp(v: number) { return `${v?.toFixed(1)} dBm`; }

const COLORS = ["#3b82f6", "#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#fb923c", "#e879f9", "#4ade80", "#f472b6"];

const REASON_COLORS: Record<string, string> = {
  periodic: "#3b82f6",
  meas_update: "#22d3ee",
  rsrp_drop: "#fbbf24",
  low_rsrp: "#f87171",
  enable: "#a78bfa",
  none: "#475569",
};

// ── Navbar ─────────────────────────────────────────────────────────────────

function Navbar({ connected }: { connected: boolean }) {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <a href="#" className="navbar-brand">
          <div className="logo">📡</div>
          <span>MDT AI Copilot</span>
          <span className="badge badge-purple" style={{ fontSize: "0.65rem" }}>5G RAN</span>
        </a>
        <div className="flex items-center gap-4">
          <span className="text-xs text-muted">OpenAirInterface MDT Analytics</span>
          <span className={`badge ${connected ? "badge-green" : "badge-red"}`}>
            {connected && <span className="pulse-indicator"></span>} {connected ? "Connected" : "No Data"}
          </span>
        </div>
      </div>
    </nav>
  );
}

// ── Metric Card ────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, accent }: {
  label: string; value: string | number; sub?: string; accent?: string;
}) {
  return (
    <div className="metric-card fade-in">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color: accent || "var(--text-primary)" }}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

// ── Custom tooltip ─────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card" style={{ padding: "1rem", minWidth: 180, border: "1px solid var(--border-hover)", boxShadow: "var(--shadow-card)" }}>
      <p className="text-xs text-muted mb-3 font-mono">{label}</p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center justify-between gap-4 mt-2">
          <span style={{ color: p.color, fontSize: "0.875rem", fontWeight: 500 }}>{p.name}</span>
          <strong style={{ color: "var(--text-primary)", fontSize: "0.875rem" }}>
            {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
          </strong>
        </div>
      ))}
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function Home() {
  const [activeTab, setActiveTab] = useState("overview");
  const [stats, setStats] = useState<Stats | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [anomalyData, setAnomalyData] = useState<AnomalyResult | null>(null);
  const [cells, setCells] = useState<Cell[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [selectedUe, setSelectedUe] = useState<number>(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, smp, anom, c] = await Promise.all([
        apiFetch<Stats>("/stats"),
        apiFetch<{ count: number; rows: Sample[] }>("/samples?limit=500"),
        apiFetch<AnomalyResult>("/anomalies"),
        apiFetch<{ cells: Cell[] }>("/cells"),
      ]);
      setStats(s);
      setSamples(smp.rows);
      setAnomalyData(anom);
      setCells(c.cells);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleInitDb = async () => {
    setLoading(true);
    try {
      await apiPost("/init", {});
      await loadData();
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const sendChat = async (question: string) => {
    if (!question.trim() || chatLoading) return;
    const userMsg: ChatMessage = { role: "user", content: question, timestamp: new Date() };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await apiPost<{ question: string; answer: string }>("/chat", { question });
      setChatMessages(prev => [...prev, { role: "assistant", content: res.answer, timestamp: new Date() }]);
    } catch (e: any) {
      setChatMessages(prev => [...prev, { role: "assistant", content: `⚠️ Error: ${e.message}`, timestamp: new Date() }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Prepare chart data
  const ueIds = [...new Set(samples.map(s => s.ue_rrc_id))].sort();
  const ueData = samples
    .filter(s => s.ue_rrc_id === selectedUe)
    .slice(-200)
    .map(s => ({
      time: s.logged_at.slice(11, 19),
      rsrp: s.rsrp_dbm,
      anomaly: anomalyData?.anomalies.some(a => a.ue_rrc_id === s.ue_rrc_id && a.logged_at === s.logged_at),
    }));

  const reasonData = Object.entries(
    samples.reduce((acc, s) => { acc[s.reason] = (acc[s.reason] || 0) + 1; return acc; }, {} as Record<string, number>)
  ).map(([name, value]) => ({ name, value }));

  const tabs = [
    { id: "overview", label: "📊 Overview" },
    { id: "rsrp", label: "📈 RSRP Trends" },
    { id: "anomalies", label: "⚠️ Anomalies" },
    { id: "cells", label: "🗼 Cells" },
    { id: "copilot", label: "🤖 AI Copilot" },
  ];

  const hasData = (stats?.ue_samples || 0) > 0;

  return (
    <div>
      <Navbar connected={hasData} />

      <main className="container" style={{ paddingTop: "2rem", paddingBottom: "4rem" }}>
        {/* Page header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 style={{ background: "var(--gradient-primary)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", fontSize: "1.75rem" }}>
              MDT Coverage Analytics
            </h1>
            <p className="text-sm text-muted mt-4">
              5G Minimization of Drive Tests — Agentic AI Copilot with LangGraph
            </p>
          </div>
          <div className="flex gap-2">
            <button className="btn btn-ghost" onClick={loadData} disabled={loading}>
              {loading ? <span className="spinner">↻</span> : "↻"} Refresh
            </button>
            {!hasData && (
              <button className="btn btn-primary" onClick={handleInitDb} disabled={loading}>
                ⚙️ Generate Data
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="card mb-4" style={{ borderColor: "var(--accent-red)", color: "var(--accent-red)" }}>
            ⚠️ {error} — Make sure the FastAPI backend is running on port 8000.
          </div>
        )}

        {/* Tabs */}
        <div className="tabs mb-6" style={{ overflowX: "auto" }}>
          {tabs.map(t => (
            <button key={t.id} className={`tab ${activeTab === t.id ? "active" : ""}`} onClick={() => setActiveTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>

        {/* ── OVERVIEW TAB ── */}
        {activeTab === "overview" && (
          <div className="fade-in">
            <div className="grid-4 mb-6">
              <MetricCard label="UE Samples" value={stats ? stats.ue_samples.toLocaleString() : "—"} sub="in ue_samples table" />
              <MetricCard label="Active UEs" value={stats?.unique_ues ?? "—"} sub="distinct UE IDs" accent="var(--accent-cyan)" />
              <MetricCard label="Avg RSRP" value={stats ? formatRsrp(stats.avg_rsrp_dbm) : "—"} sub="across all cells" accent={stats && stats.avg_rsrp_dbm > -90 ? "var(--accent-green)" : "var(--accent-amber)"} />
              <MetricCard label="Anomalies" value={anomalyData?.anomaly_count.toLocaleString() ?? "—"}
                sub={anomalyData ? `${(anomalyData.anomaly_rate * 100).toFixed(1)}% rate` : ""}
                accent={anomalyData && anomalyData.anomaly_rate > 0.1 ? "var(--accent-red)" : "var(--accent-green)"} />
            </div>

            <div className="grid-2">
              <div className="card">
                <h3 className="mb-4">RSRP Distribution</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={
                    samples.reduce((acc, s) => {
                      const bucket = Math.floor(s.rsrp_dbm / 5) * 5;
                      const existing = acc.find(a => a.rsrp === bucket);
                      if (existing) existing.count++; else acc.push({ rsrp: bucket, count: 1 });
                      return acc;
                    }, [] as { rsrp: number; count: number }[]).sort((a, b) => a.rsrp - b.rsrp)
                  }>
                    <defs>
                      <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="var(--accent-indigo)" stopOpacity={0.8}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} />
                    <XAxis dataKey="rsrp" stroke="var(--text-muted)" tick={{ fontSize: 11 }} tickFormatter={v => `${v}`} />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine x={-100} stroke="var(--accent-rose)" strokeDasharray="4 4" label={{ value: "−100", fill: "var(--accent-rose)", fontSize: 10 }} />
                    <Bar dataKey="count" fill="url(#colorCount)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 className="mb-4">Trigger Reason Distribution</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={reasonData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={3}>
                      {reasonData.map((entry) => (
                        <Cell key={entry.name} fill={REASON_COLORS[entry.name] || "#475569"} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: "0.75rem", color: "var(--text-secondary)" }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ── RSRP TRENDS TAB ── */}
        {activeTab === "rsrp" && (
          <div className="fade-in">
            <div className="card mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3>RSRP Time Series</h3>
                <div className="flex items-center gap-4">
                  <label className="text-sm text-muted">UE:</label>
                  <select value={selectedUe} onChange={e => setSelectedUe(Number(e.target.value))}
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)", borderRadius: "var(--radius-sm)", padding: "0.375rem 0.75rem", fontSize: "0.875rem" }}>
                    {ueIds.map(id => <option key={id} value={id}>UE {id}</option>)}
                  </select>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={ueData}>
                  <defs>
                    <linearGradient id="colorRsrp" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--accent-indigo)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} />
                  <XAxis dataKey="time" stroke="var(--text-muted)" tick={{ fontSize: 10 }} interval={Math.floor(ueData.length / 8)} />
                  <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={-100} stroke="var(--accent-rose)" strokeDasharray="4 4" label={{ value: "−100 dBm", fill: "var(--accent-rose)", fontSize: 10 }} />
                  <ReferenceLine y={-90} stroke="var(--accent-amber)" strokeDasharray="4 4" label={{ value: "−90 dBm", fill: "var(--accent-amber)", fontSize: 10 }} />
                  <Area type="monotone" dataKey="rsrp" stroke="var(--accent-cyan)" strokeWidth={2} fillOpacity={1} fill="url(#colorRsrp)" name="RSRP (dBm)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3 className="mb-4">Per-UE RSRP Heatmap</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "0.75rem" }}>
                {cells.map(cell => {
                  const health = getRsrpHealth(cell.avg_rsrp_dbm);
                  return (
                    <div key={cell.cell_id} className="card" style={{ padding: "1rem" }}>
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-sm font-mono text-muted">Cell {cell.cell_id}</span>
                        <span className={`badge ${health.cls}`}>{health.label}</span>
                      </div>
                      <div style={{ fontSize: "1.5rem", fontWeight: 800, color: cell.avg_rsrp_dbm > -90 ? "var(--accent-green)" : cell.avg_rsrp_dbm > -100 ? "var(--accent-amber)" : "var(--accent-red)" }}>
                        {cell.avg_rsrp_dbm.toFixed(1)}
                      </div>
                      <div className="text-xs text-muted mt-4">dBm avg · {cell.sample_count.toLocaleString()} samples</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── ANOMALIES TAB ── */}
        {activeTab === "anomalies" && (
          <div className="fade-in">
            <div className="grid-4 mb-6">
              <MetricCard label="Flagged Samples" value={anomalyData?.anomaly_count.toLocaleString() ?? "—"} accent="var(--accent-red)" />
              <MetricCard label="True Positives" value={anomalyData?.true_positives.toLocaleString() ?? "—"} sub="matched ground truth" accent="var(--accent-green)" />
              <MetricCard label="False Positives" value={anomalyData?.false_positives.toLocaleString() ?? "—"} accent="var(--accent-amber)" />
              <MetricCard label="Anomaly Rate" value={anomalyData ? `${(anomalyData.anomaly_rate * 100).toFixed(1)}%` : "—"} sub="of all samples" />
            </div>

            <div className="card">
              <h3 className="mb-4">⚠️ Anomalous Samples</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>UE ID</th>
                      <th>RSRP (dBm)</th>
                      <th>Z-Score</th>
                      <th>Trigger Reason</th>
                      <th>Ground Truth</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(anomalyData?.anomalies ?? []).slice(0, 100).map((a, i) => (
                      <tr key={i}>
                        <td><span className="badge badge-blue">UE {a.ue_rrc_id}</span></td>
                        <td><span className={a.rsrp_dbm <= -100 ? "text-red" : a.rsrp_dbm <= -90 ? "text-amber" : "text-green"}>{a.rsrp_dbm} dBm</span></td>
                        <td><span className="font-mono text-amber">{a.z_score?.toFixed(2)}</span></td>
                        <td><span className="badge" style={{ background: `${REASON_COLORS[a.reason]}20`, color: REASON_COLORS[a.reason], border: `1px solid ${REASON_COLORS[a.reason]}40` }}>{a.reason}</span></td>
                        <td>{a.is_injected_anomaly ? <span className="badge badge-red">injected</span> : <span className="badge badge-green">normal</span>}</td>
                        <td className="text-xs text-muted font-mono">{a.logged_at?.slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(anomalyData?.anomaly_count ?? 0) > 100 && (
                <p className="text-xs text-muted mt-4">Showing 100 of {anomalyData?.anomaly_count} anomalies</p>
              )}
            </div>
          </div>
        )}

        {/* ── CELLS TAB ── */}
        {activeTab === "cells" && (
          <div className="fade-in">
            <div className="card mb-6">
              <h3 className="mb-4">🗼 Cell Health Overview</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={cells} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} horizontal={false} />
                  <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} domain={[-120, -60]} />
                  <YAxis dataKey="cell_id" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} tickFormatter={v => `Cell ${v}`} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine x={-100} stroke="var(--accent-rose)" strokeDasharray="4 4" />
                  <ReferenceLine x={-90} stroke="var(--accent-amber)" strokeDasharray="4 4" />
                  <Bar dataKey="avg_rsrp_dbm" name="Avg RSRP (dBm)" radius={[0, 4, 4, 0]}>
                    {cells.map((cell, i) => (
                      <Cell key={i} fill={cell.avg_rsrp_dbm > -90 ? "var(--accent-teal)" : cell.avg_rsrp_dbm > -100 ? "var(--accent-amber)" : "var(--accent-rose)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Cell ID</th>
                      <th>Health</th>
                      <th>Samples</th>
                      <th>UEs</th>
                      <th>Avg RSRP</th>
                      <th>Min RSRP</th>
                      <th>Avg Δ RSRP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cells.map(cell => {
                      const health = getRsrpHealth(cell.avg_rsrp_dbm);
                      return (
                        <tr key={cell.cell_id}>
                          <td><span className="font-mono">Cell {cell.cell_id}</span></td>
                          <td><span className={`badge ${health.cls}`}>● {health.label}</span></td>
                          <td>{cell.sample_count.toLocaleString()}</td>
                          <td>{cell.unique_ues}</td>
                          <td><span className={health.cls === "badge-green" ? "text-green" : health.cls === "badge-amber" ? "text-amber" : "text-red"}>{cell.avg_rsrp_dbm.toFixed(1)} dBm</span></td>
                          <td className="font-mono">{cell.min_rsrp_dbm} dBm</td>
                          <td className="font-mono">{cell.avg_delta_db?.toFixed(2)} dB</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── COPILOT TAB ── */}
        {activeTab === "copilot" && (
          <div className="fade-in">
              <div className="card" style={{ maxHeight: 700 }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span style={{ fontSize: "1.5rem" }}>🤖</span> MDT AI Copilot
                  </h3>
                  <p className="text-xs text-muted mt-2">LangGraph ReAct agent • RAG Retrieval</p>
                </div>
                <span className="badge badge-purple" style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem" }}>Cohere Command-R Plus</span>
              </div>

              {/* Example prompts */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
                {[
                  "How many MDT samples in cell 1?",
                  "Which UEs show anomalous RSRP?",
                  "Why use a ring buffer?",
                  "What's the rsrp_drop threshold?",
                  "Cell 1 anomalies — expected given MDT triggers?",
                ].map(q => (
                  <button key={q} className="btn btn-ghost" style={{ fontSize: "0.7rem", padding: "0.3rem 0.75rem" }}
                    onClick={() => sendChat(q)}>
                    💬 {q}
                  </button>
                ))}
              </div>

              <div className="chat-container">
                <div className="chat-messages">
                  {chatMessages.length === 0 && (
                    <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "2rem" }}>
                      <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>📡</div>
                      <p className="text-sm">Ask the copilot anything about your MDT data, the system architecture, or 5G concepts.</p>
                    </div>
                  )}
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`chat-message ${msg.role}`}>
                      <div className={`chat-avatar ${msg.role}`}>
                        {msg.role === "user" ? "👤" : "🤖"}
                      </div>
                      <div className={`chat-bubble ${msg.role}`}>
                        {msg.content}
                        <div className="text-xs text-muted mt-4">{msg.timestamp.toLocaleTimeString()}</div>
                      </div>
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="chat-message">
                      <div className="chat-avatar bot">🤖</div>
                      <div className="chat-bubble bot" style={{ display: "flex", gap: "0.5rem", alignItems: "center", padding: "0.75rem 1.25rem" }}>
                        <span className="pulse-indicator"></span>
                        <span className="pulse-indicator" style={{ animationDelay: "0.2s" }}></span>
                        <span className="pulse-indicator" style={{ animationDelay: "0.4s" }}></span>
                        <span className="text-muted text-xs" style={{ marginLeft: "0.5rem" }}>Agent is thinking...</span>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="chat-input-row">
                  <input className="chat-input" value={chatInput} onChange={e => setChatInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(chatInput); } }}
                    placeholder="Ask the MDT Copilot anything..." disabled={chatLoading} />
                  <button className="btn btn-primary" onClick={() => sendChat(chatInput)} disabled={chatLoading || !chatInput.trim()}>
                    {chatLoading ? <span className="spinner">↻</span> : "→"}
                  </button>
                  <button className="btn btn-ghost" onClick={() => setChatMessages([])} title="Clear chat">🗑️</button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={{ textAlign: "center", padding: "2rem", borderTop: "1px solid var(--border)" }}>
        <p className="text-xs text-muted">
          MDT AI Copilot · LangGraph + OpenRouter GPT-OSS-20B · Cohere embed-v4.0 · OpenAirInterface 5G NR · Built with ❤️ for 5G/6G research
        </p>
      </footer>
    </div>
  );
}

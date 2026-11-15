import { useCallback, useEffect, useMemo, useState } from "react";
import { BarChart3 } from "lucide-react";
import {
  Activity, BrainCircuit, Cpu, Database, Wifi, WifiOff,
  LayoutDashboard, Radio, ClipboardList, FileText, Play,
  Square, RefreshCw, Download, CheckCircle2, CircleHelp,
  AlertTriangle, Clock3, ChevronRight, ShieldCheck
} from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid,
  XAxis, YAxis, Tooltip, ReferenceLine
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

type Page = "overview" | "live" | "sessions" | "research";
interface SensorSample {
  timestamp_us: number; ax_g: number; ay_g: number; az_g: number;
  gx_dps: number; gy_dps: number; gz_dps: number;
}
interface Prediction {
  status: string; research_model_score?: number | null;
  candidate_for_review?: boolean; message?: string;
}
interface LatestResponse {
  sensor: SensorSample | null; prediction: Prediction;
  samples_received: number; last_received_at: string | null;
  sensor_connected: boolean; recording_session_id: string | null;
}
interface Session {
  id: string; label: string; started_at: string; ended_at: string | null;
  status: string; sample_count: number; prediction_count: number;
}
interface CandidateEvent {
  event_id: number; start_at: string; end_at: string;
  duration_seconds: number | null; window_count: number;
  max_score: number; mean_score: number;
  review_status: string; review_note: string;
}
interface ModelMetrics {
  precision?: number; recall?: number; f1?: number;
  balanced_accuracy?: number; confusion_matrix?: number[][];
  n_windows?: number; n_subjects?: number;
  class_counts?: { non_freezing: number; freezing: number };
}
interface SessionReport {
  sample_count: number; prediction_windows: number; candidate_segments: number;
  review_counts: Record<string, number>; mean_research_model_score: number | null;
  max_research_model_score: number | null; recorded_span_seconds: number | null;
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function time(value?: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function dateTime(value?: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}
function percent(value?: number) {
  return value === undefined || value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function StatCard({ label, value, note, icon: Icon }: {
  label: string; value: string; note: string; icon: typeof Activity;
}) {
  return <div className="stat-card">
    <div className="stat-label">{label}<Icon size={17} /></div>
    <div className="stat-value">{value}</div>
    <div className="stat-note">{note}</div>
  </div>;
}

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [reviewingEvent, setReviewingEvent] = useState<number | null>(null);
  const [latest, setLatest] = useState<LatestResponse | null>(null);
  const [history, setHistory] = useState<SensorSample[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [events, setEvents] = useState<CandidateEvent[]>([]);
  const [report, setReport] = useState<SessionReport | null>(null);
  const [label, setLabel] = useState("Mobility recording");
  const [backendOnline, setBackendOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [health, current, latestData, historyData, metricsData, sessionData] = await Promise.all([
        api<{ backend: string; model_loaded: boolean }>("/api/health"),
        api<{ active: boolean; session: Session | null }>("/api/sessions/current"),
        api<LatestResponse>("/api/imu/latest"),
        api<SensorSample[]>("/api/imu/history"),
        api<ModelMetrics>("/api/model/metrics").catch(() => null),
        api<Session[]>("/api/sessions"),
      ]);
      setBackendOnline(health.backend === "online");
      setModelLoaded(health.model_loaded);
      setActiveSession(current.session);
      setLatest(latestData);
      setHistory(historyData);
      setMetrics(metricsData);
      setSessions(sessionData);
      setLastRefresh(new Date());
      setError("");
    } catch (err) {
      setBackendOnline(false);
      setError(err instanceof Error ? err.message : "Could not connect to backend.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const loadSession = useCallback(async (id: string) => {
    try {
      const [eventData, reportData] = await Promise.all([
        api<CandidateEvent[]>(`/api/sessions/${id}/events`),
        api<SessionReport>(`/api/sessions/${id}/report`),
      ]);
      setEvents(eventData);
      setReport(reportData);
      setSelectedSession(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load session.");
    }
  }, []);

  useEffect(() => {
    if (selectedSession) void loadSession(selectedSession);
  }, [selectedSession, loadSession]);

  const connected = Boolean(latest?.sensor_connected);
  const sensor = latest?.sensor;
  const score = latest?.prediction?.research_model_score;
  const magnitude = sensor ? Math.sqrt(sensor.ax_g ** 2 + sensor.ay_g ** 2 + sensor.az_g ** 2) : null;

  const chartData = useMemo(() => history.map((sample, i) => ({
    sample: i + 1,
    ax: sample.ax_g, ay: sample.ay_g, az: sample.az_g,
    magnitude: Math.sqrt(sample.ax_g ** 2 + sample.ay_g ** 2 + sample.az_g ** 2),
  })), [history]);

  async function startRecording() {
    setBusy(true); setError("");
    try {
      await api<Session>("/api/sessions/start", {
        method: "POST", body: JSON.stringify({ label: label.trim() || "Mobility recording" }),
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start recording.");
    } finally { setBusy(false); }
  }

  async function stopRecording() {
    setBusy(true); setError("");
    try {
      const stopped = await api<{ id: string }>("/api/sessions/stop", { method: "POST" });
      await refresh();
      setPage("sessions");
      await loadSession(stopped.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not stop recording.");
    } finally { setBusy(false); }
  }

  
async function reviewEvent(
  eventId: number,
  status: "confirmed" | "not_observed" | "uncertain"
) {
  if (!selectedSession || reviewingEvent !== null) return;

  try {
    setReviewingEvent(eventId);
    setError("");

    await api(`/api/sessions/${selectedSession}/events/${eventId}/review`, {
      method: "POST",
      body: JSON.stringify({
        review_status: status,
        note: "Reviewed in StrideSense."
      }),
    });

    await loadSession(selectedSession);
  } catch (err) {
    setError(
      err instanceof Error ? err.message : "Could not save review."
    );
  } finally {
    setReviewingEvent(null);
  }
}

  const nav: { id: Page; label: string; icon: typeof Activity }[] = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "live", label: "Live monitoring", icon: Activity },
    { id: "sessions", label: "Sessions & review", icon: ClipboardList },
  ];

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Activity size={23} /></div>
        <div><div className="brand-name">StrideSense</div><div className="brand-sub">MOBILITY RESEARCH</div></div>
      </div>
      <div className="nav-caption">WORKSPACE</div>
      <nav className="nav-list">
        {nav.map(item => {
          const Icon = item.icon;
          return <button key={item.id} className={`nav-button ${page === item.id ? "selected" : ""}`} onClick={() => setPage(item.id)}>
            <Icon size={17} /><span>{item.label}</span><ChevronRight size={14} className="nav-chevron" />
          </button>;
        })}
      </nav>
      <div className="sidebar-bottom">
        <div className="hardware-card">
          <div className="hardware-title"><Cpu size={16} /> Device status</div>
          <div className="hardware-row"><span className={`dot ${connected ? "green" : ""}`} />{connected ? "ESP32 streaming" : "Waiting for ESP32"}</div>
          <div className="hardware-meta">GY-6500 · 6-axis IMU · 64 Hz</div>
          <div className="hardware-row"><span className={`dot ${backendOnline ? "green" : ""}`} />{backendOnline ? "Backend online" : "Backend offline"}</div>
          <div className="hardware-row"><span className={`dot ${modelLoaded ? "green" : ""}`} />{modelLoaded ? "Research model loaded" : "Model unavailable"}</div>
        </div>
        <div className="side-foot">EXPERIMENTAL PROTOTYPE · v1.0</div>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div><div className="crumb">STRIDESENSE <span>/</span> {page.replace("-", " ").toUpperCase()}</div><h1>{nav.find(n => n.id === page)?.label}</h1></div>
        <div className="topbar-actions">
          <div className="prototype-pill"><ShieldCheck size={14} /> RESEARCH PROTOTYPE</div>
          <div className={`connection-pill ${connected && backendOnline ? "online" : ""}`}>
            {connected && backendOnline ? <Wifi size={14} /> : <WifiOff size={14} />}
            {connected && backendOnline ? "LIVE" : "OFFLINE"}
          </div>
        </div>
      </header>

      {error && <div className="alert error"><AlertTriangle size={16} />{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      {!modelLoaded && backendOnline && <div className="alert warning"><AlertTriangle size={16} />Model file not loaded. Copy stridesense_rf.joblib into backend/models/ and restart FastAPI.</div>}

      {page === "overview" && <>
        <section className="hero">
          <div><div className="eyebrow">PERSONALISED MOBILITY RECORD</div><h2>Understand movement patterns over time.</h2>
            <p>StrideSense records motion and organises experimental movement-model outputs into sessions that an individual or caregiver can review.</p>
          </div>
          <div className="hero-live"><span className={`pulse ${connected ? "active" : ""}`} />{connected ? "SENSOR STREAM ACTIVE" : "AWAITING SENSOR"}</div>
        </section>

        <section className="stats-grid">
          <StatCard label="ACCELERATION MAGNITUDE" value={magnitude === null ? "—" : `${magnitude.toFixed(3)} g`} note="Resultant 3-axis acceleration" icon={Activity} />
          <StatCard label="SAMPLES RECEIVED" value={(latest?.samples_received ?? 0).toLocaleString()} note="Since backend startup" icon={Database} />
          <StatCard label="SAVED SESSIONS" value={String(sessions.length)} note="Local recording history" icon={ClipboardList} />
        </section>

        <section className="grid-two">
          <div className="panel">
            <div className="panel-heading"><div><h3>Recent movement signal</h3><p>GY-6500 accelerometer · rolling 2-second buffer</p></div><span className="tag">LIVE DATA</span></div>
            <SignalChart data={chartData} />
            <div className="legend"><span><i className="c-blue" />X axis</span><span><i className="c-purple" />Y axis</span><span><i className="c-teal" />Z axis</span></div>
          </div>
          <div className="panel ai-panel">
            <div className="panel-heading"><div><h3>Experimental model output</h3><p>Latest 2-second sensor window</p></div><BrainCircuit size={20} className="teal-icon" /></div>
            <div className="score-box"><div className="score-label">Uncalibrated research score</div><div className="score-number">{score == null ? "—" : score.toFixed(4)}</div><div className="score-track"><div className="score-fill" style={{ width: `${Math.max(0, Math.min(100, (score ?? 0) * 100))}%` }} /></div></div>
            <div className="output-status"><span className={`dot ${modelLoaded ? "green" : ""}`} />{latest?.prediction?.status ?? "Waiting for sensor window"}</div>
            <p className="model-note">{latest?.prediction?.message ?? "A full sensor window is required before inference."}</p>
          </div>
        </section>

        <section className="grid-two">
          <div className="panel">
            <div className="panel-heading"><div><h3>Gyroscope readings</h3><p>Angular velocity · degrees per second</p></div><Radio size={18} className="teal-icon" /></div>
            <div className="axis-grid">
              {[["GX", sensor?.gx_dps], ["GY", sensor?.gy_dps], ["GZ", sensor?.gz_dps]].map(([name, value]) => <div className="axis-tile" key={String(name)}><span>{name}</span><strong>{value == null ? "—" : Number(value).toFixed(2)}</strong><small>°/s</small></div>)}
            </div>
          </div>
          <div className="panel">
            <div className="panel-heading"><div><h3>Recording session</h3><p>Save a movement observation for later review</p></div><Clock3 size={18} className="teal-icon" /></div>
            <div className="recording-controls">
              <input value={label} onChange={e => setLabel(e.target.value)} maxLength={80} placeholder="Session label" disabled={Boolean(activeSession)} />
              {activeSession ? <button className="button stop" disabled={busy} onClick={() => void stopRecording()}><Square size={15} />Stop recording</button> : <button className="button primary" disabled={busy || !connected || !modelLoaded} onClick={() => void startRecording()}><Play size={15} />Start recording</button>}
            </div>
            {activeSession ? <div className="recording-state"><span className="pulse active" />Recording: {activeSession.label}</div> : <div className="helper-text">Start a session to store raw sensor samples and model windows locally.</div>}
          </div>
        </section>
      </>}

      {page === "live" && <>
        <section className="page-intro"><div><div className="eyebrow">HARDWARE STREAM</div><h2>Live movement monitoring</h2><p>Measurements received directly from the ESP32 and GY-6500. This screen is for prototype observation, not clinical monitoring.</p></div><div className={`hero-live ${connected ? "" : "muted"}`}><span className={`pulse ${connected ? "active" : ""}`} />{connected ? "CONNECTED" : "DISCONNECTED"}</div></section>
        <section className="stats-grid">
          <StatCard label="ACCELERATION X" value={sensor ? `${sensor.ax_g.toFixed(4)} g` : "—"} note="X-axis acceleration" icon={Activity} />
          <StatCard label="ACCELERATION Y" value={sensor ? `${sensor.ay_g.toFixed(4)} g` : "—"} note="Y-axis acceleration" icon={Activity} />
          <StatCard label="ACCELERATION Z" value={sensor ? `${sensor.az_g.toFixed(4)} g` : "—"} note="Z-axis acceleration" icon={Activity} />
          <StatCard label="LAST SENSOR UPDATE" value={time(latest?.last_received_at)} note="Backend receive time" icon={Clock3} />
        </section>
        <div className="panel"><div className="panel-heading"><div><h3>Six-axis sensor stream</h3><p>Latest rolling buffer · source sampling rate 64 Hz</p></div><span className="tag">{history.length} SAMPLES</span></div><SignalChart data={chartData} /></div>
        <div className="panel"><div className="panel-heading"><div><h3>Latest raw measurement</h3><p>Values reported by the GY-6500 firmware</p></div></div><div className="raw-grid">{[["AX", sensor?.ax_g, "g"], ["AY", sensor?.ay_g, "g"], ["AZ", sensor?.az_g, "g"], ["GX", sensor?.gx_dps, "°/s"], ["GY", sensor?.gy_dps, "°/s"], ["GZ", sensor?.gz_dps, "°/s"]].map(([k,v,u]) => <div className="raw-tile" key={String(k)}><span>{k}</span><strong>{v == null ? "—" : Number(v).toFixed(5)}</strong><small>{u}</small></div>)}</div></div>
      </>}

      {page === "sessions" && <>
        <section className="page-intro"><div><div className="eyebrow">PERSONAL RECORDS</div><h2>Sessions & event review</h2><p>Record a session, inspect candidate segments generated by the research model, and mark what the person actually experienced.</p></div>{activeSession && <span className="recording-state"><span className="pulse active" />Recording now</span>}</section>
        <div className="grid-two session-layout">
          <div className="panel"><div className="panel-heading"><div><h3>Recording history</h3><p>{sessions.length} saved sessions</p></div><button className="icon-button" onClick={() => void refresh()} title="Refresh"><RefreshCw size={16}/></button></div>
            {sessions.length === 0 ? <div className="empty">No saved sessions yet. Start one from the Overview page.</div> : <div className="session-list">{sessions.map(s => <button className={`session-row ${selectedSession === s.id ? "chosen" : ""}`} key={s.id} onClick={() => void loadSession(s.id)}><div className="session-row-main"><strong>{s.label}</strong><span>{dateTime(s.started_at)}</span></div><div className="session-row-meta"><span className={`session-status ${s.status}`}>{s.status}</span><span>{s.sample_count.toLocaleString()} samples</span><span>{s.prediction_count} windows</span></div></button>)}</div>}
          </div>
          <div className="panel"><div className="panel-heading"><div><h3>Session report</h3><p>{selectedSession ? selectedSession.slice(0, 8) : "Select a saved session"}</p></div>{selectedSession && <a className="download-link" href={`${API}/api/sessions/${selectedSession}/report.pdf`}><Download size={14}/>PDF</a>}</div>
            {report ? <><div className="report-grid"><div><span>Recorded samples</span><strong>{report.sample_count.toLocaleString()}</strong></div><div><span>Model windows</span><strong>{report.prediction_windows}</strong></div><div><span>Candidate segments</span><strong>{report.candidate_segments}</strong></div><div><span>Recorded span</span><strong>{report.recorded_span_seconds == null ? "—" : `${Math.round(report.recorded_span_seconds)}s`}</strong></div></div><div className="review-counts"><div><span>Confirmed by reviewer</span><strong>{report.review_counts.confirmed ?? 0}</strong></div><div><span>Not observed</span><strong>{report.review_counts.not_observed ?? 0}</strong></div><div><span>Uncertain</span><strong>{report.review_counts.uncertain ?? 0}</strong></div><div><span>Awaiting review</span><strong>{report.review_counts.awaiting_review ?? 0}</strong></div></div></> : <div className="empty">Choose a session to view its summary and review candidate segments.</div>}
          </div>
        </div>
        
<div className="panel">
  <div className="panel-heading">
    <div>
      <h3>Candidate segments for review</h3>
      <p>
        Review the movement observations flagged by the research model.
        Your feedback is subjective and does not establish a clinical event.
      </p>
    </div>
    <span className="tag">HUMAN REVIEW</span>
  </div>

  {!selectedSession ? (
    <div className="empty">Select a session first.</div>
  ) : events.length === 0 ? (
    <div className="empty">
      <CircleHelp size={19} />
      No candidate segments were generated in this session.
      This does not establish that no freezing occurred.
    </div>
  ) : (
    <div className="event-list">
      {events.map(event => (
        <div className="event-card" key={event.event_id}>
          <div className="event-top">
            <div>
              <strong>Candidate segment #{event.event_id}</strong>
              <span>{dateTime(event.start_at)}</span>
            </div>
            <span className={`review-badge ${event.review_status}`}>
              {event.review_status.replaceAll("_", " ")}
            </span>
          </div>

          <div className="event-meta">
            <span>{event.window_count} overlapping windows</span>
            <span>Max score: {event.max_score.toFixed(3)}</span>
            <span>Mean score: {event.mean_score.toFixed(3)}</span>
          </div>

          <div className="review-question">
            <strong>We'd like your feedback</strong>
            <p>
              During this time, did you experience a moment when your
              walking suddenly stopped or your feet felt temporarily
              stuck, despite trying to move?
            </p>
            <span>
              Think about what you remember experiencing at this time.
              There are no right or wrong answers.
            </span>
          </div>

          {event.review_status === "awaiting_review" ? (
            <div className="review-actions">
  <button
    className="button small positive"
    disabled={reviewingEvent === event.event_id}
    onClick={() => void reviewEvent(event.event_id, "confirmed")}
  >
    <CheckCircle2 size={14} />
    {reviewingEvent === event.event_id ? "Saving..." : "Yes, I experienced this"}
  </button>

  <button
    className="button small neutral"
    disabled={reviewingEvent === event.event_id}
    onClick={() => void reviewEvent(event.event_id, "not_observed")}
  >
    <Square size={13} />
    {reviewingEvent === event.event_id ? "Saving..." : "No, I didn't notice this"}
  </button>

  <button
    className="button small neutral"
    disabled={reviewingEvent === event.event_id}
    onClick={() => void reviewEvent(event.event_id, "uncertain")}
  >
    <CircleHelp size={14} />
    {reviewingEvent === event.event_id ? "Saving..." : "I'm unsure"}
  </button>
</div>
          ) : (
            <div className="review-complete">
              <CheckCircle2 size={16} />
              <span>
                Response recorded:{" "}
                <strong>
                  {event.review_status === "confirmed"
                    ? "Yes, I experienced this"
                    : event.review_status === "not_observed"
                    ? "No, I didn't notice this"
                    : "I'm unsure"}
                </strong>
              </span>
            </div>
          )}

          <div className="review-disclaimer">
            This is a research prototype. Model-generated segments are
            observations for review, not medical diagnoses.
          </div>
        </div>
      ))}
    </div>
  )}
</div>
      </>}

      {page === "research" && <>
        <section className="page-intro"><div><div className="eyebrow">MODEL TRANSPARENCY</div><h2>Research & model evaluation</h2><p>Participant-level cross-validation on Daphnet. These results describe the public research dataset, not the live GY-6500 prototype.</p></div><span className="tag">DAPHNET · UCI</span></section>
        <section className="stats-grid">
          <StatCard label="BALANCED ACCURACY" value={percent(metrics?.balanced_accuracy)} note="5-fold participant-grouped CV" icon={BarChart3} />
          <StatCard label="FREEZING-CLASS F1" value={percent(metrics?.f1)} note="Precision / recall balance" icon={BrainCircuit} />
          <StatCard label="FREEZING PRECISION" value={percent(metrics?.precision)} note="Positive predictive value on OOF predictions" icon={CheckCircle2} />
          <StatCard label="FREEZING RECALL" value={percent(metrics?.recall)} note="Share of labelled windows identified" icon={Activity} />
        </section>
        <div className="grid-two">
          <div className="panel"><div className="panel-heading"><div><h3>Confusion matrix</h3><p>Rows = true labels; columns = predicted labels</p></div></div>
            {metrics?.confusion_matrix ? <div className="matrix"><div className="matrix-axis">ACTUAL ↓ / PREDICTED →</div><div className="matrix-grid"><div className="matrix-cell"><span>True non-freezing / predicted non-freezing</span><strong>{metrics.confusion_matrix[0][0].toLocaleString()}</strong></div><div className="matrix-cell bad"><span>True non-freezing / predicted freezing</span><strong>{metrics.confusion_matrix[0][1].toLocaleString()}</strong></div><div className="matrix-cell bad"><span>True freezing / predicted non-freezing</span><strong>{metrics.confusion_matrix[1][0].toLocaleString()}</strong></div><div className="matrix-cell"><span>True freezing / predicted freezing</span><strong>{metrics.confusion_matrix[1][1].toLocaleString()}</strong></div></div></div> : <div className="empty">Metrics unavailable. Check the saved model bundle.</div>}
          </div>
          <div className="panel"><div className="panel-heading"><div><h3>Evaluation dataset</h3><p>Research provenance</p></div><Database size={18} className="teal-icon"/></div><div className="device-row"><span>Dataset</span><strong>Daphnet FOG</strong></div><div className="device-row"><span>Participants</span><strong>{metrics?.n_subjects ?? "—"}</strong></div><div className="device-row"><span>Windows</span><strong>{metrics?.n_windows?.toLocaleString() ?? "—"}</strong></div><div className="device-row"><span>Non-freezing windows</span><strong>{metrics?.class_counts?.non_freezing?.toLocaleString() ?? "—"}</strong></div><div className="device-row"><span>Freezing windows</span><strong>{metrics?.class_counts?.freezing?.toLocaleString() ?? "—"}</strong></div><div className="device-row"><span>Validation</span><strong>5-fold GroupKFold</strong></div></div>
        </div>
      </>}

      <footer className="footer"><span>STRIDESENSE AI</span><span><Clock3 size={13}/> Updated {lastRefresh ? time(lastRefresh.toISOString()) : "—"}</span><span>SYNTHETIC / PUBLIC RESEARCH DATA · NOT FOR CLINICAL USE</span></footer>
    </main>
  </div>;
}

function SignalChart({ data }: { data: { sample: number; ax: number; ay: number; az: number }[] }) {
  if (!data.length) return <div className="empty-chart">Waiting for sensor measurements…</div>;
  return <ResponsiveContainer width="100%" height={270}>
    <LineChart data={data} margin={{ top: 5, right: 12, left: -15, bottom: 0 }}>
      <CartesianGrid strokeDasharray="3 5" stroke="#26364a" />
      <XAxis dataKey="sample" stroke="#8095ad" tick={{ fontSize: 10 }} />
      <YAxis stroke="#8095ad" tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
      <Tooltip contentStyle={{ background: "#101d2e", border: "1px solid #34465d", borderRadius: 9, color: "#eef5ff" }} />
      <ReferenceLine y={0} stroke="#53677d" />
      <Line dataKey="ax" name="X axis" stroke="#61c8ff" strokeWidth={1.8} dot={false} isAnimationActive={false} />
      <Line dataKey="ay" name="Y axis" stroke="#b19aff" strokeWidth={1.8} dot={false} isAnimationActive={false} />
      <Line dataKey="az" name="Z axis" stroke="#4cdeba" strokeWidth={1.8} dot={false} isAnimationActive={false} />
    </LineChart>
  </ResponsiveContainer>;
}


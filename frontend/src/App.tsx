
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Cpu,
  Database,
  Wifi,
  WifiOff,
  Clock3,
  RefreshCw,
  Radio,
  BarChart3,
  Layers,
  ShieldAlert
} from "lucide-react";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine
} from "recharts";

import "./App.css";

const API = "http://127.0.0.1:8000";

interface SensorSample {
  timestamp_us: number;
  ax_g: number;
  ay_g: number;
  az_g: number;
  gx_dps: number;
  gy_dps: number;
  gz_dps: number;
}

interface Prediction {
  status: string;
  research_model_score?: number;
  message?: string;
}

interface LatestResponse {
  sensor: SensorSample | null;
  prediction: Prediction;
  samples_received: number;
  last_received_at: string | null;
  sensor_connected: boolean;
}

interface ModelMetrics {
  precision?: number;
  recall?: number;
  f1?: number;
  balanced_accuracy?: number;
}

type Page = "overview" | "live" | "research";

function StatCard({
  title,
  value,
  subtitle,
  icon: Icon
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ElementType;
}) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{title}</span>
        <Icon size={19} />
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-subtitle">{subtitle}</div>
    </div>
  );
}

function App() {
  const [page, setPage] = useState<Page>("overview");
  const [latest, setLatest] = useState<LatestResponse | null>(null);
  const [history, setHistory] = useState<SensorSample[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    let requesting = false;

    async function refreshData() {
      if (requesting) return;
      requesting = true;

      try {
        const [latestRes, historyRes, healthRes, metricsRes] =
          await Promise.all([
            fetch(`${API}/api/imu/latest`),
            fetch(`${API}/api/imu/history`),
            fetch(`${API}/api/health`),
            fetch(`${API}/api/model/metrics`)
          ]);

        if (!latestRes.ok || !historyRes.ok || !healthRes.ok) {
          throw new Error("Backend request failed");
        }

        const [latestData, historyData, healthData] =
          await Promise.all([
            latestRes.json(),
            historyRes.json(),
            healthRes.json()
          ]);

        let metricsData = null;

        if (metricsRes.ok) {
          metricsData = await metricsRes.json();
        }

        if (cancelled) return;

        setLatest(latestData);
        setHistory(historyData);
        setMetrics(metricsData);
        setBackendOnline(healthData.backend === "online");
        setLastUpdate(new Date());
        setLoading(false);

      } catch (error) {
        if (!cancelled) {
          setBackendOnline(false);
          setLoading(false);
        }
      } finally {
        requesting = false;
      }
    }

    refreshData();

    const timer = window.setInterval(refreshData, 500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const connected = useMemo(() => {
    if (!latest?.last_received_at) return false;

    const elapsed =
      Date.now() - new Date(latest.last_received_at).getTime();

    return elapsed < 3000;
  }, [latest, lastUpdate]);

  const sensor = latest?.sensor;

  const accelerationMagnitude = sensor
    ? Math.sqrt(
        sensor.ax_g ** 2 +
        sensor.ay_g ** 2 +
        sensor.az_g ** 2
      )
    : 0;

  const chartData = useMemo(
    () =>
      history.map((sample, index) => ({
        index,
        ax: sample.ax_g,
        ay: sample.ay_g,
        az: sample.az_g,
        gx: sample.gx_dps,
        gy: sample.gy_dps,
        gz: sample.gz_dps,
        magnitude: Math.sqrt(
          sample.ax_g ** 2 +
          sample.ay_g ** 2 +
          sample.az_g ** 2
        )
      })),
    [history]
  );

  const score = latest?.prediction?.research_model_score;

  function formatMetric(value?: number) {
    return value === undefined || value === null
      ? "—"
      : `${(value * 100).toFixed(1)}%`;
  }

  const navigation = [
    { id: "overview", label: "Overview", icon: Layers },
    { id: "live", label: "Live Monitoring", icon: Activity },
    { id: "research", label: "Research & AI", icon: BrainCircuit }
  ] as const;

  return (
    <div className="app-shell">

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={25} />
          </div>
          <div>
            <h1>StrideSense</h1>
            <span>AI MOBILITY RESEARCH</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${
                  page === item.id ? "active" : ""
                }`}
                onClick={() => setPage(item.id)}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="device-card">
            <div className="device-heading">
              <Cpu size={17} />
              Hardware
            </div>
            <div className="device-status">
              <span className={`status-dot ${connected ? "online" : ""}`} />
              {connected ? "ESP32 Connected" : "Waiting for device"}
            </div>
            <span className="device-caption">GY-6500 · 64 Hz</span>
          </div>

          <div className="sidebar-footer">
            STRIDESENSE AI · RESEARCH BUILD
          </div>
        </div>
      </aside>

      <main className="main-content">

        <header className="topbar">
          <div>
            <div className="breadcrumb">
              STRIDESENSE <span>/</span> {page.toUpperCase()}
            </div>
            <h2>
              {page === "overview" && "Overview"}
              {page === "live" && "Live Monitoring"}
              {page === "research" && "Research & AI"}
            </h2>
          </div>

          <div className="topbar-right">
            <div className="prototype-badge">
              <ShieldAlert size={14} />
              EXPERIMENTAL RESEARCH PROTOTYPE
            </div>

            <div className={`connection-pill ${
              connected && backendOnline ? "connected" : ""
            }`}>
              {connected && backendOnline
                ? <Wifi size={15} />
                : <WifiOff size={15} />}
              {connected && backendOnline ? "LIVE" : "OFFLINE"}
            </div>
          </div>
        </header>

        {!backendOnline && (
          <div className="warning-banner">
            Backend unavailable. Check that FastAPI and the ESP32 serial
            bridge are running.
          </div>
        )}

        {loading && (
          <div className="loading-banner">
            <RefreshCw size={16} className="spin" />
            Connecting to StrideSense...
          </div>
        )}

        {page === "overview" && (
          <>
            <section className="welcome-section">
              <div>
                <div className="eyebrow">YOUR MOVEMENT DATA</div>
                <h3>Hardware intelligence, in real time.</h3>
                <p>
                  Movement measurements from your ESP32 are being
                  processed by your research machine learning model.
                </p>
              </div>

              <div className={`live-indicator ${
                connected ? "live" : ""
              }`}>
                <span className="pulse-dot" />
                {connected ? "SENSOR STREAM ACTIVE" : "AWAITING SENSOR"}
              </div>
            </section>

            <section className="stats-grid">
              <StatCard
                title="ACCELERATION X"
                value={sensor ? sensor.ax_g.toFixed(3) + " g" : "—"}
                subtitle="Linear acceleration"
                icon={Activity}
              />

              <StatCard
                title="ACCELERATION Y"
                value={sensor ? sensor.ay_g.toFixed(3) + " g" : "—"}
                subtitle="Linear acceleration"
                icon={Activity}
              />

              <StatCard
                title="ACCELERATION Z"
                value={sensor ? sensor.az_g.toFixed(3) + " g" : "—"}
                subtitle="Linear acceleration"
                icon={Activity}
              />

              <StatCard
                title="SAMPLES RECEIVED"
                value={(latest?.samples_received ?? 0).toLocaleString()}
                subtitle="Since backend startup"
                icon={Database}
              />
            </section>

            <section className="content-grid">
              <div className="panel large-panel">
                <div className="panel-header">
                  <div>
                    <h3>Live acceleration</h3>
                    <p>Three-axis accelerometer measurements</p>
                  </div>
                  <span className="small-tag">GY-6500</span>
                </div>

                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={270}>
                    <LineChart data={chartData}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#253247"
                      />
                      <XAxis
                        dataKey="index"
                        stroke="#8190a7"
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis
                        stroke="#8190a7"
                        tick={{ fontSize: 11 }}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#111c2d",
                          border: "1px solid #30415b",
                          borderRadius: 10,
                          color: "#ffffff"
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="ax"
                        stroke="#48b8ff"
                        dot={false}
                        strokeWidth={2}
                        name="X axis"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="ay"
                        stroke="#a78bfa"
                        dot={false}
                        strokeWidth={2}
                        name="Y axis"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="az"
                        stroke="#39d5ad"
                        dot={false}
                        strokeWidth={2}
                        name="Z axis"
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-chart">
                    Waiting for sensor measurements...
                  </div>
                )}

                <div className="chart-legend">
                  <span><i className="blue" /> X Axis</span>
                  <span><i className="purple" /> Y Axis</span>
                  <span><i className="green" /> Z Axis</span>
                </div>
              </div>

              <div className="panel ai-panel">
                <div className="panel-header">
                  <div>
                    <h3>AI analysis</h3>
                    <p>Research model output</p>
                  </div>
                  <BrainCircuit size={21} className="accent-icon" />
                </div>

                <div className="ai-score">
                  <span>Latest model score</span>
                  <strong>
                    {score !== undefined
                      ? score.toFixed(4)
                      : "—"}
                  </strong>
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{
                        width: `${Math.min((score ?? 0) * 100, 100)}%`
                      }}
                    />
                  </div>
                </div>

                <div className="ai-status">
                  <span className="status-dot online" />
                  {latest?.prediction?.status ?? "Waiting"}
                </div>

                <div className="ai-explanation">
                  {latest?.prediction?.message ??
                    "Waiting for a complete movement window."}
                </div>

                <div className="info-note">
                  This is an experimental research-model score,
                  not a clinical probability or diagnosis.
                </div>
              </div>
            </section>

            <section className="content-grid">
              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h3>Gyroscope readings</h3>
                    <p>Angular velocity · degrees per second</p>
                  </div>
                  <Radio size={19} className="accent-icon" />
                </div>

                {sensor ? (
                  <div className="axis-grid">
                    {[
                      ["GX", sensor.gx_dps],
                      ["GY", sensor.gy_dps],
                      ["GZ", sensor.gz_dps]
                    ].map(([name, value]) => (
                      <div className="axis-value" key={String(name)}>
                        <span>{name}</span>
                        <strong>{Number(value).toFixed(2)}</strong>
                        <small>°/s</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-chart">No sensor data</div>
                )}
              </div>

              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h3>Device status</h3>
                    <p>Hardware and backend connection</p>
                  </div>
                  <Cpu size={20} className="accent-icon" />
                </div>

                <div className="device-row">
                  <span>ESP32 board</span>
                  <strong>{connected ? "Connected" : "Offline"}</strong>
                </div>
                <div className="device-row">
                  <span>GY-6500 sensor</span>
                  <strong>{connected ? "Receiving data" : "No data"}</strong>
                </div>
                <div className="device-row">
                  <span>FastAPI backend</span>
                  <strong>{backendOnline ? "Online" : "Offline"}</strong>
                </div>
                <div className="device-row">
                  <span>Random Forest model</span>
                  <strong>{metrics ? "Loaded" : "Unavailable"}</strong>
                </div>
                <div className="device-row">
                  <span>Acceleration magnitude</span>
                  <strong>{accelerationMagnitude.toFixed(3)} g</strong>
                </div>
              </div>
            </section>
          </>
        )}

        {page === "live" && (
          <>
            <section className="welcome-section">
              <div>
                <div className="eyebrow">HARDWARE STREAM</div>
                <h3>Live movement measurements</h3>
                <p>
                  The following measurements are collected directly
                  from the physical GY-6500 sensor.
                </p>
              </div>
              <div className={`live-indicator ${
                connected ? "live" : ""
              }`}>
                <span className="pulse-dot" />
                {connected ? "STREAMING" : "DISCONNECTED"}
              </div>
            </section>

            <section className="stats-grid">
              <StatCard title="ACCELERATION MAGNITUDE"
                value={sensor ? accelerationMagnitude.toFixed(4) + " g" : "—"}
                subtitle="Resultant acceleration" icon={Activity} />
              <StatCard title="GYROSCOPE X"
                value={sensor ? sensor.gx_dps.toFixed(2) : "—"}
                subtitle="Degrees per second" icon={Radio} />
              <StatCard title="GYROSCOPE Y"
                value={sensor ? sensor.gy_dps.toFixed(2) : "—"}
                subtitle="Degrees per second" icon={Radio} />
              <StatCard title="GYROSCOPE Z"
                value={sensor ? sensor.gz_dps.toFixed(2) : "—"}
                subtitle="Degrees per second" icon={Radio} />
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <h3>Accelerometer waveform</h3>
                  <p>Recent 128 sensor samples</p>
                </div>
                <span className="small-tag">64 Hz SOURCE</span>
              </div>
              <ResponsiveContainer width="100%" height={330}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#253247" />
                  <XAxis dataKey="index" stroke="#8190a7" />
                  <YAxis stroke="#8190a7" domain={["auto", "auto"]} />
                  <Tooltip />
                  <ReferenceLine y={0} stroke="#54647b" />
                  <Line dataKey="ax" stroke="#48b8ff" dot={false} name="X" isAnimationActive={false} />
                  <Line dataKey="ay" stroke="#a78bfa" dot={false} name="Y" isAnimationActive={false} />
                  <Line dataKey="az" stroke="#39d5ad" dot={false} name="Z" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <h3>Latest measurement</h3>
                  <p>Raw sensor values</p>
                </div>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>CHANNEL</th>
                      <th>VALUE</th>
                      <th>UNIT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sensor && [
                      ["Acceleration X", sensor.ax_g, "g"],
                      ["Acceleration Y", sensor.ay_g, "g"],
                      ["Acceleration Z", sensor.az_g, "g"],
                      ["Gyroscope X", sensor.gx_dps, "°/s"],
                      ["Gyroscope Y", sensor.gy_dps, "°/s"],
                      ["Gyroscope Z", sensor.gz_dps, "°/s"]
                    ].map(([name, value, unit]) => (
                      <tr key={String(name)}>
                        <td>{name}</td>
                        <td>{Number(value).toFixed(5)}</td>
                        <td>{unit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {page === "research" && (
          <>
            <section className="welcome-section">
              <div>
                <div className="eyebrow">MODEL EVALUATION</div>
                <h3>Research & AI performance</h3>
                <p>
                  Participant-level cross-validation results from
                  your trained Daphnet research model.
                </p>
              </div>
              <div className="small-tag">DAPHNET DATASET</div>
            </section>

            <section className="stats-grid">
              <StatCard
                title="BALANCED ACCURACY"
                value={formatMetric(metrics?.balanced_accuracy)}
                subtitle="Participant-level evaluation"
                icon={BarChart3}
              />
              <StatCard
                title="F1 SCORE"
                value={formatMetric(metrics?.f1)}
                subtitle="Freezing class"
                icon={BrainCircuit}
              />
              <StatCard
                title="PRECISION"
                value={formatMetric(metrics?.precision)}
                subtitle="Freezing class"
                icon={Activity}
              />
              <StatCard
                title="RECALL"
                value={formatMetric(metrics?.recall)}
                subtitle="Freezing class"
                icon={Database}
              />
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <h3>Model evaluation results</h3>
                  <p>Actual results from the saved training run</p>
                </div>
              </div>

              {metrics ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>METRIC</th>
                        <th>RESULT</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Precision</td>
                        <td>{formatMetric(metrics.precision)}</td>
                      </tr>
                      <tr>
                        <td>Recall</td>
                        <td>{formatMetric(metrics.recall)}</td>
                      </tr>
                      <tr>
                        <td>F1-score</td>
                        <td>{formatMetric(metrics.f1)}</td>
                      </tr>
                      <tr>
                        <td>Balanced accuracy</td>
                        <td>{formatMetric(metrics.balanced_accuracy)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="empty-chart">
                  Model evaluation metrics unavailable. Check the backend.
                </p>
              )}
            </section>

            <section className="panel research-note">
              <ShieldAlert size={22} />
              <div>
                <h3>Research limitations</h3>
                <p>
                  These measurements represent cross-validation on the
                  Daphnet research dataset. They do not establish
                  clinical validity or prove that the model generalises
                  to the live GY-6500 device.
                </p>
                <p>
                  Model scores from live sensor readings must be
                  interpreted as experimental outputs, not medical
                  assessments or clinical probabilities.
                </p>
              </div>
            </section>
          </>
        )}

        <footer className="page-footer">
          <span>STRIDESENSE AI</span>
          <span>
            <Clock3 size={13} />
            {lastUpdate
              ? `Last refreshed ${lastUpdate.toLocaleTimeString()}`
              : "Waiting for backend"}
          </span>
          <span>EXPERIMENTAL RESEARCH PROTOTYPE</span>
        </footer>

      </main>
    </div>
  );
}

export default App;
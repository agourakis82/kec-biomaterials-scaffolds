"""DARWIN Monitoring Dashboard - Performance Dashboards Épicos

🚀 DASHBOARD REVOLUTIONARY - VISUALIZAÇÃO BEYOND STATE-OF-THE-ART
Router FastAPI para dashboards de monitoring em tempo real:

Features Disruptivas:
- 📊 Real-time performance dashboards
- ⚡ JAX speedup visualization
- 🎯 Agent collaboration metrics
- 💰 Cost monitoring e optimization
- 🏥 Health status monitoring
- 📈 Historical performance trends
- 🚨 Live alerts display
- 🔥 Epic performance visualization

Technology: FastAPI + WebSockets + Chart.js + Real-time updates
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import asyncio
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..core.logging import get_logger
from ..monitoring import get_monitoring, DarwinMonitoring

logger = get_logger("darwin.monitoring_dashboard")
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.disconnect(connection)

manager = ConnectionManager()

@router.get("/", response_class=HTMLResponse)
async def monitoring_dashboard(request: Request):
    """
    🚀 DASHBOARD PRINCIPAL - DARWIN MONITORING REVOLUTIONARY
    
    Dashboard épico com visualização em tempo real de:
    - Performance JAX (speedup, throughput)
    - Health system monitoring
    - Active alerts
    - Agent collaboration
    - Cost monitoring
    """
    
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 DARWIN Monitoring Dashboard - Revolutionary Performance</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .dashboard-container {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            gap: 20px;
        }
        
        .status-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            flex: 1;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .status-card h3 {
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .status-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .status-revolutionary {
            color: #00ff88;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .status-achievement {
            color: #ffaa00;
            text-shadow: 0 0 10px #ffaa00;
        }
        
        .status-baseline {
            color: #88aaff;
            text-shadow: 0 0 10px #88aaff;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .chart-container h3 {
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.4em;
        }
        
        .chart-container canvas {
            max-height: 400px;
        }
        
        .alerts-section {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .alert-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }
        
        .alert-critical {
            border-left-color: #ff4444;
            background: rgba(255,68,68,0.2);
        }
        
        .alert-warning {
            border-left-color: #ffaa00;
            background: rgba(255,170,0,0.2);
        }
        
        .alert-info {
            border-left-color: #44aaff;
            background: rgba(68,170,255,0.2);
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .connected {
            background: rgba(0,255,136,0.3);
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        
        .disconnected {
            background: rgba(255,68,68,0.3);
            color: #ff4444;
            border: 1px solid #ff4444;
        }
        
        .epic-glow {
            animation: glow 2s infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(0,255,136,0.5); }
            to { text-shadow: 0 0 30px rgba(0,255,136,0.8); }
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
        }
        
        .cost-section {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .cost-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .cost-value {
            font-weight: bold;
            color: #ffaa00;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1 class="epic-glow">🚀 DARWIN Monitoring Dashboard</h1>
            <p>Revolutionary Performance Monitoring - Beyond State-of-the-Art</p>
            <div id="connectionStatus" class="connection-status disconnected">
                ⚠️ Connecting...
            </div>
        </div>

        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-card">
                <h3>🚀 System Status</h3>
                <div id="systemStatus" class="status-value">Loading...</div>
                <div id="uptime">Uptime: --</div>
            </div>
            <div class="status-card">
                <h3>⚡ JAX Performance</h3>
                <div id="jaxSpeedup" class="status-value">--x</div>
                <div id="performanceLevel">--</div>
            </div>
            <div class="status-card">
                <h3>🎯 Scaffolds Processed</h3>
                <div id="scaffoldsProcessed" class="status-value">--</div>
                <div id="throughput">-- scaffolds/s</div>
            </div>
            <div class="status-card">
                <h3>🚨 Active Alerts</h3>
                <div id="activeAlerts" class="status-value">--</div>
                <div id="alertStatus">System Normal</div>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="charts-grid">
            <!-- JAX Performance Chart -->
            <div class="chart-container">
                <h3>⚡ JAX Speedup Performance</h3>
                <canvas id="speedupChart"></canvas>
            </div>

            <!-- Scaffold Throughput Chart -->
            <div class="chart-container">
                <h3>🎯 Scaffold Processing Throughput</h3>
                <canvas id="throughputChart"></canvas>
            </div>

            <!-- System Health Chart -->
            <div class="chart-container">
                <h3>🏥 System Health Monitoring</h3>
                <canvas id="healthChart"></canvas>
            </div>

            <!-- Agent Collaboration Chart -->
            <div class="chart-container">
                <h3>🤖 Agent Collaboration Score</h3>
                <canvas id="collaborationChart"></canvas>
            </div>
        </div>

        <!-- Cost Monitoring Section -->
        <div class="cost-section">
            <h3>💰 Cost Monitoring & Optimization</h3>
            <div id="costMetrics">
                <div class="cost-item">
                    <span>Compute Cost (Hourly)</span>
                    <span id="hourlyComputeCost" class="cost-value">$--</span>
                </div>
                <div class="cost-item">
                    <span>Daily Estimated Cost</span>
                    <span id="dailyCost" class="cost-value">$--</span>
                </div>
                <div class="cost-item">
                    <span>Monthly Projection</span>
                    <span id="monthlyCost" class="cost-value">$--</span>
                </div>
                <div class="cost-item">
                    <span>Cost per Million Scaffolds</span>
                    <span id="costPerMillion" class="cost-value">$--</span>
                </div>
            </div>
        </div>

        <!-- Active Alerts Section -->
        <div class="alerts-section">
            <h3>🚨 Active Alerts & Notifications</h3>
            <div id="alertsList">
                <div class="loading">No active alerts - System running optimally 🚀</div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let ws = null;
        let charts = {};
        let lastUpdate = null;
        
        // Chart configurations
        const chartColors = {
            primary: '#00ff88',
            secondary: '#ffaa00',
            tertiary: '#88aaff',
            quaternary: '#ff6b6b',
            success: '#00ff88',
            warning: '#ffaa00',
            error: '#ff6b6b'
        };

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            connectWebSocket();
            loadInitialData();
        });

        function initializeCharts() {
            // JAX Speedup Chart
            const speedupCtx = document.getElementById('speedupChart').getContext('2d');
            charts.speedup = new Chart(speedupCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'JAX Speedup Factor',
                        data: [],
                        borderColor: chartColors.primary,
                        backgroundColor: chartColors.primary + '30',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { 
                            labels: { color: 'white' }
                        }
                    },
                    scales: {
                        x: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            suggestedMin: 0
                        }
                    }
                }
            });

            // Throughput Chart
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            charts.throughput = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Scaffolds/Second',
                        data: [],
                        borderColor: chartColors.secondary,
                        backgroundColor: chartColors.secondary + '30',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            suggestedMin: 0
                        }
                    }
                }
            });

            // Health Chart
            const healthCtx = document.getElementById('healthChart').getContext('2d');
            charts.health = new Chart(healthCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU Usage %',
                            data: [],
                            borderColor: chartColors.tertiary,
                            backgroundColor: chartColors.tertiary + '20',
                        },
                        {
                            label: 'Memory Usage %',
                            data: [],
                            borderColor: chartColors.quaternary,
                            backgroundColor: chartColors.quaternary + '20',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: { 
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });

            // Collaboration Chart
            const collaborationCtx = document.getElementById('collaborationChart').getContext('2d');
            charts.collaboration = new Chart(collaborationCtx, {
                type: 'radar',
                data: {
                    labels: ['Dr_Biomaterials', 'Dr_Quantum', 'Dr_Medical', 'Dr_Chemical', 'Dr_Physics'],
                    datasets: [{
                        label: 'Collaboration Score',
                        data: [0, 0, 0, 0, 0],
                        borderColor: chartColors.success,
                        backgroundColor: chartColors.success + '20',
                        pointBackgroundColor: chartColors.success
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        r: {
                            ticks: { 
                                color: 'white',
                                backdropColor: 'transparent'
                            },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            angleLines: { color: 'rgba(255,255,255,0.1)' },
                            pointLabels: { color: 'white' },
                            min: 0,
                            max: 1
                        }
                    }
                }
            });
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/monitoring/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                updateConnectionStatus(true);
                console.log('🚀 Connected to DARWIN monitoring stream');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                updateConnectionStatus(false);
                setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus(false);
            };
        }

        async function loadInitialData() {
            try {
                const response = await fetch('/monitoring/dashboard-data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }

        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connectionStatus');
            if (connected) {
                statusEl.className = 'connection-status connected';
                statusEl.textContent = '✅ Connected';
            } else {
                statusEl.className = 'connection-status disconnected';
                statusEl.textContent = '❌ Disconnected';
            }
        }

        function updateDashboard(data) {
            if (!data || !data.overview) return;

            const overview = data.overview;
            
            // Update status cards
            updateStatusCards(overview);
            
            // Update charts with metrics
            if (data.metrics) {
                updateCharts(data.metrics);
            }
            
            // Update alerts
            if (data.alerts) {
                updateAlerts(data.alerts);
            }
            
            // Update cost monitoring
            updateCostMonitoring(overview);
            
            lastUpdate = new Date();
        }

        function updateStatusCards(overview) {
            // System Status
            const statusEl = document.getElementById('systemStatus');
            const status = overview.status || 'unknown';
            statusEl.textContent = status.toUpperCase();
            statusEl.className = `status-value status-${getStatusClass(status)}`;
            
            // Uptime
            const uptimeEl = document.getElementById('uptime');
            const uptime = overview.uptime_seconds || 0;
            uptimeEl.textContent = `Uptime: ${formatUptime(uptime)}`;
            
            // JAX Performance
            const speedupEl = document.getElementById('jaxSpeedup');
            const speedup = overview.average_speedup || 0;
            speedupEl.textContent = `${speedup.toFixed(1)}x`;
            speedupEl.className = `status-value ${getSpeedupClass(speedup)}`;
            
            const perfLevelEl = document.getElementById('performanceLevel');
            perfLevelEl.textContent = overview.performance_level || 'baseline';
            
            // Scaffolds Processed
            const scaffoldsEl = document.getElementById('scaffoldsProcessed');
            scaffoldsEl.textContent = (overview.scaffolds_processed || 0).toLocaleString();
            
            // Active Alerts
            const alertsEl = document.getElementById('activeAlerts');
            alertsEl.textContent = overview.active_alerts || 0;
            
            const alertStatusEl = document.getElementById('alertStatus');
            const alertCount = overview.active_alerts || 0;
            alertStatusEl.textContent = alertCount === 0 ? 'System Normal' : `${alertCount} Active`;
        }

        function updateCharts(metrics) {
            const now = new Date();
            const timeLabel = now.toLocaleTimeString();
            
            // Update speedup chart
            if (metrics.jax_speedup && metrics.jax_speedup.length > 0) {
                const latest = metrics.jax_speedup[metrics.jax_speedup.length - 1];
                addDataPoint(charts.speedup, timeLabel, latest.value);
            }
            
            // Update throughput chart
            if (metrics.scaffold_throughput && metrics.scaffold_throughput.length > 0) {
                const latest = metrics.scaffold_throughput[metrics.scaffold_throughput.length - 1];
                addDataPoint(charts.throughput, timeLabel, latest.value);
            }
            
            // Update health chart
            if (metrics.cpu_usage_percent && metrics.memory_usage_percent) {
                const cpuLatest = metrics.cpu_usage_percent[metrics.cpu_usage_percent.length - 1];
                const memLatest = metrics.memory_usage_percent[metrics.memory_usage_percent.length - 1];
                
                charts.health.data.labels.push(timeLabel);
                charts.health.data.datasets[0].data.push(cpuLatest ? cpuLatest.value : 0);
                charts.health.data.datasets[1].data.push(memLatest ? memLatest.value : 0);
                
                // Keep only last 20 points
                if (charts.health.data.labels.length > 20) {
                    charts.health.data.labels.shift();
                    charts.health.data.datasets[0].data.shift();
                    charts.health.data.datasets[1].data.shift();
                }
                
                charts.health.update();
            }
            
            // Update collaboration chart with mock data
            const collaborationScores = [0.95, 0.88, 0.92, 0.85, 0.90];
            charts.collaboration.data.datasets[0].data = collaborationScores;
            charts.collaboration.update();
        }

        function addDataPoint(chart, label, value) {
            chart.data.labels.push(label);
            chart.data.datasets[0].data.push(value);
            
            // Keep only last 20 points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update();
        }

        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (!alerts || alerts.length === 0) {
                alertsList.innerHTML = '<div class="loading">No active alerts - System running optimally 🚀</div>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <strong>🚨 ${alert.severity.toUpperCase()}: ${alert.condition_name}</strong><br>
                    ${alert.message}<br>
                    <small>Triggered: ${new Date(alert.triggered_at).toLocaleString()}</small>
                </div>
            `).join('');
        }

        function updateCostMonitoring(overview) {
            // Estimate costs based on usage
            const scaffoldsProcessed = overview.scaffolds_processed || 0;
            const uptimeHours = (overview.uptime_seconds || 0) / 3600;
            
            // Mock cost calculations (would be real in production)
            const hourlyRate = 0.50; // $0.50/hour estimate
            const hourlyCost = hourlyRate;
            const dailyCost = hourlyCost * 24;
            const monthlyCost = dailyCost * 30;
            const costPerMillion = scaffoldsProcessed > 0 ? (hourlyCost * uptimeHours) / (scaffoldsProcessed / 1000000) : 0;
            
            document.getElementById('hourlyComputeCost').textContent = `$${hourlyCost.toFixed(2)}`;
            document.getElementById('dailyCost').textContent = `$${dailyCost.toFixed(2)}`;
            document.getElementById('monthlyCost').textContent = `$${monthlyCost.toFixed(2)}`;
            document.getElementById('costPerMillion').textContent = costPerMillion > 0 ? `$${costPerMillion.toFixed(2)}` : '$--';
        }

        function getStatusClass(status) {
            switch (status.toLowerCase()) {
                case 'healthy': return 'revolutionary';
                case 'degraded': return 'achievement';
                case 'unhealthy': return 'baseline';
                default: return 'baseline';
            }
        }

        function getSpeedupClass(speedup) {
            if (speedup >= 100) return 'status-revolutionary epic-glow';
            if (speedup >= 10) return 'status-achievement';
            return 'status-baseline';
        }

        function formatUptime(seconds) {
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            
            if (days > 0) return `${days}d ${hours}h ${mins}m`;
            if (hours > 0) return `${hours}h ${mins}m`;
            return `${mins}m`;
        }

        // Auto-refresh every 30 seconds if no WebSocket connection
        setInterval(() => {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                loadInitialData();
            }
        }, 30000);
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=dashboard_html)

@router.get("/dashboard-data")
async def get_dashboard_data():
    """
    📊 DASHBOARD DATA API - Dados épicos para dashboard
    
    Retorna todos os dados necessários para o dashboard:
    - Performance metrics
    - System health
    - Active alerts
    - Cost information
    """
    try:
        monitoring = await get_monitoring()
        dashboard_data = await monitoring.get_dashboard_data()
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    🔄 WEBSOCKET STREAM - Updates em tempo real
    
    WebSocket para streaming de dados em tempo real para o dashboard.
    """
    await manager.connect(websocket)
    
    try:
        monitoring = await get_monitoring()
        
        while True:
            # Get latest dashboard data
            dashboard_data = await monitoring.get_dashboard_data()
            
            # Send to client
            await manager.send_personal_message(
                json.dumps(dashboard_data, default=str), 
                websocket
            )
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/metrics")
async def get_metrics():
    """
    📈 METRICS API - Métricas de performance
    """
    try:
        monitoring = await get_monitoring()
        return await monitoring.get_metrics_summary()
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{metric_name}")
async def get_specific_metric(metric_name: str, limit: int = 100):
    """
    📊 SPECIFIC METRIC API - Métrica específica
    """
    try:
        monitoring = await get_monitoring()
        metrics = await monitoring.get_recent_metrics(metric_name, limit)
        return {"metric_name": metric_name, "data": metrics}
    except Exception as e:
        logger.error(f"Error getting metric {metric_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_health_status():
    """
    🏥 HEALTH API - Status de saúde do sistema
    """
    try:
        monitoring = await get_monitoring()
        summary = await monitoring.get_metrics_summary()
        return summary["health_status"]
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_active_alerts():
    """
    🚨 ALERTS API - Alertas ativos
    """
    try:
        monitoring = await get_monitoring()
        return await monitoring.get_active_alerts()
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_recent_logs(level: Optional[str] = None, limit: int = 100):
    """
    📝 LOGS API - Logs estruturados recentes
    """
    try:
        monitoring = await get_monitoring()
        return await monitoring.get_recent_logs(level, limit)
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost-analysis")
async def get_cost_analysis():
    """
    💰 COST ANALYSIS API - Análise de custos épica
    
    Análise detalhada de custos operacionais:
    - Compute costs per hour/day/month
    - Cost per scaffold processed
    - Cost optimization recommendations
    """
    try:
        monitoring = await get_monitoring()
        summary = await monitoring.get_metrics_summary()
        
        # Calculate cost estimates
        uptime_hours = summary["performance_stats"]["uptime_seconds"] / 3600
        scaffolds_processed = summary["performance_stats"]["scaffold_processed"]
        
        # Mock cost calculations (would be integrated with actual GCP billing)
        hourly_compute_cost = 0.50  # $0.50/hour estimate for Cloud Run
        vertex_ai_cost_per_1k = 0.01  # $0.01 per 1000 Vertex AI requests
        storage_cost_daily = 0.05  # $0.05/day for logs and metrics storage
        
        hourly_cost = hourly_compute_cost
        daily_cost = hourly_cost * 24 + storage_cost_daily
        monthly_cost = daily_cost * 30
        
        # Cost per million scaffolds
        cost_per_million = 0
        if scaffolds_processed > 0:
            total_cost_so_far = uptime_hours * hourly_compute_cost
            cost_per_million = (total_cost_so_far / scaffolds_processed) * 1000000
        
        # Optimization recommendations
        recommendations = []
        
        if uptime_hours > 0:
            avg_throughput = scaffolds_processed / uptime_hours
            if avg_throughput < 50:
                recommendations.append("Consider optimizing batch sizes for better throughput")
            if cost_per_million > 10:
                recommendations.append("High cost per million scaffolds - review instance configuration")
        
        recommendations.extend([
            "Consider using preemptible instances for batch processing",
            "Implement request batching for Vertex AI calls",
            "Use Cloud Storage lifecycle policies for log retention"
        ])
        
        cost_analysis = {
            "current_costs": {
                "hourly_compute": hourly_cost,
                "daily_estimated": daily_cost,
                "monthly_projection": monthly_cost,
                "cost_per_million_scaffolds": cost_per_million
            },
            "usage_metrics": {
                "uptime_hours": uptime_hours,
                "scaffolds_processed": scaffolds_processed,
                "average_throughput_per_hour": scaffolds_processed / max(uptime_hours, 1)
            },
            "cost_breakdown": {
                "compute": hourly_cost,
                "vertex_ai_calls": vertex_ai_cost_per_1k * (scaffolds_processed / 1000),
                "storage": storage_cost_daily / 24,  # per hour
                "networking": 0.02  # minimal networking costs
            },
            "optimization_recommendations": recommendations,
            "cost_targets": {
                "target_cost_per_million": 5.0,
                "target_daily_cost": 10.0,
                "efficiency_score": min(100, (5.0 / max(cost_per_million, 0.1)) * 100)
            }
        }
        
        return cost_analysis
        
    except Exception as e:
        logger.error(f"Error getting cost analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to broadcast updates
async def broadcast_dashboard_updates():
    """Background task to broadcast dashboard updates to all connected clients."""
    while True:
        try:
            if manager.active_connections:
                monitoring = await get_monitoring()
                dashboard_data = await monitoring.get_dashboard_data()
                await manager.broadcast(json.dumps(dashboard_data, default=str))
            
            await asyncio.sleep(10)  # Broadcast every 10 seconds
            
        except Exception as e:
            logger.error(f"Error broadcasting dashboard updates: {e}")
            await asyncio.sleep(30)  # Wait longer on error

# Start background task
import json
asyncio.create_task(broadcast_dashboard_updates())
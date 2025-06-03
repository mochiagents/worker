import asyncio
import time
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import traceback
import psutil
import platform
from worker.core.logging import MochiLogger

@dataclass
class HealthMetrics:
    """Health metrics collected by the heartbeat system."""
    timestamp: float
    agent_id: str
    uptime_seconds: float
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    mcp_connections_active: int
    mcp_connections_failed: int
    last_successful_query_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class HealthAlert:
    """Represents a health alert condition."""
    name: str
    condition: Callable[[HealthMetrics], bool]
    severity: str  # 'low', 'medium', 'high', 'critical'
    cooldown_seconds: float = 300.0  # 5 minutes default
    last_triggered: Optional[float] = None
    
    def should_trigger(self, metrics: HealthMetrics) -> bool:
        """Check if alert should trigger given current metrics."""
        current_time = time.time()
        
        # Check cooldown
        if self.last_triggered and (current_time - self.last_triggered) < self.cooldown_seconds:
            return False
            
        return self.condition(metrics)
    
    def trigger(self) -> None:
        """Mark alert as triggered."""
        self.last_triggered = time.time()

class HeartbeatManager:
    """
    Enhanced heartbeat system with health monitoring, metrics collection,
    and configurable alerting capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        interval_seconds: float = 60.0,
        logger: Optional[MochiLogger] = None,
        enable_system_metrics: bool = True,
        enable_alerts: bool = True,
        metrics_history_size: int = 100
    ):
        self.agent_id = agent_id
        self.interval_seconds = interval_seconds
        self.logger = logger
        self.enable_system_metrics = enable_system_metrics
        self.enable_alerts = enable_alerts
        self.metrics_history_size = metrics_history_size
        
        # Runtime state
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Metrics and history
        self.current_metrics: Optional[HealthMetrics] = None
        self.metrics_history: List[HealthMetrics] = []
        
        # Task counters (to be updated by agent)
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.last_successful_query_time: Optional[float] = None
        
        # MCP connection tracking (to be updated by agent)
        self.mcp_connections_active = 0
        self.mcp_connections_failed = 0
        
        # Alert system
        self.alerts: List[HealthAlert] = []
        self._setup_default_alerts()
        
        # Callback for external health monitoring
        self.health_callbacks: List[Callable[[HealthMetrics], None]] = []
        
    def _setup_default_alerts(self):
        """Setup default health alerts."""
        if not self.enable_alerts:
            return
            
        # High memory usage alert
        self.alerts.append(HealthAlert(
            name="high_memory_usage",
            condition=lambda m: m.memory_percent > 85.0,
            severity="high",
            cooldown_seconds=300.0
        ))
        
        # High CPU usage alert
        self.alerts.append(HealthAlert(
            name="high_cpu_usage", 
            condition=lambda m: m.cpu_percent > 90.0,
            severity="high",
            cooldown_seconds=300.0
        ))
        
        # No successful queries in last hour
        self.alerts.append(HealthAlert(
            name="no_recent_queries",
            condition=lambda m: (
                m.last_successful_query_time is None or 
                (time.time() - m.last_successful_query_time) > 3600
            ),
            severity="medium",
            cooldown_seconds=1800.0  # 30 minutes
        ))
        
        # High failure rate
        self.alerts.append(HealthAlert(
            name="high_failure_rate",
            condition=lambda m: (
                m.completed_tasks + m.failed_tasks > 10 and
                m.failed_tasks / (m.completed_tasks + m.failed_tasks) > 0.5
            ),
            severity="high",
            cooldown_seconds=600.0  # 10 minutes
        ))
        
        # All MCP connections failed
        self.alerts.append(HealthAlert(
            name="all_mcp_connections_failed",
            condition=lambda m: (
                m.mcp_connections_active == 0 and 
                m.mcp_connections_failed > 0
            ),
            severity="critical",
            cooldown_seconds=180.0  # 3 minutes
        ))
    
    def add_alert(self, alert: HealthAlert):
        """Add a custom health alert."""
        self.alerts.append(alert)
    
    def add_health_callback(self, callback: Callable[[HealthMetrics], None]):
        """Add a callback to be called on each heartbeat with current metrics."""
        self.health_callbacks.append(callback)
    
    def update_task_counters(
        self, 
        active: Optional[int] = None,
        completed_delta: int = 0,
        failed_delta: int = 0,
        last_successful: Optional[float] = None
    ):
        """Update task-related counters."""
        if active is not None:
            self.active_tasks = active
        self.completed_tasks += completed_delta
        self.failed_tasks += failed_delta
        if last_successful is not None:
            self.last_successful_query_time = last_successful
    
    def update_mcp_counters(self, active: int, failed: int):
        """Update MCP connection counters."""
        self.mcp_connections_active = active
        self.mcp_connections_failed = failed
        
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        metrics = {}
        
        if self.enable_system_metrics:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                process = psutil.Process()
                process_memory = process.memory_info()
                
                metrics.update({
                    'memory_usage_mb': process_memory.rss / 1024 / 1024,
                    'memory_percent': memory.percent,
                    'cpu_percent': psutil.cpu_percent(interval=0.1)
                })
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to collect system metrics: {e}")
                metrics.update({
                    'memory_usage_mb': 0.0,
                    'memory_percent': 0.0,
                    'cpu_percent': 0.0
                })
        else:
            metrics.update({
                'memory_usage_mb': 0.0,
                'memory_percent': 0.0,
                'cpu_percent': 0.0
            })
            
        return metrics
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics."""
        current_time = time.time()
        system_metrics = self._collect_system_metrics()
        
        metrics = HealthMetrics(
            timestamp=current_time,
            agent_id=self.agent_id,
            uptime_seconds=current_time - self.start_time,
            memory_usage_mb=system_metrics['memory_usage_mb'],
            memory_percent=system_metrics['memory_percent'],
            cpu_percent=system_metrics['cpu_percent'],
            active_tasks=self.active_tasks,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            mcp_connections_active=self.mcp_connections_active,
            mcp_connections_failed=self.mcp_connections_failed,
            last_successful_query_time=self.last_successful_query_time
        )
        
        return metrics
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check all alerts against current metrics."""
        for alert in self.alerts:
            try:
                if alert.should_trigger(metrics):
                    alert.trigger()
                    self._handle_alert(alert, metrics)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error checking alert '{alert.name}': {e}", exc_info=True)
    
    def _handle_alert(self, alert: HealthAlert, metrics: HealthMetrics):
        """Handle a triggered alert."""
        if self.logger:
            self.logger.warning(
                f"Health alert '{alert.name}' triggered with severity '{alert.severity}'. "
                f"Metrics: {json.dumps(metrics.to_dict(), indent=2)}",
                event_type="HEALTH_ALERT",
                extra={
                    "alert_name": alert.name,
                    "alert_severity": alert.severity,
                    "health_metrics": metrics.to_dict()
                }
            )
    
    def _store_metrics(self, metrics: HealthMetrics):
        """Store metrics in history with size limit."""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history = self.metrics_history[-self.metrics_history_size:]
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        if self.logger:
            self.logger.info(
                f"Heartbeat started for agent '{self.agent_id}' with {self.interval_seconds}s interval",
                event_type="HEARTBEAT_START"
            )
        
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check alerts
                if self.enable_alerts:
                    self._check_alerts(metrics)
                
                # Call health callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error in health callback: {e}", exc_info=True)
                
                # Log heartbeat
                if self.logger:
                    uptime_hours = metrics.uptime_seconds / 3600
                    self.logger.info(
                        f"Heartbeat pulse - Agent: {self.agent_id}, "
                        f"Uptime: {uptime_hours:.1f}h, "
                        f"Memory: {metrics.memory_usage_mb:.1f}MB ({metrics.memory_percent:.1f}%), "
                        f"CPU: {metrics.cpu_percent:.1f}%, "
                        f"Tasks: {metrics.active_tasks} active, "
                        f"{metrics.completed_tasks} completed, "
                        f"{metrics.failed_tasks} failed, "
                        f"MCP: {metrics.mcp_connections_active} active connections",
                        event_type="HEARTBEAT_PULSE",
                        extra={"health_metrics": metrics.to_dict()}
                    )
                
                # Wait for next heartbeat or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), 
                        timeout=self.interval_seconds
                    )
                    # If we get here, shutdown was signaled
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue to next heartbeat
                    continue
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in heartbeat loop: {e}", exc_info=True)
                # Sleep before retrying
                await asyncio.sleep(min(self.interval_seconds, 10.0))
        
        if self.logger:
            self.logger.info(
                f"Heartbeat stopped for agent '{self.agent_id}'",
                event_type="HEARTBEAT_STOP"
            )
    
    async def start(self):
        """Start the heartbeat system."""
        if self.is_running:
            if self.logger:
                self.logger.warning("Heartbeat already running")
            return
        
        self.shutdown_event.clear()
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.is_running = True
        
        if self.logger:
            self.logger.info(f"Heartbeat system started for agent '{self.agent_id}'")
    
    async def stop(self):
        """Stop the heartbeat system."""
        if not self.is_running:
            return
        
        self.shutdown_event.set()
        
        if self.heartbeat_task and not self.heartbeat_task.done():
            try:
                await asyncio.wait_for(self.heartbeat_task, timeout=5.0)
            except asyncio.TimeoutError:
                if self.logger:
                    self.logger.warning("Heartbeat task did not stop gracefully, cancelling")
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
        
        self.is_running = False
        
        if self.logger:
            self.logger.info(f"Heartbeat system stopped for agent '{self.agent_id}'")
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get the most recent health metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[HealthMetrics]:
        """Get metrics history, optionally limited to most recent N entries."""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:] if limit > 0 else []
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's health status."""
        if not self.current_metrics:
            return {"status": "unknown", "message": "No metrics available"}
        
        metrics = self.current_metrics
        status = "healthy"
        issues = []
        
        # Check for various health issues
        if metrics.memory_percent > 85:
            status = "warning"
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.cpu_percent > 90:
            status = "warning" 
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.mcp_connections_active == 0 and metrics.mcp_connections_failed > 0:
            status = "critical"
            issues.append("All MCP connections failed")
        
        if metrics.last_successful_query_time:
            time_since_success = time.time() - metrics.last_successful_query_time
            if time_since_success > 3600:  # 1 hour
                status = "warning"
                issues.append(f"No successful queries in {time_since_success/3600:.1f} hours")
        
        # Check failure rate
        total_tasks = metrics.completed_tasks + metrics.failed_tasks
        if total_tasks > 10:
            failure_rate = metrics.failed_tasks / total_tasks
            if failure_rate > 0.5:
                status = "critical"
                issues.append(f"High failure rate: {failure_rate*100:.1f}%")
        
        return {
            "status": status,
            "uptime_hours": metrics.uptime_seconds / 3600,
            "memory_usage_mb": metrics.memory_usage_mb,
            "memory_percent": metrics.memory_percent,
            "cpu_percent": metrics.cpu_percent,
            "active_tasks": metrics.active_tasks,
            "completed_tasks": metrics.completed_tasks,
            "failed_tasks": metrics.failed_tasks,
            "mcp_connections_active": metrics.mcp_connections_active,
            "issues": issues,
            "last_heartbeat": datetime.fromtimestamp(
                metrics.timestamp, tz=timezone.utc
            ).isoformat()
        } 
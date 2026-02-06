import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .data_generator import NUM_ZONES, ZONE_LENGTH_KM, CORRIDOR_LENGTH_KM


DARK_TEMPLATE = "plotly_dark"

STATUS_COLORS = {
    "idle": "#4CAF50",
    "en_route": "#2196F3",
    "boarding": "#FF9800",
}


def plot_corridor(snapshot):
    """Visualize pods along The Line corridor as colored dots.

    Pods are colored by status (idle=green, en_route=blue, boarding=orange).
    Zone queues are shown as a bar overlay below the corridor.

    Args:
        snapshot: Dict from SimulationEnvironment.get_snapshot().

    Returns:
        Plotly Figure object.
    """
    positions = []
    colors = []
    hover_texts = []

    for pos_km, status in snapshot["pod_positions"]:
        positions.append(pos_km)
        colors.append(STATUS_COLORS.get(status, "#999999"))
        hover_texts.append(f"Position: {pos_km:.1f}km | Status: {status}")

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        subplot_titles=["Pod Positions Along The Line", "Passenger Queues by Zone"],
        vertical_spacing=0.15,
    )

    y_jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(positions))
    fig.add_trace(
        go.Scatter(
            x=positions, y=y_jitter, mode="markers",
            marker=dict(color=colors, size=4, opacity=0.7),
            text=hover_texts, hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=1,
    )

    for status, color in STATUS_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=8),
                name=status.replace("_", " ").title(),
            ),
            row=1, col=1,
        )

    fig.add_shape(
        type="line", x0=0, y0=0, x1=CORRIDOR_LENGTH_KM, y1=0,
        line=dict(color="white", width=2, dash="dot"),
        row=1, col=1,
    )

    zone_centers = [(z + 0.5) * ZONE_LENGTH_KM for z in range(NUM_ZONES)]
    fig.add_trace(
        go.Bar(
            x=zone_centers, y=snapshot["zone_queues"],
            marker_color="#EF5350", opacity=0.8,
            name="Queue Length", width=ZONE_LENGTH_KM * 0.8,
        ),
        row=2, col=1,
    )

    hour = snapshot["hour"]
    minute = snapshot["minute"]
    fig.update_layout(
        template=DARK_TEMPLATE, height=500,
        title_text=f"The Line Corridor | {hour:02d}:{minute:02d} | "
                   f"Waiting: {snapshot['total_waiting']} | "
                   f"Completed: {snapshot['total_completed']}",
        xaxis=dict(title="Position (km)", range=[0, CORRIDOR_LENGTH_KM]),
        yaxis=dict(title="", range=[-1, 1], showticklabels=False),
        xaxis2=dict(title="Position (km)", range=[0, CORRIDOR_LENGTH_KM]),
        yaxis2=dict(title="Queue Length"),
    )

    return fig


def plot_demand_heatmap(demand_summary):
    """Create a zone-by-hour demand heatmap.

    Args:
        demand_summary: DataFrame from DemandGenerator.generate_demand_summary()
            with columns like zone_X_dest for each zone.

    Returns:
        Plotly Figure object.
    """
    hourly = demand_summary.groupby("hour").sum()

    dest_cols = [f"zone_{z}_dest" for z in range(NUM_ZONES)]
    matrix = hourly[dest_cols].values.T

    fig = px.imshow(
        matrix,
        labels=dict(x="Hour of Day", y="Zone ID", color="Trip Demand"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=[str(z) for z in range(NUM_ZONES)],
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title="Trip Demand Heatmap (Destination Zones)",
    )

    fig.update_layout(template=DARK_TEMPLATE, height=600)
    return fig


def plot_wait_time_distribution(wait_times):
    """Plot histogram of passenger wait times.

    Args:
        wait_times: List of wait times in minutes.

    Returns:
        Plotly Figure object.
    """
    if not wait_times:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="No wait time data")
        return fig

    fig = px.histogram(
        x=wait_times, nbins=50,
        labels=dict(x="Wait Time (minutes)", y="Passenger Count"),
        title="Passenger Wait Time Distribution",
        color_discrete_sequence=["#42A5F5"],
    )

    avg = np.mean(wait_times)
    p95 = np.percentile(wait_times, 95)

    fig.add_vline(x=avg, line_dash="dash", line_color="#FF9800",
                  annotation_text=f"Mean: {avg:.1f}min")
    fig.add_vline(x=p95, line_dash="dash", line_color="#EF5350",
                  annotation_text=f"P95: {p95:.1f}min")

    fig.update_layout(template=DARK_TEMPLATE, height=400)
    return fig


def plot_utilization_over_time(utilization_data):
    """Plot fleet utilization percentage over the simulation day.

    Args:
        utilization_data: List of (timestep, utilization) tuples.

    Returns:
        Plotly Figure object.
    """
    if not utilization_data:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="No utilization data")
        return fig

    timesteps, utils = zip(*utilization_data)
    hours = [t / 60.0 for t in timesteps]

    window = 30
    smoothed = pd.Series(utils).rolling(window=window, min_periods=1).mean().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=[u * 100 for u in utils],
        mode="lines", name="Raw",
        line=dict(color="#42A5F5", width=1), opacity=0.3,
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=[u * 100 for u in smoothed],
        mode="lines", name=f"{window}-min Average",
        line=dict(color="#FF9800", width=2),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE, height=400,
        title="Fleet Utilization Over Time",
        xaxis_title="Hour of Day",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_throughput_over_time(throughput_data):
    """Plot passenger throughput over the simulation day.

    Args:
        throughput_data: List of (timestep, throughput) tuples.

    Returns:
        Plotly Figure object.
    """
    if not throughput_data:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="No throughput data")
        return fig

    timesteps, throughputs = zip(*throughput_data)
    hours = [t / 60.0 for t in timesteps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=throughputs, mode="lines",
        line=dict(color="#66BB6A", width=2),
        name="Throughput",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE, height=400,
        title="Passenger Throughput (trips/hour)",
        xaxis_title="Hour of Day",
        yaxis_title="Trips per Hour",
    )
    return fig


def plot_zone_service_quality(scores):
    """Plot per-zone service quality scores.

    Args:
        scores: Dict mapping zone_id to quality score (0-100).

    Returns:
        Plotly Figure object.
    """
    zones = list(range(NUM_ZONES))
    values = [scores.get(z, 0) for z in zones]
    positions = [(z + 0.5) * ZONE_LENGTH_KM for z in zones]

    colors = []
    for v in values:
        if v >= 70:
            colors.append("#4CAF50")
        elif v >= 40:
            colors.append("#FF9800")
        else:
            colors.append("#EF5350")

    fig = go.Figure(go.Bar(
        x=positions, y=values,
        marker_color=colors, width=ZONE_LENGTH_KM * 0.8,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE, height=400,
        title="Service Quality by Zone",
        xaxis_title="Position Along The Line (km)",
        yaxis_title="Quality Score (0-100)",
        yaxis=dict(range=[0, 110]),
    )
    return fig


def plot_strategy_comparison(comparison_df):
    """Create side-by-side comparison charts for dispatch strategies.

    Args:
        comparison_df: DataFrame from SimulationMetrics.compare_strategies().

    Returns:
        Plotly Figure object.
    """
    metrics_to_plot = [
        ("Avg Wait (min)", "lower is better"),
        ("Utilization (%)", "higher is better"),
        ("Deadhead (km)", "lower is better"),
        ("Throughput (trips/hr)", "higher is better"),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{m[0]} ({m[1]})" for m in metrics_to_plot],
        vertical_spacing=0.2, horizontal_spacing=0.15,
    )

    colors = ["#42A5F5", "#66BB6A", "#FF9800", "#EF5350"]

    for idx, (metric, _) in enumerate(metrics_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1

        fig.add_trace(
            go.Bar(
                x=comparison_df["Strategy"],
                y=comparison_df[metric],
                marker_color=colors[:len(comparison_df)],
                text=[f"{v:.1f}" for v in comparison_df[metric]],
                textposition="outside",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        template=DARK_TEMPLATE, height=600,
        title_text="Strategy Comparison Dashboard",
    )
    return fig


def plot_pod_status_over_time(snapshots):
    """Plot stacked area chart of pod statuses over time.

    Args:
        snapshots: List of snapshot dicts from simulation.

    Returns:
        Plotly Figure object.
    """
    if not snapshots:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="No data")
        return fig

    hours = [s["timestep"] / 60.0 for s in snapshots]
    idle = [s["status_counts"].get("idle", 0) for s in snapshots]
    en_route = [s["status_counts"].get("en_route", 0) for s in snapshots]
    boarding = [s["status_counts"].get("boarding", 0) for s in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=idle, mode="lines", name="Idle",
        line=dict(color=STATUS_COLORS["idle"]), stackgroup="one",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=en_route, mode="lines", name="En Route",
        line=dict(color=STATUS_COLORS["en_route"]), stackgroup="one",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=boarding, mode="lines", name="Boarding",
        line=dict(color=STATUS_COLORS["boarding"]), stackgroup="one",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE, height=400,
        title="Pod Status Distribution Over Time",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Pods",
    )
    return fig

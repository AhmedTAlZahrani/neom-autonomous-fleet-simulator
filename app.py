import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_generator import DemandGenerator, NUM_ZONES, TOTAL_TIMESTEPS
from src.environment import SimulationEnvironment
from src.fleet_optimizer import get_dispatcher
from src.metrics import SimulationMetrics
from src.visualization import (
    plot_corridor,
    plot_demand_heatmap,
    plot_wait_time_distribution,
    plot_utilization_over_time,
    plot_throughput_over_time,
    plot_zone_service_quality,
    plot_strategy_comparison,
    plot_pod_status_over_time,
)


st.set_page_config(
    page_title="NEOM Fleet Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_simulation(fleet_size, pod_capacity, strategy_name, n_steps, seed=42):
    """Run a fleet simulation with the given parameters.

    Args:
        fleet_size: Number of pods in the fleet.
        pod_capacity: Maximum passengers per pod.
        strategy_name: Dispatch strategy ('greedy', 'lp', 'qlearning').
        n_steps: Number of timesteps to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (SimulationEnvironment, SimulationMetrics, list of snapshots).
    """
    gen = DemandGenerator(seed=seed)
    gen.add_random_events(n_events=5)
    demand_data = gen.generate_full_day()

    env = SimulationEnvironment(fleet_size=fleet_size, pod_capacity=pod_capacity, seed=seed)
    dispatcher = get_dispatcher(strategy_name)
    metrics = SimulationMetrics()
    snapshots = []

    progress = st.progress(0)
    status_text = st.empty()

    steps = min(n_steps, len(demand_data))

    for t in range(steps):
        od_matrix = demand_data[t]["od_matrix"]
        env.inject_passengers(od_matrix)

        state = env.get_state()
        assignments = dispatcher.dispatch(state)
        env.apply_dispatch(assignments)
        env.step()

        if t % 10 == 0:
            metrics.record_snapshot(env)
            snapshots.append(env.get_snapshot())

        if t % 60 == 0:
            progress.progress(t / steps)
            hour = t // 60
            status_text.text(f"Simulating hour {hour:02d}:00 | "
                             f"Completed: {len(env.completed_passengers)} trips")

    progress.progress(1.0)
    status_text.text(f"Simulation complete: {len(env.completed_passengers)} trips")

    metrics.finalize(env)
    return env, metrics, snapshots


@st.cache_data
def generate_demand_summary(seed=42):
    """Generate and cache demand summary data.

    Args:
        seed: Random seed.

    Returns:
        DataFrame with demand summary.
    """
    gen = DemandGenerator(seed=seed)
    gen.add_random_events(n_events=5)
    demand_data = gen.generate_full_day()
    return gen.generate_demand_summary(demand_data)


def main():
    """Main Streamlit application entry point."""
    st.title("NEOM Autonomous Fleet Simulator")
    st.markdown("**Fleet Dispatch Optimization for The Line's Autonomous Pod Network**")

    st.sidebar.header("Simulation Parameters")
    fleet_size = st.sidebar.slider("Fleet Size", 100, 1000, 500, step=50)
    pod_capacity = st.sidebar.slider("Pod Capacity", 2, 12, 6)
    strategy = st.sidebar.selectbox(
        "Dispatch Strategy",
        ["greedy", "lp", "qlearning"],
        format_func=lambda x: {"greedy": "Greedy (Nearest Pod)",
                                "lp": "Linear Programming",
                                "qlearning": "Q-Learning"}[x],
    )
    sim_hours = st.sidebar.slider("Simulation Duration (hours)", 1, 24, 12)
    n_steps = sim_hours * 60

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Simulation", "Demand Patterns", "Performance", "Strategy Comparison"
    ])

    with tab1:
        st.header("Live Simulation")

        if st.button("Run Simulation", type="primary"):
            env, metrics, snapshots = run_simulation(
                fleet_size, pod_capacity, strategy, n_steps
            )

            st.session_state["metrics"] = metrics
            st.session_state["snapshots"] = snapshots
            st.session_state["env"] = env

            if snapshots:
                st.subheader("Final Corridor State")
                fig = plot_corridor(snapshots[-1])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Pod Status Over Time")
                fig_status = plot_pod_status_over_time(snapshots)
                st.plotly_chart(fig_status, use_container_width=True)

            summary = metrics.summary()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Wait", f"{summary['avg_wait_min']:.1f} min")
            col2.metric("Utilization", f"{summary['fleet_utilization']:.1f}%")
            col3.metric("Throughput", f"{summary['throughput_trips_hr']:.0f} trips/hr")
            col4.metric("Service Rate", f"{summary['service_rate']:.1f}%")

    with tab2:
        st.header("Demand Patterns")

        demand_df = generate_demand_summary()
        fig_heatmap = plot_demand_heatmap(demand_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Hourly Demand Totals")
        hourly_totals = demand_df.groupby("hour")["total_trips"].sum().reset_index()
        hourly_totals.columns = ["Hour", "Total Trips"]
        st.bar_chart(hourly_totals.set_index("Hour"))

        st.subheader("Scheduled Events")
        st.markdown("""
        | Event Type | Zone | Start Hour | Duration | Magnitude |
        |-----------|------|-----------|----------|-----------|
        | Concert | Leisure Zone 7 | 19:00 | 3.0h | 2.5x |
        | Sports | Leisure Zone 17 | 15:00 | 2.5h | 2.0x |
        | Conference | Cultural Zone 27 | 09:00 | 4.0h | 1.5x |
        | Festival | Leisure Zone 37 | 12:00 | 5.0h | 3.0x |
        | Exhibition | Cultural Zone 47 | 10:00 | 3.0h | 1.3x |
        """)

    with tab3:
        st.header("Performance Metrics")

        if "metrics" not in st.session_state:
            st.info("Run a simulation first to see performance metrics.")
        else:
            metrics = st.session_state["metrics"]
            snapshots = st.session_state["snapshots"]

            summary = metrics.summary()

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Wait", f"{summary['avg_wait_min']:.1f} min")
            col1.metric("Median Wait", f"{summary['median_wait_min']:.1f} min")
            col1.metric("P95 Wait", f"{summary['p95_wait_min']:.1f} min")

            col2.metric("Fleet Utilization", f"{summary['fleet_utilization']:.1f}%")
            col2.metric("Deadhead Distance", f"{summary['deadhead_km']:.0f} km")
            col2.metric("Total Distance", f"{summary['total_km']:.0f} km")

            col3.metric("Throughput", f"{summary['throughput_trips_hr']:.0f} trips/hr")
            col3.metric("Energy", f"{summary['energy_kwh']:.0f} kWh")
            col3.metric("Abandoned", f"{summary['abandoned_trips']}")

            st.subheader("Wait Time Distribution")
            wait_times = metrics.get_wait_time_distribution()
            fig_wait = plot_wait_time_distribution(wait_times)
            st.plotly_chart(fig_wait, use_container_width=True)

            st.subheader("Fleet Utilization Over Time")
            util_data = metrics.get_utilization_over_time()
            fig_util = plot_utilization_over_time(util_data)
            st.plotly_chart(fig_util, use_container_width=True)

            st.subheader("Throughput Over Time")
            throughput_data = metrics.get_throughput_over_time()
            fig_through = plot_throughput_over_time(throughput_data)
            st.plotly_chart(fig_through, use_container_width=True)

            st.subheader("Zone Service Quality")
            scores = metrics.per_zone_service_quality()
            fig_quality = plot_zone_service_quality(scores)
            st.plotly_chart(fig_quality, use_container_width=True)

    with tab4:
        st.header("Strategy Comparison")

        if st.button("Run All Strategies", type="primary", key="compare"):
            strategies = ["greedy", "lp", "qlearning"]
            strategy_names = ["Greedy", "Linear Programming", "Q-Learning"]
            all_metrics = []

            compare_steps = min(n_steps, 720)

            for name, sname in zip(strategies, strategy_names):
                st.text(f"Running {sname}...")
                _, m, _ = run_simulation(fleet_size, pod_capacity, name, compare_steps)
                all_metrics.append(m)

            comparison_df = SimulationMetrics.compare_strategies(all_metrics, strategy_names)
            st.session_state["comparison_df"] = comparison_df

        if "comparison_df" in st.session_state:
            comparison_df = st.session_state["comparison_df"]

            st.subheader("Comparison Table")
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("Visual Comparison")
            fig_compare = plot_strategy_comparison(comparison_df)
            st.plotly_chart(fig_compare, use_container_width=True)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_generator import NUM_ZONES, ZONE_LENGTH_KM


ENERGY_KWH_PER_KM = 0.15


class SimulationMetrics:

    def __init__(self):
        self.snapshots = []
        self.completed_passengers = []
        self.abandoned_passengers = []
        self.pod_stats = []

    def record_snapshot(self, env):
        """Record a simulation state snapshot for time-series metrics.

        Args:
            env: SimulationEnvironment instance.
        """
        snapshot = env.get_snapshot()
        self.snapshots.append(snapshot)

    def finalize(self, env):
        """Collect final simulation data from the environment.

        Args:
            env: SimulationEnvironment instance after simulation completes.
        """
        self.completed_passengers = env.completed_passengers
        self.abandoned_passengers = env.abandoned_passengers
        self.pod_stats = [
            {
                "pod_id": pod.pod_id,
                "km_traveled": pod.km_traveled,
                "km_traveled_empty": pod.km_traveled_empty,
                "trips_completed": pod.trips_completed,
            }
            for pod in env.pods
        ]
        print(f"Metrics finalized: {len(self.completed_passengers)} completed, "
              f"{len(self.abandoned_passengers)} abandoned")

    def average_wait_time(self):
        """Calculate average passenger wait time in minutes.

        Returns:
            Float average wait time, or 0.0 if no completed trips.
        """
        if not self.completed_passengers:
            return 0.0
        waits = [p.pickup_timestep - p.arrival_timestep for p in self.completed_passengers]
        return np.mean(waits)

    def median_wait_time(self):
        """Calculate median passenger wait time in minutes.

        Returns:
            Float median wait time.
        """
        if not self.completed_passengers:
            return 0.0
        waits = [p.pickup_timestep - p.arrival_timestep for p in self.completed_passengers]
        return np.median(waits)

    def p95_wait_time(self):
        """Calculate 95th percentile passenger wait time.

        Returns:
            Float p95 wait time in minutes.
        """
        if not self.completed_passengers:
            return 0.0
        waits = [p.pickup_timestep - p.arrival_timestep for p in self.completed_passengers]
        return np.percentile(waits, 95)

    def fleet_utilization(self):
        """Calculate fleet utilization rate.

        Measures the percentage of time pods were occupied (not idle).

        Returns:
            Float utilization rate between 0 and 1.
        """
        if not self.snapshots:
            return 0.0

        total_busy = 0
        total_pods = 0
        for snap in self.snapshots:
            counts = snap["status_counts"]
            busy = counts.get("en_route", 0) + counts.get("boarding", 0)
            total = sum(counts.values())
            total_busy += busy
            total_pods += total

        return total_busy / max(total_pods, 1)

    def deadhead_km(self):
        """Calculate total deadhead (empty travel) kilometers.

        Returns:
            Float total empty kilometers driven by the fleet.
        """
        return sum(ps["km_traveled_empty"] for ps in self.pod_stats)

    def total_km(self):
        """Calculate total kilometers driven by the fleet.

        Returns:
            Float total kilometers.
        """
        return sum(ps["km_traveled"] for ps in self.pod_stats)

    def passenger_throughput(self):
        """Calculate average passenger throughput in trips per hour.

        Returns:
            Float trips per hour over the simulation period.
        """
        if not self.snapshots:
            return 0.0
        hours = len(self.snapshots) / 60.0
        return len(self.completed_passengers) / max(hours, 0.01)

    def energy_consumption(self):
        """Estimate total energy consumption in kWh.

        Uses a constant rate of 0.15 kWh/km for electric pods.

        Returns:
            Float total energy consumption in kWh.
        """
        return self.total_km() * ENERGY_KWH_PER_KM

    def per_zone_service_quality(self):
        """Calculate service quality score for each zone.

        Score is based on average wait time and abandonment rate.
        Higher is better (0-100 scale).

        Returns:
            Dict mapping zone_id to quality score.
        """
        zone_waits = {z: [] for z in range(NUM_ZONES)}
        zone_abandoned = {z: 0 for z in range(NUM_ZONES)}

        for p in self.completed_passengers:
            wait = p.pickup_timestep - p.arrival_timestep
            zone_waits[p.origin_zone].append(wait)

        for p in self.abandoned_passengers:
            zone_abandoned[p.origin_zone] += 1

        scores = {}
        for z in range(NUM_ZONES):
            if zone_waits[z]:
                avg_wait = np.mean(zone_waits[z])
                wait_score = max(0, 100 - avg_wait * 5)
            else:
                wait_score = 50.0

            total = len(zone_waits[z]) + zone_abandoned[z]
            if total > 0:
                abandon_rate = zone_abandoned[z] / total
                abandon_penalty = abandon_rate * 40
            else:
                abandon_penalty = 0

            scores[z] = round(max(0, wait_score - abandon_penalty), 1)

        return scores

    def get_wait_time_distribution(self):
        """Get the full distribution of wait times.

        Returns:
            List of wait times in minutes for all completed passengers.
        """
        return [p.pickup_timestep - p.arrival_timestep for p in self.completed_passengers]

    def get_utilization_over_time(self):
        """Get fleet utilization at each recorded timestep.

        Returns:
            List of (timestep, utilization) tuples.
        """
        results = []
        for snap in self.snapshots:
            counts = snap["status_counts"]
            busy = counts.get("en_route", 0) + counts.get("boarding", 0)
            total = sum(counts.values())
            util = busy / max(total, 1)
            results.append((snap["timestep"], util))
        return results

    def get_throughput_over_time(self, window=60):
        """Get rolling throughput over time.

        Args:
            window: Rolling window size in timesteps (minutes).

        Returns:
            List of (timestep, throughput) tuples.
        """
        if not self.snapshots:
            return []

        results = []
        for i, snap in enumerate(self.snapshots):
            start_idx = max(0, i - window)
            trips = snap["total_completed"]
            if start_idx > 0:
                trips -= self.snapshots[start_idx]["total_completed"]
            throughput = trips * (60.0 / min(window, i + 1))
            results.append((snap["timestep"], throughput))
        return results

    def summary(self):
        """Generate a complete metrics summary.

        Returns:
            Dict with all key performance metrics.
        """
        return {
            "avg_wait_min": round(self.average_wait_time(), 2),
            "median_wait_min": round(self.median_wait_time(), 2),
            "p95_wait_min": round(self.p95_wait_time(), 2),
            "fleet_utilization": round(self.fleet_utilization() * 100, 1),
            "deadhead_km": round(self.deadhead_km(), 0),
            "total_km": round(self.total_km(), 0),
            "throughput_trips_hr": round(self.passenger_throughput(), 0),
            "energy_kwh": round(self.energy_consumption(), 0),
            "completed_trips": len(self.completed_passengers),
            "abandoned_trips": len(self.abandoned_passengers),
            "service_rate": round(
                len(self.completed_passengers) /
                max(len(self.completed_passengers) + len(self.abandoned_passengers), 1) * 100,
                1
            ),
        }

    def save_metrics(self, path="output/metrics.json"):
        """Save metrics summary to a JSON file.

        Args:
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Metrics saved to {output}")

    @staticmethod
    def compare_strategies(metrics_list, strategy_names):
        """Create a comparison table across multiple strategies.

        Args:
            metrics_list: List of SimulationMetrics instances.
            strategy_names: List of strategy name strings.

        Returns:
            DataFrame with side-by-side strategy comparison.
        """
        rows = []
        for name, m in zip(strategy_names, metrics_list):
            s = m.summary()
            rows.append({
                "Strategy": name,
                "Avg Wait (min)": s["avg_wait_min"],
                "Median Wait (min)": s["median_wait_min"],
                "P95 Wait (min)": s["p95_wait_min"],
                "Utilization (%)": s["fleet_utilization"],
                "Deadhead (km)": s["deadhead_km"],
                "Throughput (trips/hr)": s["throughput_trips_hr"],
                "Energy (kWh)": s["energy_kwh"],
                "Service Rate (%)": s["service_rate"],
            })

        df = pd.DataFrame(rows)
        print("\n=== Strategy Comparison ===")
        print(df.to_string(index=False))
        return df


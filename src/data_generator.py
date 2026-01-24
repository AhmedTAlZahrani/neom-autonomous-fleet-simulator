import numpy as np
import pandas as pd
from pathlib import Path


ZONE_TYPES = [
    "residential", "commercial", "mixed", "industrial", "leisure", "cultural"
]

CORRIDOR_LENGTH_KM = 170.0
NUM_ZONES = 50
ZONE_LENGTH_KM = CORRIDOR_LENGTH_KM / NUM_ZONES
TIMESTEPS_PER_HOUR = 60
SIMULATION_HOURS = 24
TOTAL_TIMESTEPS = SIMULATION_HOURS * TIMESTEPS_PER_HOUR


class ZoneConfig:

    def __init__(self, zone_id, zone_type, position_km):
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.position_km = position_km
        self.morning_attraction = self._get_attraction("morning")
        self.evening_attraction = self._get_attraction("evening")

    def _get_attraction(self, period):
        """Compute demand attraction factor for a time period.

        Args:
            period: Either 'morning' or 'evening'.

        Returns:
            Float multiplier for demand attraction.
        """
        weights = {
            "morning": {
                "residential": 0.3, "commercial": 2.5, "mixed": 1.5,
                "industrial": 2.0, "leisure": 0.4, "cultural": 0.6,
            },
            "evening": {
                "residential": 2.5, "commercial": 0.3, "mixed": 1.5,
                "industrial": 0.3, "leisure": 2.0, "cultural": 1.8,
            },
        }
        return weights[period][self.zone_type]


class DemandGenerator:

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.zones = self._build_zones()
        self.events = []

    def _build_zones(self):
        """Create zone configurations along the corridor.

        Returns:
            List of ZoneConfig objects for all 50 zones.
        """
        type_pattern = [
            "residential", "residential", "mixed", "commercial", "industrial",
            "mixed", "residential", "leisure", "mixed", "commercial",
        ]
        zones = []
        for i in range(NUM_ZONES):
            zone_type = type_pattern[i % len(type_pattern)]
            position = (i + 0.5) * ZONE_LENGTH_KM
            zones.append(ZoneConfig(i, zone_type, position))
        print(f"Built {len(zones)} zones across {CORRIDOR_LENGTH_KM}km corridor")
        return zones

    def _base_demand_rate(self, hour):
        """Compute base hourly demand rate using bimodal distribution.

        Args:
            hour: Hour of the day (0-23).

        Returns:
            Base demand multiplier for the given hour.
        """
        morning_peak = np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
        evening_peak = np.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2)
        midday = 0.3 * np.exp(-0.5 * ((hour - 12.0) / 2.0) ** 2)
        night_base = 0.05
        return night_base + morning_peak + 0.85 * evening_peak + midday

    def _compute_od_matrix(self, hour, minute):
        """Compute origin-destination demand matrix for a given timestep.

        Args:
            hour: Hour of the day (0-23).
            minute: Minute within the hour (0-59).

        Returns:
            2D numpy array of shape (NUM_ZONES, NUM_ZONES) with trip counts.
        """
        fractional_hour = hour + minute / 60.0
        base_rate = self._base_demand_rate(fractional_hour)

        is_morning = 6 <= hour <= 10
        is_evening = 16 <= hour <= 20

        od = np.zeros((NUM_ZONES, NUM_ZONES))

        for orig in self.zones:
            for dest in self.zones:
                if orig.zone_id == dest.zone_id:
                    continue

                distance = abs(orig.position_km - dest.position_km)
                distance_decay = np.exp(-distance / 40.0)

                if is_morning:
                    direction_factor = (
                        (1.0 - orig.morning_attraction / 3.0) *
                        dest.morning_attraction / 2.5
                    )
                elif is_evening:
                    direction_factor = (
                        (1.0 - orig.evening_attraction / 3.0) *
                        dest.evening_attraction / 2.5
                    )
                else:
                    direction_factor = 0.5

                rate = base_rate * distance_decay * max(direction_factor, 0.05) * 0.15
                trips = self.rng.poisson(max(rate, 0.0))
                od[orig.zone_id, dest.zone_id] = trips

        return od

    def add_event(self, zone_id, start_hour, duration_hours, magnitude):
        """Schedule a stochastic event that causes a demand spike.

        Args:
            zone_id: Zone where the event occurs.
            start_hour: Event start time (fractional hours).
            duration_hours: Duration of the event in hours.
            magnitude: Demand multiplier during the event.
        """
        self.events.append({
            "zone_id": zone_id,
            "start_hour": start_hour,
            "duration_hours": duration_hours,
            "magnitude": magnitude,
        })
        print(f"Event scheduled: zone {zone_id} at hour {start_hour} "
              f"(duration={duration_hours}h, magnitude={magnitude}x)")

    def _apply_events(self, od, hour, minute):
        """Apply event-driven demand spikes to the OD matrix.

        Args:
            od: Origin-destination matrix to modify.
            hour: Current hour.
            minute: Current minute.

        Returns:
            Modified OD matrix with event effects.
        """
        t = hour + minute / 60.0
        for event in self.events:
            start = event["start_hour"]
            end = start + event["duration_hours"]
            if start <= t <= end:
                zid = event["zone_id"]
                od[:, zid] *= event["magnitude"]
                if t >= end - 0.5:
                    od[zid, :] *= event["magnitude"] * 0.8
        return od

    def add_random_events(self, n_events=5):
        """Generate random events throughout the simulation day.

        Args:
            n_events: Number of random events to schedule.
        """
        event_types = [
            ("concert", 2.5, 3.0),
            ("sports", 2.0, 2.5),
            ("conference", 1.5, 4.0),
            ("festival", 3.0, 5.0),
            ("exhibition", 1.3, 3.0),
        ]
        leisure_zones = [z.zone_id for z in self.zones if z.zone_type in ("leisure", "cultural")]

        for _ in range(n_events):
            name, mag, dur = event_types[self.rng.randint(len(event_types))]
            zone_id = self.rng.choice(leisure_zones) if leisure_zones else self.rng.randint(NUM_ZONES)
            start = self.rng.uniform(8.0, 20.0)
            jittered_mag = mag * self.rng.uniform(0.8, 1.2)
            jittered_dur = dur * self.rng.uniform(0.7, 1.3)
            self.add_event(zone_id, round(start, 1), round(jittered_dur, 1), round(jittered_mag, 2))

    def generate_full_day(self):
        """Generate demand data for an entire 24-hour simulation.

        Returns:
            List of dicts, each with keys 'timestep', 'hour', 'minute',
            and 'od_matrix' (NUM_ZONES x NUM_ZONES array).
        """
        print("Generating 24-hour demand data...")
        demand_data = []

        for t in range(TOTAL_TIMESTEPS):
            hour = t // TIMESTEPS_PER_HOUR
            minute = t % TIMESTEPS_PER_HOUR
            od = self._compute_od_matrix(hour, minute)
            od = self._apply_events(od, hour, minute)
            demand_data.append({
                "timestep": t,
                "hour": hour,
                "minute": minute,
                "od_matrix": od,
            })

            if t % 360 == 0:
                total_trips = int(od.sum())
                print(f"  Hour {hour:02d}:{minute:02d} | Demand: {total_trips} trips")

        print(f"Generated {len(demand_data)} timesteps")
        return demand_data

    def generate_demand_summary(self, demand_data):
        """Create a summary DataFrame of hourly demand totals.

        Args:
            demand_data: Output from generate_full_day().

        Returns:
            DataFrame with columns: hour, minute, total_trips, zone demands.
        """
        rows = []
        for entry in demand_data:
            od = entry["od_matrix"]
            row = {
                "timestep": entry["timestep"],
                "hour": entry["hour"],
                "minute": entry["minute"],
                "total_trips": int(od.sum()),
            }
            for z in range(NUM_ZONES):
                row[f"zone_{z}_origin"] = int(od[z, :].sum())
                row[f"zone_{z}_dest"] = int(od[:, z].sum())
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"Demand summary: {len(df)} rows, peak={df['total_trips'].max()} trips/min")
        return df

    def save_demand(self, demand_data, path="data/demand.npz"):
        """Save demand data to compressed numpy file.

        Args:
            demand_data: Output from generate_full_day().
            path: File path for the output.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        matrices = np.array([d["od_matrix"] for d in demand_data])
        np.savez_compressed(output, demand=matrices)
        print(f"Demand data saved to {output} ({matrices.shape})")


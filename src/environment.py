from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import deque

from .data_generator import NUM_ZONES, ZONE_LENGTH_KM, CORRIDOR_LENGTH_KM


BASE_SPEED_KMH = 60.0
BASE_SPEED_KPM = BASE_SPEED_KMH / 60.0


class Pod:

    def __init__(self, pod_id, position_km, capacity=6):
        self.pod_id = pod_id
        self.position_km = position_km
        self.capacity = capacity
        self.status = "idle"
        self.passengers = []
        self.destination_km = None
        self.destination_zone = None
        self.km_traveled = 0.0
        self.km_traveled_empty = 0.0
        self.trips_completed = 0
        self.boarding_timer = 0

    def is_idle(self):
        """Check if the pod is available for dispatch.

        Returns:
            True if the pod is idle.
        """
        return self.status == "idle"

    def get_zone(self):
        """Determine which zone the pod is currently in.

        Returns:
            Integer zone index (0 to NUM_ZONES-1).
        """
        zone = int(self.position_km / ZONE_LENGTH_KM)
        return min(zone, NUM_ZONES - 1)


class Passenger:

    def __init__(self, passenger_id, origin_zone, destination_zone, arrival_timestep):
        self.passenger_id = passenger_id
        self.origin_zone = origin_zone
        self.destination_zone = destination_zone
        self.arrival_timestep = arrival_timestep
        self.pickup_timestep = None
        self.dropoff_timestep = None

    def wait_time(self, current_timestep):
        """Calculate how long this passenger has been waiting.

        Args:
            current_timestep: Current simulation timestep.

        Returns:
            Wait time in minutes (timesteps).
        """
        if self.pickup_timestep is not None:
            return self.pickup_timestep - self.arrival_timestep
        return current_timestep - self.arrival_timestep


class SimulationEnvironment:

    def __init__(self, fleet_size: int = 500, pod_capacity: int = 6, seed: int = 42) -> None:
        # TODO: parallelize simulation runs
        self.fleet_size = fleet_size
        self.maxVehicles = fleet_size
        self.pod_capacity = pod_capacity
        self.rng = np.random.RandomState(seed)
        self.timestep = 0
        self.passenger_counter = 0

        self.pods = self._init_fleet()
        self.queues = {z: deque() for z in range(NUM_ZONES)}
        self.completed_passengers = []
        self.abandoned_passengers = []

        self._history = []
        print(f"Environment initialized: {fleet_size} pods, {pod_capacity} capacity")

    def _init_fleet(self) -> List["Pod"]:
        """Distribute pods evenly across the corridor.

        Returns:
            List of Pod objects positioned along The Line.
        """
        pods = []
        for i in range(self.fleet_size):
            position = (i / self.fleet_size) * CORRIDOR_LENGTH_KM
            pods.append(Pod(i, position, self.pod_capacity))
        print(f"  Fleet deployed: {len(pods)} pods across {CORRIDOR_LENGTH_KM}km")
        return pods

    def _zone_center(self, zone_id: int) -> float:
        """Get the center position of a zone in kilometers.

        Args:
            zone_id: Zone index.

        Returns:
            Position in km along the corridor.
        """
        return (zone_id + 0.5) * ZONE_LENGTH_KM

    def _travel_time_minutes(self, from_km: float, to_km: float) -> float:
        """Calculate travel time between two positions.

        Includes a congestion factor based on distance.

        Args:
            from_km: Origin position in km.
            to_km: Destination position in km.

        Returns:
            Travel time in minutes (float).
        """
        distance = abs(to_km - from_km)
        congestion = 1.0 + 0.1 * (distance / CORRIDOR_LENGTH_KM)
        return distance / (BASE_SPEED_KPM / congestion)

    def inject_passengers(self, od_matrix: np.ndarray) -> None:
        """Add new passengers to zone queues based on demand matrix.

        Args:
            od_matrix: NUM_ZONES x NUM_ZONES array of trip counts.
        """
        for orig in range(NUM_ZONES):
            for dest in range(NUM_ZONES):
                count = int(od_matrix[orig, dest])
                for _ in range(count):
                    p = Passenger(
                        self.passenger_counter, orig, dest, self.timestep
                    )
                    self.queues[orig].append(p)
                    self.passenger_counter += 1

    def _abandon_long_waiters(self, max_wait: int = 30) -> None:
        """Remove passengers who have waited too long.

        Args:
            max_wait: Maximum wait time in minutes before abandoning.
        """
        for zone_id in range(NUM_ZONES):
            remaining = deque()
            while self.queues[zone_id]:
                p = self.queues[zone_id].popleft()
                if self.timestep - p.arrival_timestep > max_wait:
                    self.abandoned_passengers.append(p)
                else:
                    remaining.append(p)
            self.queues[zone_id] = remaining

    def apply_dispatch(self, assignments: List[Tuple[int, int]]) -> None:
        """Apply dispatch decisions from an optimizer.

        Args:
            assignments: List of (pod_id, zone_id) tuples indicating
                which pod should go to which zone for pickup.
        """
        for pod_id, target_zone in assignments:
            pod = self.pods[pod_id]
            if not pod.is_idle():
                continue

            if not self.queues[target_zone]:
                continue

            passengers_to_board = []
            destination = None

            while self.queues[target_zone] and len(passengers_to_board) < pod.capacity:
                p = self.queues[target_zone][0]
                if destination is None:
                    destination = p.destination_zone
                if p.destination_zone == destination:
                    passengers_to_board.append(self.queues[target_zone].popleft())
                else:
                    break

            if not passengers_to_board:
                continue

            pod_zone = pod.get_zone()
            if pod_zone != target_zone:
                empty_dist = abs(self._zone_center(pod_zone) - self._zone_center(target_zone))
                pod.km_traveled_empty += empty_dist
                pod.km_traveled += empty_dist
                pod.position_km = self._zone_center(target_zone)

            for p in passengers_to_board:
                p.pickup_timestep = self.timestep

            pod.passengers = passengers_to_board
            pod.destination_zone = destination
            pod.destination_km = self._zone_center(destination)
            pod.status = "boarding"
            pod.boarding_timer = 2

    def _update_pods(self) -> None:
        """Advance pod movements by one timestep (1 minute)."""
        for pod in self.pods:
            if pod.status == "boarding":
                pod.boarding_timer -= 1
                if pod.boarding_timer <= 0:
                    pod.status = "en_route"

            elif pod.status == "en_route":
                if pod.destination_km is None:
                    pod.status = "idle"
                    continue

                direction = 1.0 if pod.destination_km > pod.position_km else -1.0
                move_dist = BASE_SPEED_KPM
                remaining = abs(pod.destination_km - pod.position_km)

                if move_dist >= remaining:
                    pod.km_traveled += remaining
                    pod.position_km = pod.destination_km

                    for p in pod.passengers:
                        p.dropoff_timestep = self.timestep
                        self.completed_passengers.append(p)

                    pod.trips_completed += 1
                    pod.passengers = []
                    pod.destination_km = None
                    pod.destination_zone = None
                    pod.status = "idle"
                else:
                    pod.position_km += direction * move_dist
                    pod.km_traveled += move_dist

    def step(self, od_matrix: Optional[np.ndarray] = None) -> None:
        """Advance the simulation by one timestep (1 minute).

        Args:
            od_matrix: Optional demand matrix to inject new passengers.
        """
        if od_matrix is not None:
            self.inject_passengers(od_matrix)

        self._update_pods()
        self._abandon_long_waiters()
        self.timestep += 1

    def get_state(self) -> Dict:
        """Capture the current simulation state for dispatchers.

        Returns:
            Dict with pod positions, statuses, queue lengths, and stats.
        """
        pod_states = []
        for pod in self.pods:
            pod_states.append({
                "pod_id": pod.pod_id,
                "position_km": pod.position_km,
                "zone": pod.get_zone(),
                "status": pod.status,
                "n_passengers": len(pod.passengers),
                "destination_zone": pod.destination_zone,
            })

        queue_lengths = {z: len(self.queues[z]) for z in range(NUM_ZONES)}

        wait_times = []
        for z in range(NUM_ZONES):
            for p in self.queues[z]:
                wait_times.append(p.wait_time(self.timestep))

        idle_pods_per_zone = np.zeros(NUM_ZONES)
        for pod in self.pods:
            if pod.is_idle():
                idle_pods_per_zone[pod.get_zone()] += 1

        return {
            "timestep": self.timestep,
            "hour": self.timestep // 60,
            "minute": self.timestep % 60,
            "pods": pod_states,
            "queue_lengths": queue_lengths,
            "wait_times": wait_times,
            "idle_pods_per_zone": idle_pods_per_zone,
            "total_waiting": sum(queue_lengths.values()),
            "total_completed": len(self.completed_passengers),
            "total_abandoned": len(self.abandoned_passengers),
        }

    def get_snapshot(self) -> Dict:
        """Create a lightweight snapshot for visualization.

        Returns:
            Dict with summary data suitable for plotting.
        """
        state = self.get_state()

        status_counts = {"idle": 0, "en_route": 0, "boarding": 0}
        for p in state["pods"]:
            status_counts[p["status"]] += 1

        zone_queues = [state["queue_lengths"].get(z, 0) for z in range(NUM_ZONES)]

        return {
            "timestep": state["timestep"],
            "hour": state["hour"],
            "minute": state["minute"],
            "pod_positions": [(p["position_km"], p["status"]) for p in state["pods"]],
            "status_counts": status_counts,
            "zone_queues": zone_queues,
            "avg_wait": np.mean(state["wait_times"]) if state["wait_times"] else 0.0,
            "total_waiting": state["total_waiting"],
            "total_completed": state["total_completed"],
            "total_abandoned": state["total_abandoned"],
        }

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.timestep = 0
        self.passenger_counter = 0
        self.pods = self._init_fleet()
        self.queues = {z: deque() for z in range(NUM_ZONES)}
        self.completed_passengers = []
        self.abandoned_passengers = []
        self._history = []
        print("Environment reset")


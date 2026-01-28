import numpy as np
from scipy.optimize import linprog

from .data_generator import NUM_ZONES


class GreedyDispatcher:

    def __init__(self):
        self.name = "Greedy"
        self.total_dispatches = 0

    def dispatch(self, env_state):
        """Generate pod-to-zone assignments using greedy nearest-first logic.

        Args:
            env_state: Environment state dict from SimulationEnvironment.get_state().

        Returns:
            List of (pod_id, zone_id) assignment tuples.
        """
        queue_lengths = env_state["queue_lengths"]
        pods = env_state["pods"]

        zones_with_demand = [
            (z, queue_lengths[z]) for z in range(NUM_ZONES) if queue_lengths[z] > 0
        ]
        zones_with_demand.sort(key=lambda x: x[1], reverse=True)

        idle_pods = [p for p in pods if p["status"] == "idle"]
        assigned_pods = set()
        assignments = []

        for zone_id, _ in zones_with_demand:
            if not idle_pods:
                break

            best_pod = None
            best_dist = float("inf")

            for pod in idle_pods:
                if pod["pod_id"] in assigned_pods:
                    continue
                dist = abs(pod["position_km"] - (zone_id + 0.5) * (170.0 / NUM_ZONES))
                if dist < best_dist:
                    best_dist = dist
                    best_pod = pod

            if best_pod is not None:
                assignments.append((best_pod["pod_id"], zone_id))
                assigned_pods.add(best_pod["pod_id"])
                self.total_dispatches += 1

        return assignments


class LPDispatcher:

    def __init__(self):
        self.name = "Linear Programming"
        self.total_dispatches = 0

    def dispatch(self, env_state):
        """Generate pod-to-zone assignments using linear programming.

        Minimizes total travel distance weighted by queue urgency.
        Uses a sparse formulation for efficiency.

        Args:
            env_state: Environment state dict from SimulationEnvironment.get_state().

        Returns:
            List of (pod_id, zone_id) assignment tuples.
        """
        queue_lengths = env_state["queue_lengths"]
        pods = env_state["pods"]
        wait_times = env_state.get("wait_times", [])

        zones_with_demand = [z for z in range(NUM_ZONES) if queue_lengths[z] > 0]
        idle_pods = [p for p in pods if p["status"] == "idle"]

        if not zones_with_demand or not idle_pods:
            return []

        n_pods = len(idle_pods)
        n_zones = len(zones_with_demand)
        n_vars = n_pods * n_zones

        cost_vector = np.zeros(n_vars)
        for i, pod in enumerate(idle_pods):
            for j, zone_id in enumerate(zones_with_demand):
                zone_center = (zone_id + 0.5) * (170.0 / NUM_ZONES)
                distance = abs(pod["position_km"] - zone_center)
                urgency = queue_lengths[zone_id]
                cost_vector[i * n_zones + j] = distance / max(urgency, 1)

        a_ub_rows = []
        b_ub_vals = []

        for i in range(n_pods):
            row = np.zeros(n_vars)
            for j in range(n_zones):
                row[i * n_zones + j] = 1.0
            a_ub_rows.append(row)
            b_ub_vals.append(1.0)

        for j, zone_id in enumerate(zones_with_demand):
            row = np.zeros(n_vars)
            for i in range(n_pods):
                row[i * n_zones + j] = 1.0
            a_ub_rows.append(row)
            needed = min(queue_lengths[zone_id], 3)
            b_ub_vals.append(float(needed))

        a_ub = np.array(a_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, 1)] * n_vars

        try:
            result = linprog(
                cost_vector, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                method="highs", options={"time_limit": 0.5}
            )
        except Exception:
            return []

        if not result.success:
            return []

        assignments = []
        x = result.x.reshape(n_pods, n_zones)

        for i in range(n_pods):
            best_j = np.argmax(x[i])
            if x[i, best_j] > 0.3:
                zone_id = zones_with_demand[best_j]
                assignments.append((idle_pods[i]["pod_id"], zone_id))
                self.total_dispatches += 1

        return assignments


class QLearningDispatcher:

    def __init__(self, n_occupancy_bins=5, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.05):
        self.name = "Q-Learning"
        self.n_bins = n_occupancy_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.total_dispatches = 0

        self.n_time_bins = 24
        self.n_zone_groups = 10
        self.n_actions = self.n_zone_groups * self.n_zone_groups

        state_size = self.n_zone_groups * self.n_bins * self.n_time_bins
        self.q_table = {}
        self.prev_state = None
        self.prev_action = None
        self.rng = np.random.RandomState(42)

    def _discretize_state(self, env_state):
        """Convert continuous state to discrete representation.

        Args:
            env_state: Environment state dict.

        Returns:
            Tuple representing discretized state for Q-table lookup.
        """
        queue_lengths = env_state["queue_lengths"]
        idle_counts = env_state["idle_pods_per_zone"]
        hour = env_state["hour"]

        group_size = NUM_ZONES // self.n_zone_groups
        group_demand = []
        group_supply = []

        for g in range(self.n_zone_groups):
            start = g * group_size
            end = start + group_size
            demand = sum(queue_lengths.get(z, 0) for z in range(start, end))
            supply = sum(idle_counts[start:end])
            demand_bin = min(int(demand / 5), self.n_bins - 1)
            supply_bin = min(int(supply / 10), self.n_bins - 1)
            group_demand.append(demand_bin)
            group_supply.append(supply_bin)

        return tuple(group_demand + group_supply + [hour])

    def _decode_action(self, action, env_state):
        """Convert action index to pod assignments.

        Args:
            action: Integer action index.
            env_state: Environment state dict.

        Returns:
            List of (pod_id, zone_id) assignment tuples.
        """
        group_size = NUM_ZONES // self.n_zone_groups
        source_group = action // self.n_zone_groups
        target_group = action % self.n_zone_groups

        if source_group == target_group:
            return self._greedy_fallback(env_state)

        source_start = source_group * group_size
        source_end = source_start + group_size
        target_start = target_group * group_size
        target_end = target_start + group_size

        pods = env_state["pods"]
        queue_lengths = env_state["queue_lengths"]

        idle_in_source = [
            p for p in pods
            if p["status"] == "idle" and source_start <= p["zone"] < source_end
        ]

        target_zones = [
            z for z in range(target_start, target_end) if queue_lengths.get(z, 0) > 0
        ]

        if not idle_in_source or not target_zones:
            return self._greedy_fallback(env_state)

        assignments = []
        n_rebalance = min(len(idle_in_source), len(target_zones), 5)

        target_zones.sort(key=lambda z: queue_lengths.get(z, 0), reverse=True)

        for i in range(n_rebalance):
            pod = idle_in_source[i]
            zone = target_zones[i % len(target_zones)]
            assignments.append((pod["pod_id"], zone))
            self.total_dispatches += 1

        extra = self._greedy_fallback(env_state, exclude_pods={a[0] for a in assignments})
        assignments.extend(extra)

        return assignments

    def _greedy_fallback(self, env_state, exclude_pods=None):
        """Fallback to greedy dispatch for unassigned pods.

        Args:
            env_state: Environment state dict.
            exclude_pods: Set of pod IDs already assigned.

        Returns:
            List of (pod_id, zone_id) assignment tuples.
        """
        exclude_pods = exclude_pods or set()
        queue_lengths = env_state["queue_lengths"]
        pods = env_state["pods"]

        zones_with_demand = [
            (z, queue_lengths[z]) for z in range(NUM_ZONES) if queue_lengths[z] > 0
        ]
        zones_with_demand.sort(key=lambda x: x[1], reverse=True)

        idle_pods = [
            p for p in pods
            if p["status"] == "idle" and p["pod_id"] not in exclude_pods
        ]

        assigned = set()
        assignments = []

        for zone_id, _ in zones_with_demand:
            best_pod = None
            best_dist = float("inf")

            for pod in idle_pods:
                if pod["pod_id"] in assigned:
                    continue
                dist = abs(pod["position_km"] - (zone_id + 0.5) * (170.0 / NUM_ZONES))
                if dist < best_dist:
                    best_dist = dist
                    best_pod = pod

            if best_pod is not None:
                assignments.append((best_pod["pod_id"], zone_id))
                assigned.add(best_pod["pod_id"])
                self.total_dispatches += 1

        return assignments

    def _compute_reward(self, env_state):
        """Compute reward signal from environment state.

        Args:
            env_state: Environment state dict.

        Returns:
            Float reward value (negative total wait time).
        """
        wait_times = env_state.get("wait_times", [])
        total_wait = sum(wait_times) if wait_times else 0
        abandoned = env_state.get("total_abandoned", 0)
        return -(total_wait + abandoned * 30)

    def dispatch(self, env_state):
        """Generate pod assignments using Q-learning with epsilon-greedy.

        Updates Q-table based on reward from previous action, then
        selects and executes a new action.

        Args:
            env_state: Environment state dict from SimulationEnvironment.get_state().

        Returns:
            List of (pod_id, zone_id) assignment tuples.
        """
        state = self._discretize_state(env_state)
        reward = self._compute_reward(env_state)

        if self.prev_state is not None and self.prev_action is not None:
            old_q = self.q_table.get((self.prev_state, self.prev_action), 0.0)
            best_future = max(
                [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)],
                default=0.0,
            )
            new_q = old_q + self.lr * (reward + self.gamma * best_future - old_q)
            self.q_table[(self.prev_state, self.prev_action)] = new_q

        if self.rng.random() < self.epsilon:
            action = self.rng.randint(self.n_actions)
        else:
            q_values = [
                self.q_table.get((state, a), 0.0) for a in range(self.n_actions)
            ]
            action = int(np.argmax(q_values))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.prev_state = state
        self.prev_action = action

        return self._decode_action(action, env_state)

    def get_stats(self):
        """Return current Q-learning statistics.

        Returns:
            Dict with epsilon, Q-table size, and total dispatches.
        """
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "total_dispatches": self.total_dispatches,
        }


def get_dispatcher(strategy_name):
    """Factory function to create a dispatcher by name.

    Args:
        strategy_name: One of 'greedy', 'lp', or 'qlearning'.

    Returns:
        Dispatcher instance.
    """
    dispatchers = {
        "greedy": GreedyDispatcher,
        "lp": LPDispatcher,
        "qlearning": QLearningDispatcher,
    }
    if strategy_name not in dispatchers:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Choose from {list(dispatchers.keys())}")
    dispatcher = dispatchers[strategy_name]()
    print(f"Dispatcher created: {dispatcher.name}")
    return dispatcher


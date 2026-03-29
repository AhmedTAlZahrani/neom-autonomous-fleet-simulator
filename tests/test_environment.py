import sys
import numpy as np
from collections import deque

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.environment import Pod, Passenger, SimulationEnvironment, BASE_SPEED_KPM
from src.fleet_optimizer import GreedyDispatcher, LPDispatcher, QLearningDispatcher, get_dispatcher
from src.data_generator import NUM_ZONES, ZONE_LENGTH_KM, CORRIDOR_LENGTH_KM


# --- Pod tests ---

def test_pod_initial_state():
    pod = Pod(0, 10.0, capacity=4)
    assert pod.pod_id == 0
    assert pod.position_km == 10.0
    assert pod.capacity == 4
    assert pod.status == "idle"
    assert pod.passengers == []
    assert pod.km_traveled == 0.0


def test_pod_is_idle():
    pod = Pod(1, 5.0)
    assert pod.is_idle() is True
    pod.status = "en_route"
    assert pod.is_idle() is False


def test_pod_get_zone():
    pod = Pod(0, 0.0)
    assert pod.get_zone() == 0

    pod2 = Pod(1, CORRIDOR_LENGTH_KM - 0.1)
    assert pod2.get_zone() == NUM_ZONES - 1

    mid_zone = 25
    pod3 = Pod(2, (mid_zone + 0.5) * ZONE_LENGTH_KM)
    assert pod3.get_zone() == mid_zone


def test_pod_get_zone_clamps_to_max():
    pod = Pod(0, CORRIDOR_LENGTH_KM + 100.0)
    assert pod.get_zone() == NUM_ZONES - 1


# --- Passenger tests ---

def test_passenger_wait_time_before_pickup():
    p = Passenger(0, 1, 5, arrival_timestep=10)
    assert p.wait_time(20) == 10


def test_passenger_wait_time_after_pickup():
    p = Passenger(0, 1, 5, arrival_timestep=10)
    p.pickup_timestep = 15
    assert p.wait_time(100) == 5


# --- SimulationEnvironment tests ---

def test_env_init():
    env = SimulationEnvironment(fleet_size=10, seed=0)
    assert len(env.pods) == 10
    assert env.timestep == 0
    assert len(env.queues) == NUM_ZONES
    assert all(len(q) == 0 for q in env.queues.values())


def test_env_fleet_positions_spread():
    env = SimulationEnvironment(fleet_size=10, seed=0)
    positions = [p.position_km for p in env.pods]
    assert positions[0] == 0.0
    assert positions[-1] > 0.0
    assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))


def test_env_step_increments_timestep():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    env.step()
    assert env.timestep == 1
    env.step()
    assert env.timestep == 2


def test_env_inject_passengers():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 1] = 3
    od[2, 4] = 2
    env.inject_passengers(od)
    assert len(env.queues[0]) == 3
    assert len(env.queues[2]) == 2
    assert env.passenger_counter == 5


def test_env_step_with_od_matrix():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 1] = 1
    env.step(od_matrix=od)
    assert env.timestep == 1
    assert env.passenger_counter == 1


def test_env_abandon_long_waiters():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 1] = 2
    env.inject_passengers(od)
    assert len(env.queues[0]) == 2

    # advance time past the max_wait threshold
    env.timestep = 35
    env._abandon_long_waiters(max_wait=30)
    assert len(env.queues[0]) == 0
    assert len(env.abandoned_passengers) == 2


def test_env_apply_dispatch_basic():
    env = SimulationEnvironment(fleet_size=10, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 3] = 2
    env.inject_passengers(od)

    # find a pod and assign it to zone 0
    pod = env.pods[0]
    env.apply_dispatch([(pod.pod_id, 0)])

    assert pod.status == "boarding"
    assert pod.boarding_timer == 2
    assert len(pod.passengers) > 0
    assert pod.destination_zone == 3


def test_env_apply_dispatch_skips_nonempty_pod():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 1] = 1
    env.inject_passengers(od)

    pod = env.pods[0]
    pod.status = "en_route"
    env.apply_dispatch([(pod.pod_id, 0)])
    # pod was not idle, so queue should still have passengers
    assert len(env.queues[0]) == 1


def test_env_apply_dispatch_skips_empty_queue():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    # no passengers injected
    pod = env.pods[0]
    env.apply_dispatch([(pod.pod_id, 0)])
    assert pod.status == "idle"


def test_env_boarding_then_enroute():
    env = SimulationEnvironment(fleet_size=10, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 1
    env.inject_passengers(od)
    env.apply_dispatch([(0, 0)])

    pod = env.pods[0]
    assert pod.status == "boarding"

    env.step()  # boarding_timer 2 -> 1
    assert pod.status == "boarding"

    env.step()  # boarding_timer 1 -> 0, switch to en_route
    assert pod.status == "en_route"


def test_env_pod_completes_trip():
    env = SimulationEnvironment(fleet_size=10, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    # short trip: zone 0 -> zone 1
    od[0, 1] = 1
    env.inject_passengers(od)
    env.apply_dispatch([(0, 0)])

    # run enough steps to finish boarding + travel
    for _ in range(100):
        env.step()
        if env.pods[0].status == "idle" and env.pods[0].trips_completed > 0:
            break

    assert env.pods[0].trips_completed >= 1
    assert len(env.completed_passengers) >= 1
    assert env.completed_passengers[0].dropoff_timestep is not None


def test_env_get_state_structure():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    state = env.get_state()
    assert "timestep" in state
    assert "pods" in state
    assert "queue_lengths" in state
    assert "idle_pods_per_zone" in state
    assert len(state["pods"]) == 5


def test_env_get_snapshot_structure():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    snap = env.get_snapshot()
    assert "pod_positions" in snap
    assert "status_counts" in snap
    assert "zone_queues" in snap
    assert snap["status_counts"]["idle"] == 5


def test_env_reset():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 1] = 3
    env.step(od_matrix=od)
    env.step()

    env.reset()
    assert env.timestep == 0
    assert env.passenger_counter == 0
    assert all(len(q) == 0 for q in env.queues.values())
    assert len(env.completed_passengers) == 0
    assert len(env.abandoned_passengers) == 0
    assert len(env.pods) == 5


def test_env_zone_center():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    center_0 = env._zone_center(0)
    assert center_0 == 0.5 * ZONE_LENGTH_KM
    center_last = env._zone_center(NUM_ZONES - 1)
    assert center_last == (NUM_ZONES - 0.5) * ZONE_LENGTH_KM


def test_env_travel_time_positive():
    env = SimulationEnvironment(fleet_size=5, seed=0)
    tt = env._travel_time_minutes(0.0, 50.0)
    assert tt > 0


# --- GreedyDispatcher tests ---

def test_greedy_dispatch_returns_assignments():
    env = SimulationEnvironment(fleet_size=20, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 3
    od[10, 20] = 2
    env.inject_passengers(od)

    dispatcher = GreedyDispatcher()
    state = env.get_state()
    assignments = dispatcher.dispatch(state)

    assert isinstance(assignments, list)
    assert len(assignments) > 0
    for pod_id, zone_id in assignments:
        assert isinstance(pod_id, (int, np.integer))
        assert 0 <= zone_id < NUM_ZONES


def test_greedy_dispatch_no_demand():
    dispatcher = GreedyDispatcher()
    env = SimulationEnvironment(fleet_size=5, seed=0)
    state = env.get_state()
    assignments = dispatcher.dispatch(state)
    assert assignments == []


def test_greedy_dispatch_increments_counter():
    env = SimulationEnvironment(fleet_size=20, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 1
    env.inject_passengers(od)

    dispatcher = GreedyDispatcher()
    assert dispatcher.total_dispatches == 0
    dispatcher.dispatch(env.get_state())
    assert dispatcher.total_dispatches > 0


# --- LPDispatcher tests ---

def test_lp_dispatch_returns_list():
    env = SimulationEnvironment(fleet_size=100, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 3
    od[10, 20] = 2
    env.inject_passengers(od)

    dispatcher = LPDispatcher()
    assignments = dispatcher.dispatch(env.get_state())
    assert isinstance(assignments, list)
    # LP may return empty if solver threshold filters assignments
    for pod_id, zone_id in assignments:
        assert isinstance(pod_id, (int, np.integer))
        assert 0 <= zone_id < NUM_ZONES


def test_lp_dispatch_no_demand():
    dispatcher = LPDispatcher()
    env = SimulationEnvironment(fleet_size=5, seed=0)
    assignments = dispatcher.dispatch(env.get_state())
    assert assignments == []


# --- QLearningDispatcher tests ---

def test_qlearning_dispatch_returns_assignments():
    env = SimulationEnvironment(fleet_size=20, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 3
    env.inject_passengers(od)

    dispatcher = QLearningDispatcher()
    assignments = dispatcher.dispatch(env.get_state())
    assert isinstance(assignments, list)


def test_qlearning_epsilon_decays():
    dispatcher = QLearningDispatcher(epsilon=1.0, epsilon_decay=0.9)
    env = SimulationEnvironment(fleet_size=20, seed=0)
    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 5] = 1
    env.inject_passengers(od)

    dispatcher.dispatch(env.get_state())
    assert dispatcher.epsilon < 1.0


def test_qlearning_get_stats():
    dispatcher = QLearningDispatcher()
    stats = dispatcher.get_stats()
    assert "epsilon" in stats
    assert "q_table_size" in stats
    assert "total_dispatches" in stats


# --- get_dispatcher factory ---

def test_get_dispatcher_greedy():
    d = get_dispatcher("greedy")
    assert d.name == "Greedy"


def test_get_dispatcher_lp():
    d = get_dispatcher("lp")
    assert d.name == "Linear Programming"


def test_get_dispatcher_qlearning():
    d = get_dispatcher("qlearning")
    assert d.name == "Q-Learning"


def test_get_dispatcher_unknown_raises():
    try:
        get_dispatcher("random_nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown strategy" in str(e)


# --- Integration: dispatch + apply + step ---

def test_greedy_end_to_end():
    env = SimulationEnvironment(fleet_size=50, seed=0)
    dispatcher = GreedyDispatcher()

    od = np.zeros((NUM_ZONES, NUM_ZONES))
    od[0, 3] = 5
    od[10, 15] = 3
    env.inject_passengers(od)

    state = env.get_state()
    assignments = dispatcher.dispatch(state)
    env.apply_dispatch(assignments)

    # run simulation for a while
    for _ in range(80):
        env.step()

    assert env.timestep == 80
    assert len(env.completed_passengers) > 0

# NEOM Autonomous Fleet Simulator

![Run Tests](https://github.com/AhmedTAlZahrani/neom-autonomous-fleet-simulator/actions/workflows/run-tests.yml/badge.svg)

Simulation platform for autonomous pod fleet management along The Line (170km corridor). Models trip demand across 50 zones, compares Greedy / LP / Q-Learning dispatch strategies with Streamlit visualization.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
```python
from src.data_generator import DemandGenerator
from src.environment import SimulationEnvironment
from src.fleet_optimizer import get_dispatcher
from src.metrics import SimulationMetrics

gen = DemandGenerator()
gen.add_random_events(5)
demand = gen.generate_full_day()
env = SimulationEnvironment(fleet_size=500)
dispatcher = get_dispatcher("qlearning")
metrics = SimulationMetrics()
for t in range(len(demand)):
    env.inject_passengers(demand[t]["od_matrix"])
    env.apply_dispatch(dispatcher.dispatch(env.get_state()))
    env.step()
    metrics.record_snapshot(env)
metrics.finalize(env)
```

## Project Structure

```
src/
    data_generator.py     Trip demand generation across 50 zones
    environment.py        Simulation state and pod fleet management
    fleet_optimizer.py    Greedy / LP / Q-Learning dispatchers
    metrics.py            Wait time, utilization, throughput tracking
    visualization.py      Plotly charts for simulation results
app.py                    Streamlit dashboard
```

## License

Apache License 2.0

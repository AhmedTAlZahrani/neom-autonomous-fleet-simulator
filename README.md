# NEOM Autonomous Fleet Simulator

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

"""Mininet / Open vSwitch / os-ken emulation backend for the SDN-DRL project.

This package is an *additive* extension. It never modifies the simulator
(`environments/`, `experiments/`, `utils/`, `forecasting/`), never writes to
`models/`, `results/` or `experiments/results/`, and stores all of its output
under `mininet_results/`.

Data flow:
    same traffic trace -> Mininet/OVS/os-ken -> same trained RDQN -> emulated metrics
"""

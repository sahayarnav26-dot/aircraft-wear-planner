# NASA C-MAPSS FD001

`train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` are the FD001 subset of NASA's
Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan degradation data.

- Source: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation", International Conference on Prognostics and Health Management, 2008.

FD001: 100 training engines run to failure, 100 test engines cut off before failure, one operating
condition, one fault mode (HPC degradation). Each row is one cycle: unit, cycle, 3 operating settings
and 21 sensor readings. This is simulated data, not airline data.

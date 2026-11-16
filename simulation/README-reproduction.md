# Reproducing the preliminary controller results

Run `node simulate.js` with Node.js 20 or newer in this folder. The script uses no external packages and writes `simulation_runs.csv`, `simulation_summary.csv`, and `simulation_paired_effects.csv` deterministically. The manuscript's simulation table was taken from these files.

The edge/server times, activation sizes, RTT, switching charge, network traces, and observation noise in the script are declared assumptions. The outputs are genuine calculations from this model; they are not measurements of a DNN, CIFAR-10, or physical edge/cloud hardware. The measured experiment described in `main.tex` remains future work.


# epigenome-layer-model
Simulation code for the illustrative scalar bistable-feedback model in "The Epigenome as a Distributed Replication System: A Five-Layer Framework for Aging Interventions" (K. Hamdy). 

The script `fig1_extended.py` generates all four panels of Figure 1 and prints the band-edge values reported in Section 2.1.

## Model
A one-dimensional positive-feedback model of fidelity-loss burden D: dD/dt = u * (1 + h * D^n / (K^n + D^n)) - gamma * A * D, A = a1 * a2 Each intervention raises a capacity term by a fixed dose; effects are computed by relaxing from the aged (high-damage) steady state. Three arms are simulated: cross-layer, same-layer, and shared-pool. See the manuscript Section 2.1 for the full description. The model is illustrative, not fitted: parameters were chosen to demonstrate that the predicted banded advantage is possible.

## Parameters 
| Symbol | Value | Meaning |
|--------|-------|---------------------------------|
| u | 0.02 | basal fidelity-loss rate |
| h | 8 | positive-feedback amplitude | 
| K | 0.3 | feedback half-saturation | 
| n | 6 | Hill coefficient | 
| gamma | 1 | clearance rate | 
| delta | 0.12 | per-intervention capacity dose | 
| kappa | 1 | shared-pool saturation constant | 

## Requirements
- Python 3.11
- See `requirements.txt` (numpy, scipy, matplotlib) Install: pip install -r requirements.txt

## Run
python fig1_extended.py
Outputs `Figure1_extended.png` and `Figure1_extended.pdf`, and prints the bistable band and per-arm band edges to stdout. 

## Expected output
- Bistable band: gamma*A = 0.147 to 0.391
- Cross-layer peak excess over Bliss: +0.560
- Sub-Bliss minimum: -0.081

## License
MIT (see `LICENSE`). 

## Citation
If you use this code, please cite the Zenodo archive (DOI to be added on deposition) and the associated manuscript.

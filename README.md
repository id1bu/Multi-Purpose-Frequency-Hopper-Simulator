
# Frequency Hopping Simulator (Lightweight)

A small Python simulator to model frequency-hopping behavior over a defined band.

## Features
- Configurable number of channels and time slots
- Pseudo-random (PRN) or sequential modes
- Optional adaptive avoidance list
- Outputs CSV logs and visualizations

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Default run (200 slots, 24 channels):
```bash
python fh_simulator.py
```

Custom parameters example:
```bash
python fh_simulator.py --num_time_slots 500 --num_channels 32 --mode prn --avoid_list 1100 1500
```

## Output Files
- `output/hops.csv` - detailed hop log
- `output/fh_simulator.png` - frequency vs time plot
- `output/fh_heatmap.png` - occupancy heatmap
- `output/hops_summary.csv` - hits per channel

## Future Improvements
- Add modulation / SNR simulation
- Simulate multiple transmitters and collisions
- Implement realistic PRNG (m-sequences or Gold codes)
- Add animation and interactive UI (Streamlit/Plotly)

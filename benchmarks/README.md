# Benchmarks

Parameter sensitivity analysis and performance benchmarking for strainphase.

## Files

- `parameter_sweep.py` - Tests pipeline stability across parameter grids
- `benchmark_performance.py` - Measures runtime and memory usage

## Running Benchmarks

```bash
# Run parameter sweep
python benchmarks/parameter_sweep.py

# Run performance benchmarks
python benchmarks/benchmark_performance.py --output results/
python benchmarks/benchmark_performance.py --quick  # Fast benchmarks
```

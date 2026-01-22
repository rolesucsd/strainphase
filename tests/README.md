# Tests

Unit and integration tests for strainphase.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test
pytest tests/test_core.py::TestHaplotyperConfig -v
```

## Test Structure

- `test_core.py` - Unit and integration tests for the core haplotyper pipeline

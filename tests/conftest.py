"""
Shared pytest fixtures for the QuMask test suite.

Fixture scopes
--------------
- session: computed once per `pytest` invocation. Used for expensive objects
  (config, generated instances, datasets) that no test should mutate.
- function: default scope; used for mutable objects like RNG state.

Slow fixtures (marked with `pytest.mark.slow`) invoke Qiskit or
`generate_dataset` and are only run with the full suite.
Deselect fast-only: pytest -m "not slow"

All random seeds are fixed so the suite is fully reproducible.
"""

import numpy as np
import pytest
import yaml


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def k() -> int:
    return 3


@pytest.fixture(scope="session")
def n() -> int:
    return 10


@pytest.fixture(scope="session")
def output_dim(k) -> int:
    return 2 ** k


@pytest.fixture(scope="session")
def F(n, k) -> int:
    from data.features import feature_dim
    return feature_dim(n, k)


@pytest.fixture(scope="session")
def uniform_dist(output_dim) -> np.ndarray:
    return np.ones(output_dim, dtype=np.float64) / output_dim


@pytest.fixture(scope="session")
def peaked_dist(output_dim) -> np.ndarray:
    p = np.ones(output_dim, dtype=np.float64) * (0.03 / (output_dim - 1))
    p[0] = 0.97
    return p


@pytest.fixture(params=["uniform", "peaked"])
def any_dist(request, uniform_dist, peaked_dist) -> np.ndarray:
    return {"uniform": uniform_dist, "peaked": peaked_dist}[request.param]


@pytest.fixture(scope="session")
def synthetic_bitstrings(n, k) -> dict:
    """Blocked bitstring array with a known exact marginal — no Qiskit required.

    bitstrings: uint8 (n_blocks=20, shots=10, n=10).
    Target columns [0,1,2] are forced to all-zeros, so the marginal over those
    qubits is exactly [1, 0, 0, 0, 0, 0, 0, 0]. Remaining columns are random.
    Used to give empirical_marginal a ground-truth answer to check against.
    """
    rng_s = np.random.default_rng(0)
    n_blocks, shots = 20, 10
    bits = rng_s.integers(0, 2, size=(n_blocks, shots, n), dtype=np.uint8)
    bits[:, :, :k] = 0
    target_positions = np.tile(np.arange(k, dtype=int), (n_blocks, 1))
    known_marginal = np.zeros(2 ** k, dtype=np.float64)
    known_marginal[0] = 1.0
    return {"bitstrings": bits, "target_positions": target_positions, "known_marginal": known_marginal}


@pytest.fixture(scope="session")
def tiny_instance(k, n) -> dict:
    """One real simulated instance: k=3, n=10, n_blocks=10, shots=10, seed=0.

    Keys: features (10,175) float32, p_star (8,) float64,
          bitstrings (10,10,10) uint8, target_positions (10,3) int.
    """
    from data.simulate import generate_instance
    return generate_instance(k=k, n=n, n_blocks=10, shots_per_block=10, target_depth=4, seed=0)


@pytest.fixture(scope="session", params=[0, 1, 2])
def tiny_instance_varied(k, n, request) -> dict:
    """Three tiny instances (seeds 0, 1, 2) — tests using this run once per seed."""
    from data.simulate import generate_instance
    return generate_instance(k=k, n=n, n_blocks=10, shots_per_block=10, target_depth=4, seed=request.param)


@pytest.fixture(scope="session")
def tiny_dataset_path(tmp_path_factory, k, n) -> str:
    """20-instance dataset saved to a temp .npz. Generated once, reused by all callers."""
    from data.simulate import generate_dataset, save_dataset
    path = str(tmp_path_factory.mktemp("data") / "tiny_dataset.npz")
    dataset = generate_dataset(
        n_instances=20, k=k, n=n, n_blocks=10, shots_per_block=10,
        target_depth=4, seed_base=100, n_jobs=1,
    )
    save_dataset(dataset, path)
    return path

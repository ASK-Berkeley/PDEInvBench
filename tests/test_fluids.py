"""
Tests for fluid dynamics residual computations.
Compares torch implementations against numpy reference implementations.

Can be run as:
1. Pytest test: pytest tests/test_fluids.py (skips if data not found)
2. Standalone script: python tests/test_fluids.py --filename <path_to_data>
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from loguru import logger

from pdeinvbench.losses.fluids import (
    compute_advection,
    compute_stream_function,
    laplacian,
    turbulent_flow_residual,
)


def find_turbulent_flow_data():
    """Try to find turbulent flow data in common locations."""
    possible_paths = [
        Path("../data/2D_turbulent-flow_nu=0.006153085601625313.h5"),
        Path("data/2D_turbulent-flow_nu=0.006153085601625313.h5"),
        Path("/data/shared/meta-pde/turbulent-flow-2d/train").glob("*.h5"),
    ]

    for path in possible_paths:
        if isinstance(path, Path) and path.exists():
            return str(path)
        # Handle glob results
        try:
            for file in path:
                if file.exists():
                    return str(file)
        except (TypeError, AttributeError):
            pass

    return None


@pytest.fixture
def turbulent_flow_datafile():
    """Fixture that provides path to test data."""
    data_path = find_turbulent_flow_data()
    if data_path is None:
        pytest.skip(
            "Turbulent flow test data not found. "
            "This test requires real PDE data and is skipped in CI/CD. "
            "Run manually with: python tests/test_fluids.py --filename <path>"
        )
    return data_path


def wrapper(func):
    """Convert torch tensor outputs to numpy for comparison."""

    def _wrapper(*args):
        # Convert numpy to torch
        new_args = [
            torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a
            for a in args
        ]
        out = func(*new_args)
        return out.cpu().numpy()

    return _wrapper


def _maybe_unsqueeze_np(u):
    """Ensure last dimension exists for channel."""
    return u if u.shape[-1] == 1 else np.expand_dims(u, axis=-1)


def compare_funcs(f1, f2):
    """Compare outputs of two functions and log the difference."""

    def compare(f1args, f2args):
        reference = _maybe_unsqueeze_np(f1(*f1args))
        newout = wrapper(f2)(*f2args)
        diff = np.linalg.norm(reference - newout)
        logger.info(f"Diff between {f1.__name__} and {f2.__name__}: {diff:.2e}")

        # Assert reasonable accuracy
        assert diff < 1e-3, f"Difference too large: {diff:.2e}"
        return diff

    return compare


def run_fluids_comparison(filename: str):
    """
    Run the fluids residual computation comparison.

    Args:
        filename: Path to turbulent flow HDF5 data file
    """
    # Import numpy reference implementations
    from fluids_numpy_reference import (
        advection as advection_np_base,
        compute_stream_function as compute_stream_function_np,
        laplacian as laplacian_np,
        tf_residual_numpy,
    )

    logger.info(f"Loading data from: {filename}")

    # Load data
    try:
        dataset = h5py.File(filename, "r")
        traj_idx = "0000"
        data = dataset[traj_idx]["data"][:]
        logger.info(f"Data shape: {data.shape}")
        t = dataset[traj_idx]["grid/t"][:]
        x = dataset[traj_idx]["grid/x"][:]
        y = dataset[traj_idx]["grid/y"][:]
        dataset.close()
    except (KeyError, FileNotFoundError) as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Extract parameters
    nu = (
        os.path.basename(filename)
        .split("=")[-1]
        .replace(".h5", "")
        .replace(".hdf5", "")
    )
    nu: float = float(nu)
    logger.info(f"Viscosity parameter nu: {nu}")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute residual norm as sanity check
    residual_norm = np.linalg.norm(wrapper(turbulent_flow_residual)(data, t, x, y, nu))
    logger.info(f"Computed residual norm: {residual_norm:.2e}")

    # Test 1: Stream function computation (Fourier space)
    logger.info("\n=== Testing stream function (Fourier=False) ===")
    compute_stream_function_np.__name__ = "compute_stream_function_np, fourier=False"
    compare_funcs(compute_stream_function_np, compute_stream_function)(
        (data, x, y, False), (data, dx, dy, False)
    )

    # Test 2: Stream function computation (physical space)
    logger.info("\n=== Testing stream function (Fourier=True) ===")
    compute_stream_function_np.__name__ = "compute_stream_function_np, fourier=True"
    compare_funcs(compute_stream_function_np, compute_stream_function)(
        (data, x, y, True), (data, dx, dy, True)
    )

    # Test 3: Advection term
    logger.info("\n=== Testing advection ===")

    def advection_np(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[0]

    advection_np.__name__ = "advection_np"
    compare_funcs(advection_np, compute_advection)((data, x, y), (data, dx, dy))

    # Test 4: Laplacian
    logger.info("\n=== Testing laplacian ===")
    laplacian_np.__name__ = "laplacian_np"
    compare_funcs(laplacian_np, laplacian)((data, x, y), (data, dx, dy))

    # Test 5: Velocity component vx
    logger.info("\n=== Testing velocity vx ===")

    def advection_np_vx(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[1]

    advection_np_vx.__name__ = "advection_np for vx"
    compare_funcs(
        advection_np_vx, lambda *args: compute_advection(*args, return_velocity=True)[1]
    )((data, x, y), (data, dx, dy))

    # Test 6: Velocity component vy
    logger.info("\n=== Testing velocity vy ===")

    def advection_np_vy(u, x, y):
        return advection_np_base(u, x, y, stream_func=compute_stream_function_np)[2]

    advection_np_vy.__name__ = "advection_np for vy"
    compare_funcs(
        advection_np_vy, lambda *args: compute_advection(*args, return_velocity=True)[2]
    )((data, x, y), (data, dx, dy))

    # Test 7: Full turbulent flow residual
    logger.info("\n=== Testing full turbulent flow residual ===")
    compare_funcs(tf_residual_numpy, turbulent_flow_residual)(
        (data, t, x, y, nu), (data, t, x, y, nu)
    )

    logger.success("\n✓ All fluids tests passed!")


@pytest.mark.slow
@pytest.mark.data_required
def test_fluids_residual(turbulent_flow_datafile):
    """
    Pytest wrapper for fluids residual comparison.

    Automatically skipped if test data is not available.
    Run with: pytest tests/test_fluids.py -v -m data_required
    """
    run_fluids_comparison(turbulent_flow_datafile)


if __name__ == "__main__":
    # Allow running as standalone script with argparse
    parser = argparse.ArgumentParser(
        description="Validate torch turbulent flow residual computations against numpy reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_fluids.py --filename data/2D_turbulent-flow_nu=0.006153085601625313.h5
  python tests/test_fluids.py --auto-find
        """,
    )
    parser.add_argument(
        "--filename", type=str, help="Path to turbulent flow HDF5 data file"
    )
    parser.add_argument(
        "--auto-find",
        action="store_true",
        help="Automatically search for data file in common locations",
    )

    args = parser.parse_args()

    if args.auto_find or args.filename is None:
        logger.info("Searching for turbulent flow data...")
        filename = find_turbulent_flow_data()
        if filename is None:
            logger.error(
                "Could not find turbulent flow data. "
                "Please specify path with --filename"
            )
            sys.exit(1)
        logger.info(f"Found data at: {filename}")
    else:
        filename = args.filename

    if not Path(filename).exists():
        logger.error(f"File not found: {filename}")
        sys.exit(1)

    try:
        run_fluids_comparison(filename)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

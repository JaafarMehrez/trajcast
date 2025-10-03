## Project Overview

TrajCast is a framework for generating molecular dynamics (MD) trajectories using autoregressive equivariant message-passing neural networks (MPNNs). It predicts atomic displacements and velocities, rolling out trajectories without explicit force calculations.

## Architecture & Key Components

1

- **Core package:** `trajcast/` contains main modules:
  - `model/`: Model definitions, training, forecasting, losses, checkpointing, and utilities.
  - `nn/`: Neural network layers, encoding, message passing, non-linearity, normalization, tensor interactions.
  - `data/`: Atomic graph, dataset, trajectory handling, wrappers for LAMMPS and other formats.
  - `cli/`: Command-line interface tools.
  - `utils/`: Atomic computations and miscellaneous utilities.
  - `validation/`: Physical behavior and trajectory comparison tools.
- **Tests:** Located in `tests/unit/` and mirror the main package structure. Use realistic data files for validation.
- **Examples:** Jupyter notebooks for training (`examples/training/training.ipynb`) and inference (`examples/inference/forecasting.ipynb`).

## Developer Workflows

- **Installation:**
  - Use `pip install .` for basic setup.
  - For CUDA, install with `pip install .[cueq]` after installing compatible PyTorch and dependencies.
  - For examples, use `pip install .[examples]`.
- **Training:**
  - Start with `examples/training/training.ipynb`.
- **Inference:**
  - Use `examples/inference/forecasting.ipynb`.
- **Testing:**
  - Tests are in `tests/unit/`. Run with `pytest` from the repo root.
  - Test data is in `tests/unit/data/` and subfolders.
- **Data:**
  - Example datasets and model weights are hosted on HuggingFace (see README for links).

## Project-Specific Patterns & Conventions

- **Autoregressive rollout:** Model outputs (positions, velocities) are fed as inputs for the next step.
- **Momentum conservation:** Displacements and velocities are post-processed to enforce physical constraints.
- **Message passing:** Neural network blocks use radial basis and spherical harmonics for edge features.
- **Config-driven:** Training and forecasting use YAML config files (see `tests/unit/model/data/`).
- **Modular design:** Each submodule is self-contained and tested independently.
- **Notebooks:** Use provided notebooks for reproducible experiments and workflow demos.

## Integration Points

- **External dependencies:**
  - PyTorch (2.5.1), torch_scatter, torch_cluster
  - Optional: cuEquivariance for CUDA acceleration
- **Data formats:**
  - Supports XYZ, EXTXYZ, NPZ, and LAMMPS formats
  - Wrappers in `trajcast/data/wrappers/`

## Example Commands

```sh
# Install dependencies (CPU)
pip install torch==2.5.1
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install .

# Run tests
pytest
```

## References

- [README.md](../README.md) for architecture, installation, and usage
- [examples/](../examples/) for training/inference workflows
- [tests/unit/](../tests/unit/) for test patterns and data
- [trajcast/model/](../trajcast/model/) for model logic and conventions

---

_Update this file as project conventions evolve. For questions, see README or contact maintainers._

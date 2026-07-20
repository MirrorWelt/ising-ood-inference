# Data directory

This directory is intentionally left mostly empty in version control
(see `.gitignore`). Large artefacts are regenerated locally or fetched from
external storage:

- `topologies/*.xml` — adjacency-matrix ensembles, generated with
  `tools/generate_topology_xml.py`.
- `lattice_datasets_original/`, `lattice_datasets_33links/` — Glauber
  trajectory `.pt` files, generated with `tools/generate_trajectory_dataset.py`.

All paths are configurable via environment variables documented in
`src/ising_ood/config.py` (e.g. `ISING_OOD_DATA_ROOT`).
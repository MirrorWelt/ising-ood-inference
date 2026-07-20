# Out-of-distribution Neural Inference in Dynamical Ising Models

Reconstructing interaction topology from Glauber magnetization trajectories, and diagnosing what neural networks actually learn under distribution shift

[Quick start](#quick-start) ·
[Paper guide](#paper-guide) ·
[Code guide](#code-guide) ·
[Reference tools](#reference-tools-and-data) ·
[Reproducibility](#reproducibility-contract) ·
[Citation](#citation)

Out-of-distribution (OOD) neural inference asks whether a network trained to solve a physical inverse problem has learned a **transferable dynamics-to-structure rule**, or merely an **architecture-dependent statistical shortcut** that happens to generalize under a fixed evaluation metric. This repository studies that question in a fully controlled setting: reconstructing the interaction topology of a kinetic Ising model from Glauber-dynamics magnetization trajectories.

**Current release:** a modular, tested re-packaging of the original research-code path used to produce the manuscript's figures. The public core retains the exact Glauber-dynamics integrator, the five network architectures (CNN-2, CNN-3, GNN, Transformer, Hybrid), and the edge-population diagnostic. Data generation, training, and evaluation are exposed as thin, configuration-driven adapters around this core.

## Quick start

```bash
git clone https://github.com/<org>/ising-ood.git
cd ising-ood

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

```bash
python -m pip install -e ".[dev]"
pytest              # physics and forward-pass shape tests
ruff check .        # code-quality check
```

A CPU-friendly end-to-end smoke test (tiny lattice, few epochs) is included in `tests/test_end_to_end_quickstart.py` and runs in a few seconds; it exercises topology generation, trajectory generation, model construction, and one training epoch through the same code paths used for the paper-scale runs.

```bash
ising-ood generate-topology --config configs/topology_original.yaml
python tools/generate_trajectory_dataset.py \
    --matrices-xml data/topologies/matrices_original.xml \
    --out-dir data/lattice_datasets_original \
    --start-matrix 1 --end-matrix 50 --samples-per-matrix 700
ising-ood train --config configs/main_nc25.yaml
ising-ood evaluate --config configs/eval_topology_ood.yaml
```

## The idea in one picture

The kinetic Ising model has interaction topology encoded by an adjacency matrix $A$,

$$
H = -J\sum_{i,j} A_{ij}\,s_i s_j ,
$$

with local magnetization $m_i(t)=\langle s_i(t)\rangle$ evolving under the mean-field Glauber equation

$$
\frac{dm_i}{dt} = -m_i + \tanh\!\Big(\beta J \sum_j A_{ij} m_j\Big).
$$

The **forward problem** integrates this equation from a random initial condition to obtain a magnetization trajectory. The **inverse problem** studied here is: given only the trajectory, reconstruct the upper-triangular part of $A$ (66 candidate undirected links for $L=12$ spins). Five architectures — encoding convolutional locality, message-passing relational structure, attention-based long-range dependence, and hybrid local–global processing — are trained as 66-dimensional multi-label regressors and evaluated under three regimes:

| Resource | Meaning | Config field |
|---|---|---|
| $N_L$ | number of distinct training lattice topologies | `num_lattices` |
| $N_c$ | true number of links per lattice (25 imbalanced main setting, 33 balanced control) | `num_links` |
| $T=1/\beta$ | dynamical temperature at which trajectories are generated | `beta` |

At fixed $N_L$, $N_c$, and $T$, a single trained model is evaluated on three disjoint test sets: the **ID test set** (same topology ensemble and temperature, independent initial conditions), the **topology-shift OOD set** (unseen lattices, same temperature), and the **temperature-shift OOD set** (training lattices, unseen temperatures). This separation is the central methodological device of the paper: it prevents in-distribution accuracy from being mistaken for transferable physical rule learning.

## Paper guide

### What is new?

Prior work on Ising-model structure recovery typically reports accuracy on data drawn from the same distribution as training. This repository instead asks a sharper question: **when a model's prediction accuracy degrades or stays stable under distribution shift, what mechanism is responsible?** The answer is obtained not from accuracy alone but from an **edge-population diagnostic** — the average predicted number of links, $\hat N_c$ — which separates models that preserve the training graph density from models that collapse toward a majority no-link prediction.

The scientific criterion running through every experiment is:

1. does in-distribution accuracy reflect trajectory-level interpolation, and how does it scale with $N_L$;
2. does that accuracy survive a topology shift or a temperature shift; and
3. is the surviving accuracy explained by correct link localization, or by exploiting the class prior of an imbalanced task.

### Main results at a glance

| Manuscript result | Representative finding | Physical conclusion |
|---|---|---|
| ID accuracy vs $N_L$ (Fig. 2a–b) | Ranking Transformer > Hybrid > GNN > CNN-2 > CNN-3; accuracy decreases as $N_L$ grows under a fixed sample budget | Transformer-based models extract the most predictive information within the training ensemble |
| Topology-shift OOD accuracy vs $N_L$ (Fig. 2c) | Ranking reverses (CNN-2/CNN-3 outperform Transformer); accuracy *increases* with $N_L$, opposite to the ID trend, reaching only ~61% at $N_L=50$ against a 62.1% no-link baseline | Inductive biases favoring ID interpolation do not favor topology-level extrapolation |
| Temperature-shift OOD accuracy vs $T$ (Fig. 3a) | Near $T=1$ all models retain high accuracy; low-$T$ shift ($T=0.05$–$0.2$) degrades moderately; high-$T$ shift is far more disruptive, with a ranking crossover near $T\simeq5$–10 and accuracies falling *below* the no-link baseline | Raising $T$ suppresses magnetic ordering and destroys topology-dependent dynamical signatures |
| Predicted link count $\hat N_c$ vs $N_L$ and $T$ (Fig. 3b–c) | Transformer preserves $\hat N_c\approx21$–25 across conditions; CNN-3 collapses to $\hat N_c\approx3$ | Architectures implement distinct statistical strategies — density-preserving vs. no-link-majority collapse — under the same physical shift |
| Balanced-link control, $N_c=33$ (Fig. S4) | Removing the majority-class advantage (no-link baseline drops to 50.0%) still shows architecture-dependent degradation and density behavior | The sparse-link collapse of CNN-3 reflects an interplay between architectural bias and class prior, not architecture alone |

These are manuscript-level results, reproduced by the evaluation modules described below; the CPU quickstart exercises the same code paths at a much smaller scale and is not expected to reproduce these numbers.

### Questions raised during peer review

**Does high OOD accuracy imply the model learned Glauber dynamics?** Not necessarily. The predicted-link-count diagnostic $\hat N_c$ shows that a model such as CNN-3 can maintain near-flat OOD accuracy purely by predicting very few links, exploiting the imbalance of the $N_c=25$ task (a no-link predictor already scores 62.1%). Apparent robustness must be checked against what graph the model actually outputs, not against accuracy alone.

**Is the topology-shift result simply an artifact of task difficulty increasing with $N_L$?** No — the ID and topology-OOD trends move in *opposite* directions as $N_L$ increases. If both were purely difficulty effects, they would move together. The opposite dependence indicates that the two settings probe different capabilities: trajectory-level interpolation versus topology-level extrapolation.

**Are low- and high-temperature shifts symmetric?** No. Lowering $T$ enhances magnetic ordering; topology-dependent correlations remain partly visible, and degradation is moderate. Raising $T$ suppresses ordering and directly erodes the dynamical signature of the interaction topology, which is a stronger and more physically grounded failure mode than a generic input-distribution shift.

**Is $N_c=25$ representative, or does the majority no-link class drive every conclusion?** The Supplemental Material's balanced control at $N_c=33$ (no-link baseline exactly 50.0%) is included specifically to test this. It confirms that the sparse-link behavior of CNN-3 is shaped by the *interplay* between architecture and class prior, not by class imbalance alone.

**Should this be read as "Transformers are simply better"?** No. The Transformer's density-preserving strategy yields the best ID accuracy but becomes fragile under shift, since many preserved links are placed incorrectly. No single architecture in this study demonstrates unambiguous transferable dynamics-to-structure inference; each implements a different data-driven prior.

### Training objective

All models are trained with the Adam optimizer and binary cross-entropy loss over the 66 candidate link labels,

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log(\hat y_i) + (1-y_i)\log(1-\hat y_i)\Big].
$$

## Code guide

### Configurable run

```bash
ising-ood train --config configs/main_nc25.yaml
```

```yaml
# configs/main_nc25.yaml — reproduces the paper's main N_c = 25 setting
dataset_mode: nodes
model_type: transformer
it_time: 1000
num_lattices: 50
total_samples: 35000
num_epochs: 80
batch_size: 32
lr: 0.001
weight_decay: 0.00001
data_dir: ./data/lattice_datasets_original
```

Every experiment parameter — model type, number of training lattices $N_L$, temperature $\beta$, link count $N_c$, epochs, and dataset paths — lives in a YAML config rather than being hard-coded, so a new experiment is a new config file, not a source-code edit.

### Generate a topology ensemble and trajectory dataset

```bash
python tools/generate_topology_xml.py --mode fixed_edges --num-edges 33 \
    --num-matrices 500 --out data/topologies/matrices_33links.xml

python tools/generate_trajectory_dataset.py \
    --matrices-xml data/topologies/matrices_33links.xml \
    --out-dir data/lattice_datasets_33links \
    --start-matrix 1 --end-matrix 50 --samples-per-matrix 700
```

`--mode fixed_edges` places an exact number of undirected links and verifies connectivity by depth-first search (used for the $N_c=33$ balanced control and the 33-links main setting); `--mode original` reproduces the legacy random upper-triangular construction with random-walk connectivity checking used for the $N_c=25$ imbalanced main setting.

### Architecture

```
src/ising_ood/
├── dynamics/
│   ├── glauber.py         Eq. (2) integrator: coupling_term, generate_tensor_coupling
│   └── topology.py        adjacency-matrix generation, connectivity checks, XML I/O
├── data/
│   ├── dataset.py         SingleMatrixDataset, MultiClassCouplingDataset
│   └── discovery.py       filename parsing for datasets and checkpoints
├── models/
│   ├── cnn.py              CNN-2 (CNNModel), CNN-3 (ImprovedCNNModel)
│   ├── gnn.py               GNNModel, SimpleGCNLayer, GlobalAvgPooling
│   ├── transformer.py       TransformerModel, GlobalAttentionPooling
│   ├── hybrid.py            HybridCNNTransformer
│   └── factory.py           ModelFactory, output_to_tensor / inverse_tensor
├── training/
│   ├── engine.py            single-stage train/evaluate loop
│   └── staged.py            two-stage (original -> 33-links) transfer training
├── evaluation/
│   ├── id_ood_topology.py         Fig. 2(a)-(c): ID and topology-shift OOD accuracy vs N_L
│   ├── temperature_shift.py        Fig. 3(a): temperature-shift OOD accuracy vs T
│   └── link_diagnostics.py         Fig. 3(b)-(c), Fig. S4(b): predicted link count N-hat_c
├── config.py                paths and experiment hyper-parameters (no hard-coded paths)
└── cli.py                   `ising-ood generate-topology | train | evaluate --config <yaml>`
```

This layout follows a **stable core + thin adapters** rule: `dynamics/` and `models/` are treated as frozen reference implementations of the manuscript's physics and architectures; new experiments are expressed as new YAML configs or new scripts under `tools/`, not as edits to the core. See also the [architecture guide](docs/architecture.md) and the [figure-reproduction guide](docs/reproducing_paper_figures.md).

## Reference tools and data

```bash
# Fig. 2(a)-(c): training / ID-test / topology-shift OOD accuracy vs N_L
ising-ood evaluate --config configs/eval_topology_ood.yaml

# Fig. 3(a): temperature-shift OOD accuracy vs T
ising-ood evaluate --config configs/eval_temperature_shift.yaml
```

`data/` holds only small, human-readable reference material; large trajectory `.pt` files and topology `.xml` ensembles are regenerated locally via `tools/generate_topology_xml.py` and `tools/generate_trajectory_dataset.py`, and all paths are overridable through environment variables documented in `config.py` (e.g. `ISING_OOD_DATA_ROOT`).

## Reproducibility contract

| Level | What this repository currently provides |
|---|---|
| Install | Standard `pyproject.toml`, Python 3.9/3.11 CI, Apache-2.0 license |
| Verify | Glauber-integrator bounds test, five-architecture forward-shape tests, 66-dim tensor round-trip test, filename-discovery tests, end-to-end quickstart |
| Run | Deterministic, self-contained CPU quickstart through the same training/evaluation code paths as the paper-scale runs |
| Reuse existing data | Unified `.pt` dataset schema (`tensors`, `matrices`/`labels`, `parameters`) and checkpoint schema (`model_state_dict`, `train_acc`, `test_acc`, `class_ids`, `it_time`) shared by every consumer |
| Extend | Thin configuration files and validated dataset/model adapters |
| Reproduce manuscript analyses | ID/topology-OOD and temperature-OOD evaluation modules, plus the predicted-link-count diagnostic, are packaged and directly callable; full figure-by-figure regeneration additionally requires the paper-scale topology ensembles and checkpoints |

For scientific use, verify a small system (e.g. $L=8$, $N_L=4$) end-to-end before scaling to $L=12$, $N_L=50$, $N=35000$. Increase $N_L$, $N_c$, and $T$ independently and report the predicted-link-count diagnostic alongside accuracy — a single accuracy number is not sufficient evidence of physical rule learning.

## Numerical notes

- The Glauber integrator (`dynamics/glauber.py`) is kept bit-for-bit compatible with the original research code, including its use of the legacy global NumPy random state when no explicit generator is supplied.
- The `fixed_edges` topology mode verifies connectivity exactly via depth-first search; the `original` mode uses a random-walk connectivity check, matching the two ensembles used in the main text ($N_c=25$) and the Supplemental Material ($N_c=33$).
- `output_to_tensor` / `inverse_tensor` are exact inverses of each other on the 66-dimensional upper-triangular representation for $L=12$; this is covered by a dedicated round-trip test.
- Small tests establish implementation consistency (correct shapes, exact algebraic round-trips, bounded dynamics) — they do not certify convergence of any specific paper-scale training run.

## Scientific context

This work sits within a broader literature on distribution shift and physical inverse problems:

- P. W. Koh *et al.*, *Proceedings of ICML* (2021).
- D. Hendrycks *et al.*, *Proceedings of ICCV* (2021).
- H. C. Nguyen, R. Zecchina, and J. Berg, *Adv. Phys.* **66**, 197 (2017).
- R. J. Glauber, *J. Math. Phys.* **4**, 294 (1963).
- A. Vaswani *et al.*, *NeurIPS* (2017).
- R. Geirhos *et al.*, *Nat. Mach. Intell.* **2**, 665 (2020).

These works provide context; this repository does not claim that OOD evaluation or shortcut-learning diagnostics originated with this study. The specific contribution here is the joint use of topology-shift and temperature-shift OOD testing together with the predicted-link-count diagnostic to separate transferable dynamics-to-structure inference from architecture-dependent statistical priors, in a fully controlled kinetic Ising testbed.

## Citation

Accompanying manuscript: Yuan-Bin Zhu, Shuang Qiao, and Shi-Ju Ran, *"Out-of-distribution Neural Inference in Dynamical Ising Models"*, arXiv:2607.03039 (2026).

Until formal publication metadata is available, use `CITATION.cff` or:

```bibtex
@article{ZhuOODIsing2026,
  author  = {Zhu, Yuan-Bin and Qiao, Shuang and Ran, Shi-Ju},
  title   = {Out-of-distribution Neural Inference in Dynamical Ising Models},
  journal = {arXiv preprint},
  year    = {2026},
  eprint  = {2607.03039}
}
```

Bug reports and reproducibility questions are welcome through GitHub Issues. Contributions should preserve the Glauber-dynamics conventions and the 66-dimensional label ordering, and include a focused test.

## License

Code is released under the Apache License 2.0; see `LICENSE`.
# Reproducing the paper's figures

| Figure | What it shows | Config | Function |
|---|---|---|---|
| Fig. 2(a)/(b) | Training / ID-test accuracy vs N_L | `configs/main_nc25.yaml` (train), `configs/eval_topology_ood.yaml` (evaluate) | `evaluation.id_ood_topology.evaluate_id_accuracy_vs_nl` |
| Fig. 2(c) | Topology-shift OOD accuracy vs N_L | same training run, held-out lattices 51-500 | `evaluation.id_ood_topology.evaluate_topology_ood_vs_nl` |
| Fig. 3(a) | Temperature-shift OOD accuracy vs T | `configs/eval_temperature_shift.yaml` | `evaluation.temperature_shift.evaluate_temperature_shift_accuracy` |
| Fig. 3(b) | Predicted link count N-hat_c vs N_L | — | `evaluation.link_diagnostics.predicted_links_vs_class_count` |
| Fig. 3(c) | Predicted link count N-hat_c vs T | — | `evaluation.link_diagnostics.predicted_links_vs_temperature` |
| Fig. S4 | Balanced N_c = 33 control | `configs/control_nc33.yaml` + `--num-edges 33` for topology generation | same evaluation functions, run on the 33-links ensemble |

## Step-by-step for the main setting (N_c = 25)

    ising-ood generate-topology --config configs/topology_original.yaml
    python tools/generate_trajectory_dataset.py \
        --matrices-xml data/topologies/matrices_original.xml \
        --out-dir data/lattice_datasets_original \
        --start-matrix 1 --end-matrix 50 --samples-per-matrix 700
    ising-ood train --config configs/main_nc25.yaml
    ising-ood evaluate --config configs/eval_topology_ood.yaml
    ising-ood evaluate --config configs/eval_temperature_shift.yaml
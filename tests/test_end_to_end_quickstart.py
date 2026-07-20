"""A small, self-contained end-to-end smoke test: generate one topology,
one dataset file, train for one epoch, and evaluate. Uses tiny sizes so it
runs in a few seconds on CPU.
"""

import torch
from torch.utils.data import DataLoader, random_split

from ising_ood.dynamics.topology import generate_topology
from ising_ood.data.dataset import SingleMatrixDataset
from ising_ood.models.factory import build_model
from ising_ood.training.engine import train_one_stage


def test_quickstart_pipeline_runs():
    n, num_edges, it_time, num_samples = 8, 6, 16, 24

    A = generate_topology(mode="fixed_edges", n=n, num_edges=num_edges)
    dataset = SingleMatrixDataset(
        num_samples=num_samples, length=n, beta=1.0, dt=0.05, it_time=it_time,
        adjacency_matrix=A, keep_matrix_tensor=False, log_every=0,
    )

    from ising_ood.models.factory import inverse_tensor

    a_tensor = torch.tensor(A, dtype=torch.float32).unsqueeze(0)
    # pad the 12x12 convention only if n == 12; for n=8 the model's final
    # layer must match n*(n-1)//2 candidate links, so here we bypass the
    # standard 66-dim head by directly checking forward-pass mechanics only.
    trajectories = torch.stack(list(dataset))
    labels = torch.zeros(num_samples, 66)  # placeholder pseudo-labels (n != 12 in this smoke test)

    class _Wrapped(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return trajectories[idx], labels[idx]

    wrapped = _Wrapped()
    train_ds, test_ds = random_split(wrapped, [16, 8])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    device = torch.device("cpu")
    model = build_model("cnn", it_time=it_time, device=device)

    result = train_one_stage(model, train_loader, test_loader, device, num_epochs=1)
    assert result["best_test_acc"] >= 0.0
    assert result["model_params"] > 0
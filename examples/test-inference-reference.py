from huggingface_hub import hf_hub_download

for file in ["state_dict_e3nn.pt", "config_e3nn.yaml"]:
    hf_hub_download(
        repo_id="ibm-research/trajcast.models-arxiv2025",
        revision="main",
        filename=f"paracetamol/{file}",
        local_dir="./",
    )

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_dtype(torch.float64)

from trajcast.model.models import EfficientTrajCastModel

# initialize model
model = EfficientTrajCastModel.build_from_yaml("paracetamol/config_e3nn.yaml")

# # load state dictionary
model.load_state_dict(
    torch.load(
        "paracetamol/state_dict_e3nn.pt",
        map_location=device,
    )
)

for dataset in ["test"]:
    hf_hub_download(
        repo_id="ibm-research/trajcast.datasets-arxiv2025",
        repo_type="dataset",
        revision="main",
        filename=f"example/{dataset}.extxyz",
        local_dir="../data",
    )


from trajcast.data.dataset import AtomicGraphDataset

test_set_dict = {
    "root": ".",  # Directory where the data lies
    "name": "paracetamol_test",  # Name the processed dataset should have
    "cutoff_radius": 4.0,  # Cutoff for defining edges between nodes, should be the same as for model, will be fixed later
    "files": [
        "../data/example/test.extxyz"
    ],  # Files with the data, can be multiple ones
    "rename": True,  # Dependent on the precision it will add a tag to the processed filename
    "atom_type_mapper": {  # Mapping chemical atom types to variables within the model, not necessary but less error prone
        1: 0,  # H -> 0
        6: 1,  # C -> 1
        7: 2,  # N -> 2
        8: 3,  # O -> 3
    },
}

test_set = AtomicGraphDataset(**test_set_dict)

vel_scale = model._encoding.Normalization.stds["update_velocities"].item()
disp_scale = model._encoding.Normalization.stds["displacements"].item()

from torch_geometric.loader import DataLoader
import numpy as np

data_loader = DataLoader(test_set, batch_size=5, shuffle=False)

# get true displacements and update velociites
true_vel = test_set.update_velocities.detach().numpy()
true_disp = test_set.displacements.detach().numpy()

model.eval()
pred_vel = []
pred_disp = []
model.to(device)
with torch.no_grad():
    for batch in data_loader:
        # forward pass
        batch = model(batch.to(device))

        # save predicted displacements and velocities
        pred_disp.append(batch.target[:, 0:3].cpu().numpy() * disp_scale)
        pred_vel.append(batch.target[:, 3:].cpu().numpy() * vel_scale)

pred_disp = np.concatenate(pred_disp)
pred_vel = np.concatenate(pred_vel)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.set_aspect("equal")

ax.scatter(true_disp, pred_disp, s=3)
ax.plot([-0.4, 0.4], [-0.4, 0.4], color="k", ls="--", lw=1)

ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-0.35, 0.35)
ax.set_ylabel(r"Predicted Displacements $[\mathrm{\AA}]$")
ax.set_xlabel(r"Reference Displacements $[\mathrm{\AA}]$")
plt.show()


import ase.io

start_frame = ase.io.read(
    "../data/example/test.extxyz",
    index="-1",
)
start_frame.center()
_ = start_frame.arrays.pop("displacements")
_ = start_frame.arrays.pop("update_velocities")


protocol = {
    "units": "real",  # Units, in which the trajectory forecasting is performed. Here we use the same convention as lammps: https://docs.lammps.org/units.html
    "run": 100,  # Number of steps to be performed
    "temperature": 300.0,  # Target temperature
    "extra_dof": 6,  # Number of degrees of freedom to be subtracted from 3N for temperature definition.
    "timestep": 7.0,  # Prediction horizon the model has been trained on
    "configuration": start_frame,  # Initial configuration
    "model_type": "EfficientTrajCastModel",  # Type of model to be used
    "model": model,  # Which model to use
    "thermostat": {
        "Tdamp": 70.0
    },  # Arguments for the thermostat, if no thermostat is specified, forecast will be in NVE ensemble
    "velocities": {  # Whether to assign initial velocities (otherwise read from start_frame)
        "Temperature": 300,  # Velocities will correspond to this temperature
        "linear": True,  # Whether assigned velocities should be free of total linear momentum
        "angular": True,  # Whether assigned velocities should be free of total angular momentum
        "distribution": "gaussian",  # Distribution from which velocities are drawn (uniform or Gaussian)
    },
    "write": {  # Details on how often and where to save frames
        "filename": "./example_traj.extxyz",  # Path to the output file
        "every": 1,  # Save every Nth frame
    },
    "device": device,  # Device the forecasting will be run on
    "seed": 42,  # Seed for thermostat and initial velocities (if applicable)
    "set_momenta": {  # Directionary with target linear and/or angular momentum
        "linear": torch.zeros(3, device=device),
        "angular": torch.tensor([], device=device),
    },
    "zero_momentum": {  # Settings to zero the total linear and angular momentum induced by the thermostat
        "every": 100,  # Remove total net momenta every N steps
        "linear": False,  # Whether to remove linear momentum (should not be necessary if initiliazed with zero)
        "angular": True,  # Whether to remove angular momentum
    },
}

from trajcast.model.forecast import Forecast

forecaster = Forecast(protocol=protocol)


forecaster.generate_trajectory()

from ase import io, visualize
traj = io.read("example_traj.extxyz", index=":")
visualize.view(traj)


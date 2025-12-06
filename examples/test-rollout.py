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

import nglview

generated_traj = ase.io.read("example_traj.extxyz", index=":")
nglview.show_asetraj(generated_traj)

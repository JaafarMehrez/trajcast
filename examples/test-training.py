from huggingface_hub import hf_hub_download
from trajcast.model.training import Trainer


for dataset in ["train", "val", "test"]:
    hf_hub_download(
        repo_id="ibm-research/trajcast.datasets-arxiv2025",
        repo_type="dataset",
        revision="main",
        filename=f"example/{dataset}.extxyz",
        local_dir="../data",
    )


model_dict = {
    "precision": 32,  # Floating point precision, 32 for single, 64 for double
    "num_chem_elements": 4,  # Number of different chemical species the model means to describe
    "edge_cutoff": 4.0,  # Distance cutoff for message passing
    "num_edge_rbf": 8,  # Number of radial basis functions edge
    "num_edge_poly_cutoff": 6,  # p of polynomial cutoff function,
    "vel_max": 0.11,  # Cutoff for velocities obtained from Maxwell-Boltzmann distribution
    "num_vel_rbf": 8,  # Number of radial basis functions velocities
    "max_rotation_order": 1,  # Rotation order up to which we will resolve features and spherical harmonics
    "num_hidden_channels": 16,  # Number of features per irrep
    "num_mp_layers": 3,  # Number of message passing layers
    "edge_mlp_kwargs": {  # Settings for MLP producing weights for edge tensor product
        "n_neurons": [
            16,
            16,
            16,
        ],  # Number of neurons per layer, here 3 hidden layers with 16 neurons each
        "activation": "silu",  # Chosen activation function
    },
    "vel_mlp_kwargs": {  # Settings for MLP producing weights for velocity tensor product
        "n_neurons": [
            16,
            16,
            16,
        ],  # Number of neurons per layer, here 3 hidden layers with 16 neurons each
        "activation": "silu",  # Chosen activation function
    },
    "nl_gate_kwargs": {  # Settings for non-linear gates at the end of the update
        "activation_scalars": {  # Activation functions for scalar features
            "o": "tanh",  # Odd features
            "e": "silu",  # Even features
        },
        "activation_gates": {  # Settings for scalars which will be used to scale L>0 features after non-linearity
            "e": "silu"  # We only use even features for gating
        },
    },
    "conserve_ang_mom": True,  # Whether to conserve angular momentum,
    "o3_backend": "e3nn",  # Whether to use e3nn or cueq, we recommend the latter particularly for large systems
    "net_lin_mom": [
        0.0,
        0.0,
        0.0,
    ],  # Net linear momentum you'd expect (usually the 0 vector if momentum was properly zeroed in equilibrium MD)
    "net_ang_mom": [0.0, 0.0, 0.0],  # Same as above but for angular momentum
}


data_dict = {
    "root": ".",  # Directory where the data lies
    "name": "paracetamol_training",  # Name the processed dataset should have
    "cutoff_radius": 4.0,  # Cutoff for defining edges between nodes, should be the same as for model, will be fixed later
    "files": [
        "../data/example/train.extxyz"
    ],  # Files with the data, can be multiple ones
    "rename": True,  # Dependent on the precision it will add a tag to the processed filename
    "atom_type_mapper": {  # Mapping chemical atom types to variables within the model, not necessary but less error prone
        1: 0,  # H -> 0
        6: 1,  # C -> 1
        7: 2,  # N -> 2
        8: 3,  # O -> 3
    },
}

train_dict = {
    "seed": 1705,  # Random seed for initialisation
    "model_type": "EfficientTrajCastModel",  # The type of model you want to train, do not change!
    "device": "cpu",  # Device on which to run the training, set to cpu here in case this is run without
    "restart_latest": False,  # Whether to start from an old checkpoint
    "target_field": "target",  # Name of the field where the model will save its prediction
    "reference_fields": [  # Where the true labels are saved
        "displacements",
        "update_velocities",
    ],
    "batch_size": 10,  # How many configurations are contained in one batch, here 10
    "max_grad_norm": 0.5,  # Gradient clipping
    "num_epochs": 10,  # Maximum number of epochs to be performed, usually longer, here we just pick 10
    "criterion": {  # This is to set up the loss function, will be updated and simplified later
        "loss_type": {"main_loss": "mse"},
        "learnable_weights": False,
    },
    "optimizer": "adam",  # Which optimizer to use
    "optimizer_settings": {  # Settings for chosen optimizer
        "lr": 0.01,  # Learning rate
        "amsgrad": True,  # Whether AMSGrad is turned on
    },
    "scheduler": ["ReduceLROnPlateau"],  # Which Schedulers to use, can be multiple
    "scheduler_settings": {  # Settings dictionary for each scheduler
        "ReduceLROnPlateau": {"factor": 0.8, "patience": 25, "min_lr": 0.0001}
    },
    "chained_scheduler_hp": {  # Interaction between schedulers, this is outdated and will be removed soon
        "milestones": [
            10000000
        ],  # For using only 1 scheduler make sure this value is larger than the total number of updates
        "per_epoch": True,  # Adjust LR per epoch rather than per batch
        "monitor_lr_scheduler": False,  # For debugging, LR is monitored anyway
    },
    "tensorboard_settings": {  # Validation and tracking of weights, loss, and LR happens in tensorboard
        "loss": True,  # Track loss
        "lr": True,  # Track lr
        "loss_validation": {  # Compute validation loss based on a validation set defined in "data"
            "data": {
                "root": ".",  # Directory where the data lies
                "name": "paracetamol_validation",  # Name the processed dataset should have
                "cutoff_radius": 4.0,  # Cutoff for defining edges between nodes, should be the same as for model and training set, will be fixed later
                "files": [
                    "../data/example/val.extxyz"
                ],  # Files with the data, can be multiple ones
                "rename": True,  # Dependent on the precision it will add a tag to the processed filename
                "atom_type_mapper": {  # Mapping chemical atom types to variables within the model, should be the same as for training set
                    1: 0,  # H -> 0
                    6: 1,  # C -> 1
                    7: 2,  # N -> 2
                    8: 3,  # O -> 3
                },
            }
        },
    },
}


config = {}
config["model"] = model_dict
config["data"] = data_dict
config["training"] = train_dict

trainer = Trainer(config)

trainer.train()


trainer.dump_config_to_yaml("config_example.yaml")

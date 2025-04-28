import os
from typing import Dict, List, Optional

import ase
from ase.calculators.lammps import convert
from ase.io import lammpsrun

from trajcast.data._keys import (
    FORCES_KEY,
    TIMESTEP_KEY,
    TOTAL_ENERGY_KEY,
    VELOCITIES_KEY,
)
from trajcast.utils.misc import string2index


def lammps_dump_to_ase_atoms(
    path_to_file: str,
    lammps_units: Optional[str] = "metal",
    desired_units: Optional[str] = None,
    type_mapping: Optional[Dict] = {},
    **ase_kwargs,
) -> List[ase.Atoms]:
    """_summary_

    Args:
        path_to_file (str): _description_

    Returns:
        List[ase.Atoms]: _description_
    """

    # check whether path exists
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(
            f"Couldn't find file under path provided: {path_to_file}"
        )
    if not desired_units:
        desired_units = lammps_units

    # get the indices based on string
    index = string2index(ase_kwargs.get("index", ":"))

    # read in via ASE
    with open(path_to_file, "r") as file:
        data = lammpsrun.read_lammps_dump_text(file, index=index, units=lammps_units)

    # dependent on whether we read in a single frame or multiple frames we need to make the object into a list
    data = data if isinstance(data, List) else [data]

    # there are quite a few things we have to fix, so let's loop over the objects:
    for obj in data:
        # we will start by converting everything
        obj.arrays[VELOCITIES_KEY] = convert(
            obj.get_velocities(), "velocity", "ASE", desired_units
        )

        # now the forces
        if obj.get_calculator():
            obj.arrays[FORCES_KEY] = convert(
                obj.get_forces(), "force", "ASE", desired_units
            )
        # next change atomic numbers
        if type_mapping:
            obj.set_atomic_numbers(
                [type_mapping.get(ele) for ele in obj.get_atomic_numbers()]
            )

        # now get the correct momenta
        obj.set_momenta(
            obj.arrays[VELOCITIES_KEY]
            * convert(obj.get_masses(), "mass", "ASE", desired_units).reshape(-1, 1)
        )

        # check if we have a an energy object:
        if "v_pe" in obj.arrays.keys():
            obj.info[TOTAL_ENERGY_KEY] = convert(
                obj.arrays["v_pe"][0], "energy", lammps_units, desired_units
            )
            del obj.arrays["v_pe"]

        # delete momenta from arrays
        if "momenta" in obj.arrays.keys():
            del obj.arrays["momenta"]

        # set the celldisp to 0
        obj.positions -= obj.get_celldisp()
        obj.set_celldisp([0, 0, 0])

        # we also wrap everything into the box:
        obj.wrap()
        
        # for now drop the timestep from lammps which just enumerates
        if obj.info.get(TIMESTEP_KEY) is not None:
            obj.info.pop(TIMESTEP_KEY)
    # now go back to ASEObject if just one frame
    data = data[0] if len(data) == 1 else data

    return data

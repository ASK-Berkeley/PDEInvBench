import os
from typing import Dict

import torch
import typeguard
from jaxtyping import jaxtyped

from pdeinvbench.utils.types import PDE

"""
Hardcoded parameter normalization stats for each dataset.
These are used to normalize the parameters before training.
"""
PARAM_NORMALIZATION_STATS = {
    PDE.ReactionDiffusion2D: {
        "k": (0.06391126306498819, 0.029533048151465856),
        "Du": (0.3094992685910578, 0.13865605073673604),
        "Dv": (0.259514500345804, 0.11541850276902947),
    },
    PDE.NavierStokes2D: {"re": (1723.425, 1723.425)},
    PDE.TurbulentFlow2D: {"nu": (0.001372469573118451, 0.002146258280849241)},
    PDE.KortewegDeVries1D: {"delta": (2.899999997019768, 1.2246211546444339)},
}


@jaxtyped(typechecker=typeguard.typechecked)
def unnormalize_params(
    param_dict: Dict[str, torch.Tensor], pde: PDE
) -> Dict[str, torch.Tensor]:
    """
    Unnormalize the PDE parameters.
    """
    for param in param_dict.keys():
        if "var" not in param:
            mean, std = PARAM_NORMALIZATION_STATS[pde][param]
            param_dict[param] = param_dict[param] * std + mean
    return param_dict


@jaxtyped(typechecker=typeguard.typechecked)
def extract_params_from_path(path: str, pde: PDE) -> dict:
    """
    Extracts the PDE parameters from the h5 path and returns as a dictionary.
    """
    param_dict = {}
    if pde == PDE.ReactionDiffusion2D:
        name = os.path.basename(path)
        elements = name.split("=")[1:]
        Du = torch.Tensor([float(elements[0].split("_")[0])])
        Dv = torch.Tensor([float(elements[1].split("_")[0])])
        k = torch.Tensor(
            [float(elements[2].split(".")[0] + "." + elements[2].split(".")[1])]
        )
        param_dict = {"k": k, "Du": Du, "Dv": Dv}
    elif pde == PDE.NavierStokes2D:
        name = os.path.basename(path)
        re_string = name.split(".")[0].strip()
        re = torch.Tensor([float(re_string)])
        param_dict = {"re": re}
    elif pde == PDE.TurbulentFlow2D:
        name = os.path.basename(path)
        viscosity_string = name.split("=")[1][:-3]
        viscosity = float(viscosity_string)
        param_dict = {"nu": torch.Tensor([viscosity])}
    elif pde == PDE.KortewegDeVries1D:
        name = os.path.basename(path)
        delta = name.split("=")[-1].split("_")[0]
        param_dict = {"delta": torch.Tensor([float(delta)])}
    elif pde == PDE.DarcyFlow2D:
        # The parameter is stored as part of the h5 file so we return the parsed file index
        name = os.path.basename(path)
        index = name.split(".")[0].split("_")[-1]
        index = int(index)
        index = torch.Tensor([index])
        param_dict = {"index": index}
    else:
        raise ValueError(f"Unknown PDE type: {pde}. Cannot extract parameters.")

    if len(param_dict) == 0:
        raise ValueError(
            f"No parameters found for PDE: {pde}. Cannot extract parameters."
        )
    return param_dict

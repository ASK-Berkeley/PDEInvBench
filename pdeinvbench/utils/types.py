# Type Utilities
import enum
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, UInt8


class PDE(enum.Enum):
    """
    Describes which PDE system currently being used.
    The PDE system is used to determine the correct data loading and processing steps.
    """

    ReactionDiffusion2D = "Reaction Diffusion 2D"
    NavierStokes2D = "Navier Stokes 2D"
    TurbulentFlow2D = "Turbulent Flow 2D"
    KortewegDeVries1D = "Korteweg-de Vries 1D"
    DarcyFlow2D = "Darcy Flow 2D"


"""Define a global dictionaries of PDE attributs."""

# Number of partial derivatives for each PDE system.
PDE_PARTIALS = {
    PDE.ReactionDiffusion2D: 5,
    PDE.NavierStokes2D: 3,
    PDE.TurbulentFlow2D: 3,
    PDE.KortewegDeVries1D: 4,
    PDE.DarcyFlow2D: 4,
}

# Number of spatial dimensions for each PDE system.
PDE_NUM_SPATIAL = {
    PDE.ReactionDiffusion2D: 2,
    PDE.NavierStokes2D: 2,
    PDE.TurbulentFlow2D: 2,
    PDE.KortewegDeVries1D: 1,
    PDE.DarcyFlow2D: 2,
}

# Spatial size of the grid for each PDE system.
PDE_SPATIAL_SIZE = {
    PDE.ReactionDiffusion2D: [128, 128],
    PDE.NavierStokes2D: [64, 64],
    PDE.TurbulentFlow2D: [64, 64],
    PDE.KortewegDeVries1D: [256],
    PDE.DarcyFlow2D: [241, 241],
}


# Spatial size of the grid for each PDE system.
HIGH_RESOLUTION_PDE_SPATIAL_SIZE = {
    PDE.ReactionDiffusion2D: [512, 512],
    PDE.TurbulentFlow2D: [2048, 2048],
    PDE.DarcyFlow2D: [421, 421],
    PDE.NavierStokes2D: [256, 256],
}


# Number of parameters for each PDE system.
PDE_NUM_PARAMETERS = {
    PDE.ReactionDiffusion2D: 3,
    PDE.NavierStokes2D: 1,
    PDE.TurbulentFlow2D: 1,
    PDE.KortewegDeVries1D: 1,
    PDE.DarcyFlow2D: 128,  # NOTE: Incorrect, but we only use this in the forward problem?
}

# Range of parameter values for each PDE system.
PDE_PARAM_VALUES = {
    PDE.ReactionDiffusion2D: {
        "k": [
            0.00544908,
            0.01064798,
            0.01446092,
            0.01591103,
            0.02190137,
            0.02248171,
            0.03376446,
            0.04418002,
            0.05103662,
            0.05279494,
            0.05734164,
            0.06385121,
            0.06426775,
            0.06746974,
            0.07166788,
            0.07212561,
            0.07438393,
            0.08332919,
            0.08620312,
            0.08693649,
            0.0880078,
            0.08820963,
            0.0905649,
            0.09362309,
            0.09649866,
            0.09658253,
            0.09808294,
            0.09985239,
        ],
        "Du": [
            0.02219061,
            0.07546761,
            0.0816335,
            0.117242,
            0.1297511,
            0.1470162,
            0.1975422,
            0.2052899,
            0.2223661,
            0.2351847,
            0.238229,
            0.3073048,
            0.3356696,
            0.3410229,
            0.3570933,
            0.3594401,
            0.3844191,
            0.4004743,
            0.4182471,
            0.4187508,
            0.4282146,
            0.4363962,
            0.4394185,
            0.4521105,
            0.4605572,
            0.4644799,
            0.4954957,
            0.4978229,
        ],
        "Dv": [
            0.01647486,
            0.03266683,
            0.03295169,
            0.0336989,
            0.04517053,
            0.1197443,
            0.1431938,
            0.1512121,
            0.1513326,
            0.1761043,
            0.1856076,
            0.1935473,
            0.2369018,
            0.2541142,
            0.2725704,
            0.2871926,
            0.2925416,
            0.292952,
            0.2959587,
            0.3023561,
            0.3132344,
            0.3136975,
            0.3793569,
            0.4004971,
            0.4271173,
            0.4328981,
            0.4949132,
        ],
    },
    PDE.NavierStokes2D: {
        "re": [
            83.0,
            105.55940015,
            134.25044531,
            170.7397166,
            217.14677189,
            276.1672649,
            351.22952801,
            446.69371438,
            568.10506678,
            722.51602499,
            918.89588194,
            1168.65178436,
            1486.29134153,
            1890.26533093,
            2404.03945141,
            3057.45737879,
            3888.47430005,
            4945.36162206,
            6289.51092016,
            7999.0,
        ]
    },
    PDE.KortewegDeVries1D: {"delta": np.linspace(0.8, 5.0, 100, endpoint=True)},
    PDE.TurbulentFlow2D: {
        "nu": [
            1.00000000e-05,
            1.42792351e-05,
            2.03896555e-05,
            2.91148685e-05,
            4.15738052e-05,
            5.93642139e-05,
            8.47675566e-05,
            1.21041587e-04,
            1.72838128e-04,
            2.46799626e-04,
            3.52410989e-04,
            5.03215936e-04,
            7.18553866e-04,
            1.02603996e-03,
            1.46510658e-03,
            2.09206013e-03,
            2.98730184e-03,
            4.26563853e-03,
            6.09100555e-03,
            8.69749003e-03,
        ]
    },
    PDE.DarcyFlow2D: {
        # NOTE: This should not be used since coeff is a scalar field
        "coeff": []
    },
}

# Number of data channels for each PDE system.
PDE_NUM_CHANNELS = {
    PDE.ReactionDiffusion2D: 2,
    PDE.NavierStokes2D: 1,
    PDE.TurbulentFlow2D: 1,
    PDE.KortewegDeVries1D: 1,
    PDE.DarcyFlow2D: 1,
}

# Number of timesteps in the trajectory for each PDE system.
PDE_TRAJ_LEN = {
    PDE.ReactionDiffusion2D: 101,
    PDE.NavierStokes2D: 64,
    PDE.TurbulentFlow2D: 60,
    PDE.KortewegDeVries1D: 140,
    # This value is only used to pass some assertions so any non-zero value works
    PDE.DarcyFlow2D: 101,
}


class DataMetrics(enum.Enum):
    """
    Describes various data loss metrics, removing the need for metrics based on strings.
    """

    MSE = "Mean Squared Error"
    Relative_Error = "Relative Error"


class ParamMetrics(enum.Enum):
    """
    Describes various parameter loss metrics, removing the need for metrics based on strings.
    """

    MSE = "Mean Squared Error"
    Relative_Error = "Relative Error"


########################
# common types
TypeBatchSolField1D = Float[torch.Tensor, "batch time xspace"]
TypeBatchSolField2D = Float[torch.Tensor, "batch time channel xspace yspace"]
TypeUnBatchSolField2D = Float[torch.Tensor, "time channel xspace yspace"]

TypeXGrid = Float[torch.Tensor, "batch xspace"]
TypeYGrid = Float[torch.Tensor, "batch yspace"]
# Navier Stokes has different grid input shape
TypeNSGrid = Float[torch.Tensor, "xspace yspace"]
TypeTimeGrid = Float[torch.Tensor, "batch timesteps"]
TypeParam = Dict[
    str, Float[torch.Tensor, "batch 1"] | Float[torch.Tensor,
                                                "batch xspace yspace 1"]
]
TypeBatch = Float[torch.Tensor, "batch"]
########################
# types for logging_utils

# input dimensions for scaling functions
TypeScaleInputField1D = Float[np.ndarray, "time xspace"]
TypeScaleInputField2D = Float[np.ndarray, "time xspace yspace"]

# output dimensions for return value of scaling functions
TypeScaledField1D = UInt8[np.ndarray, "time xspace"]
TypeScaledField2D = UInt8[np.ndarray, "time 3 xspace yspace"]

# output type for return value of scaling functions
TypeLoggingField1D = Tuple[
    TypeScaledField1D,
    TypeScaledField1D,
    TypeScaledField1D,
]
TypeLoggingField2D = Tuple[
    TypeScaledField2D,
    TypeScaledField2D,
    TypeScaledField2D,
]
########################
# types for pde_module
TypeCollapsedInputSolField1D = Float[torch.Tensor,
                                     "batch channels_conditioning xspace"]
TypeCollapsedInputSolField2D = Float[
    torch.Tensor, "batch channels_conditioning xspace yspace"
]

TypeTimeFrames = Float[torch.Tensor, "batch n_past"]
TypeICIndex = Float[torch.Tensor, "batch 1"]
# input batch types for pde module


TypeBatch1D = List[
    Union[
        List[TypeXGrid],
        TypeTimeGrid,
        TypeCollapsedInputSolField1D,
        TypeTimeFrames,
        TypeICIndex,
        TypeParam,
    ]
]

TypeBatch2D = List[
    Union[
        List[Union[TypeXGrid, TypeYGrid]],
        TypeTimeGrid,
        TypeCollapsedInputSolField2D,
        TypeTimeFrames,
        TypeICIndex,
        TypeParam,
    ]
]

# output prediction types for pde module

TypePredict1D = Tuple[
    TypeBatchSolField1D,
    TypeBatchSolField1D,
    # Dict of dict so that we may store auxillary metrics during tailoring
    Dict[str, Union[torch.Tensor, Float[torch.Tensor, "batch"], Dict]],
]
TypePredict2D = Tuple[
    TypeBatchSolField2D,
    TypeBatchSolField2D,
    # Dict of dict so that we may store auxillary metrics during tailoring
    Dict[str, Union[torch.Tensor, Float[torch.Tensor, "batch"], Dict]],
]

TypeLossDict = Dict[str, Union[Dict,
                               torch.Tensor, Float[torch.Tensor, "batch"]]]

# input and output types for autoregressive rollout in pde module
TypeAutoRegressiveInitFrames = Union[
    Float[torch.Tensor, "batch n_past xspace"],
    Float[torch.Tensor, "batch n_pastxchannels xspace yspace"],
]

TypeAutoRegressivePredFrames = Union[
    Float[torch.Tensor, "batch n_fut xspace"],
    Float[torch.Tensor, "batch n_futxchannels xspace yspace"],
]
########################
# types for pde_residual

# Shape of partial differentials computed for 1d and 2d systems
TypePartials1D = TypeBatchSolField1D
TypePartials2D = Float[torch.Tensor, "batch time xspace yspace"]
TypeNSPartials2D = Float[torch.Tensor, "batch 3 time xspace yspace"]

# return type for advection residual + partials
TypeAdvectionPartialsReturnType = Union[
    TypePartials1D,
    Tuple[
        TypePartials1D,
        Float[torch.Tensor, "batch 2*time xspace"],
    ],
]
# return type for burgers residual + partials
TypeBurgersPartialsReturnType = Union[
    TypePartials1D,
    Tuple[
        TypePartials1D,
        Float[torch.Tensor, "batch 3*time xspace"],
    ],
]
# return type for 1drd residual + partials
Type1DRDPartialsReturnType = Union[
    TypePartials1D,
    Tuple[
        TypePartials1D,
        Float[torch.Tensor, "batch 2*time xspace"],
    ],
]

# Return type for 1D KDV residual + partials
Type1DKDVPartialsReturnType = Union[
    TypePartials1D,
    Tuple[
        TypePartials1D,
        Float[torch.Tensor, "batch 4*time xspace"],
    ],
]

# return type for 2drd residual + partials
Type2DRDPartialsReturnType = Union[
    TypeBatchSolField2D,
    Tuple[
        TypeBatchSolField2D,
        Float[torch.Tensor, "batch time*5 channels xspace yspace"],
    ],
]

# return types for 2dns residual + partials
TypeUnBatchedNSPartials2D = Float[torch.Tensor, "3 time xspace yspace"]
TypeUnBatchedNSResiduals2D = Float[torch.Tensor, "time xspace yspace"]


##################################
# types for finite_differences

# Return type after computing all needed partials
Type1DRPartialsTuple = Tuple[
    TypeBatchSolField1D,
    TypeBatchSolField1D,
    TypeBatchSolField1D,
    TypeBatchSolField1D,
]
Type2DRDPartialsTuple = Tuple[
    TypePartials2D,
    TypePartials2D,
    TypePartials2D,
    TypePartials2D,
    TypePartials2D,
]

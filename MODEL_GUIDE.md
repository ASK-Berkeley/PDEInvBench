# PDEInvBench
## Adding a New Model

The PDEInvBench framework is designed to be modular, allowing you to easily add new model architectures. This section describes how to add a new encoder architecture to the repository.

## Table of Contents
- [Model Architecture Components](#model-architecture-components)
- [Adding a new model](#adding-a-new-model)
    - [Step 1: Create a New Encoder Class](#step-1-create-a-new-encoder-class)
    - [Step 2: Import and Register Your Model](#step-2-import-and-register-your-model)
    - [Step 3: Create a Configuration File](#step-3-create-a-configuration-file)
    - [Step 4: Run Experiments with Your Model](#step-4-run-experiments-with-your-model)

## Model Architecture Components

The inverse model architecture in PDEInvBench consists of three main components:


```
Input Solution Field → Encoder → Downsampler → Parameter Network → PDE Parameters
```

1. **Encoder**: Extracts features from the input solution field (e.g., FNO, ResNet, ScOT)
2. **Downsampler**: Reduces the spatial dimensions of the features (e.g., ConvDownsampler)
3. **Parameter Network**: Predicts PDE parameters from the downsampled features


## Adding a new model

When creating a new model, you typically only need to modify one of these components while keeping the others the same.

### Step 1: Create a New Encoder Class

First, create a new encoder class in `pdeinvbench/models/encoder.py`. Your new encoder should follow the interface of existing encoders like `FNOEncoder`, `ResnetEncoder`, or `SwinEncoder`:

```python
import torch
import torch.nn as nn
from pdeinvbench.utils.types import PDE
from pdeinvbench.models.encoder import resolve_number_input_channels

class YourEncoder(nn.Module):
    """
    Your custom encoder for PDE inverse problems.
    """
    
    def __init__(
        self,
        n_modes: int,  # Or equivalent parameter for your architecture
        n_layers: int,
        n_past: int,
        n_future: int,
        pde: PDE,
        data_channels: int,
        hidden_channels: int,
        use_partials: bool,
        mode: str,
        batch_size: int
        # Add any architecture-specific parameters
    ):
        super(YourEncoder, self).__init__()
        
        # Store essential parameters
        self.n_past = n_past
        self.n_future = n_future
        self.pde = pde
        self.data_channels = data_channels
        self.hidden_channels = hidden_channels
        self.use_partials = use_partials
        self.mode = mode
        self.batch_size = batch_size

        
        # Calculate input channels similar to existing encoders
        in_channels = resolve_number_input_channels(
            n_past=n_past,
            data_channels=data_channels,
            use_partials=use_partials,
            pde=pde,
        )
        
        # Define your model architecture
        # Example: Custom neural network layers
        self.encoder_layers = nn.ModuleList([
            # Your custom layers here
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # Add more layers as needed
        ])
        
        # Output layer to match expected output dimensions
        self.output_layer = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        
    def forward(self, x, **kwargs):
        """
        Forward pass of your encoder.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            **kwargs: Additional arguments (may include 't' for time-dependent models)
            
        Returns:
            Output tensor of shape [batch, hidden_channels, height, width]
        """
        # Implement your forward pass
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x
```

#### Creating Custom Downsamplers

If you need a custom downsampler, create it in `pdeinvbench/models/downsampler.py`:

```python
import torch
import torch.nn as nn

class YourDownsampler(nn.Module):
    """
    Your custom downsampler for reducing spatial dimensions.
    """
    
    def __init__(
        self,
        input_dimension: int,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super(YourDownsampler, self).__init__()
        
        # Define your downsampling layers
        self.layers = nn.ModuleList([
            # Your custom downsampling layers here
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
    def forward(self, x):
        """
        Forward pass of your downsampler.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Downsampled tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
```

#### Creating Custom MLPs

If you need a custom MLP, create it in `pdeinvbench/models/mlp.py`:

```python
import torch
import torch.nn as nn

class YourMLP(nn.Module):
    """
    Your custom MLP for parameter prediction.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        dropout: float,
        out_dim: int,
        num_layers: int,
        activation: str,
    ):
        super(YourMLP, self).__init__()
        
        # Define your MLP layers
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_size
            
        layers.append(nn.Linear(current_dim, out_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of your MLP.
        
        Args:
            x: Input tensor of shape [batch, features]
            
        Returns:
            Output tensor of shape [batch, out_dim]
        """
        return self.layers(x)
```

### Step 2: Import and Register Your Model

Make sure your encoder is imported in `pdeinvbench/models/__init__.py`:

```python
from .encoder import FNOEncoder, ResnetEncoder, ScOTEncoder, YourEncoder
```

This makes your encoder available for use in configuration files.

### Step 3: Create a Configuration File

The configuration system has three levels:

#### 3.1: Create Model Architecture Config

Create `configs/model/yourmodel.yaml`:

```yaml
# configs/model/yourmodel.yaml
name: "${system_params.name}_yourmodel"
dropout: ${system_params.yourmodel_dropout}
predict_variance: False
hidden_channels: ${system_params.yourmodel_hidden_channels}
encoder_layers: ${system_params.yourmodel_encoder_layers}
downsampler_layers: ${system_params.yourmodel_downsampler_layers}
mlp_layers: ${system_params.yourmodel_mlp_layers}

model_config:
  _target_: pdeinvbench.models.inverse_model.InverseModel
  paramnet: 
    _target_: pdeinvbench.models.inverse_model.ParameterNet
    pde: ${data.pde}
    normalize: ${system_params.normalize}
    logspace: ${system_params.logspace}
    params_to_predict: ${system_params.params_to_predict}
    predict_variance: ${model.predict_variance}
    mlp_type: ${system_params.mlp_type}
    encoder:
      _target_: pdeinvbench.models.encoder.YourEncoder
      n_modes: ${system_params.yourmodel_n_modes}
      n_past: ${n_past}
      n_future: ${n_future}
      n_layers: ${model.encoder_layers}
      data_channels: ${data.num_channels}
      hidden_channels: ${model.hidden_channels}
      use_partials: True
      pde: ${data.pde}
      mode: ${mode}
      batch_size: ${data.batch_size}
      use_cn: false
      task: inverse
    downsampler: ${system_params.yourmodel_downsampler}
    mlp_hidden_size: ${model.hidden_channels}
    mlp_layers: ${model.mlp_layers}
    mlp_activation: "relu"
    mlp_dropout: ${model.dropout}
    downsample_factor: ${data.downsample_factor}
```

#### 3.2: Add Defaults to `configs/system_params/base.yaml`

Add architecture defaults that work across all PDE systems:

```yaml
# configs/system_params/base.yaml

# ============ YourModel Architecture ============
yourmodel_hidden_channels: 64
yourmodel_encoder_layers: 4
yourmodel_downsampler_layers: 4
yourmodel_dropout: 0
yourmodel_mlp_layers: 1
yourmodel_n_modes: 16

yourmodel_downsampler:
  _target_: pdeinvbench.models.downsampler.ConvDownsampler
  input_dimension: ${system_params.downsampler_input_dim}
  n_layers: ${model.downsampler_layers}
  in_channels: ${model.hidden_channels}
  out_channels: ${model.hidden_channels}
  kernel_size: 3
  stride: 1
  padding: 2
  dropout: ${model.dropout}
```

#### 3.3: (Optional) Add System-Specific Overrides

Override defaults for specific systems in `configs/system_params/{system}.yaml`:

```yaml
# configs/system_params/2dtf.yaml
defaults:
  - base

# ... existing system config ...

# Override architecture for this system
yourmodel_hidden_channels: 128  # Needs larger model
yourmodel_encoder_layers: 6
```

**That's it!** Your model now works with all PDE systems:
```bash
python train_inverse.py --config-name=1dkdv model=yourmodel
python train_inverse.py --config-name=2dtf model=yourmodel
```


#### Important Notes

- **System-specific parameters** (like `params_to_predict`, `normalize`, `downsampler_input_dim`) go in `configs/system_params/{system}.yaml`
- **Architecture defaults** go in `configs/system_params/base.yaml`
- **Model structure** goes in `configs/model/{architecture}.yaml`
- For special cases like Darcy Flow, override the downsampler in the system_params file:
  ```yaml
  # configs/system_params/2ddf.yaml
  yourmodel_downsampler:
    _target_: pdeinvbench.models.downsampler.IdentityMap
  ```

### Step 4: Run Experiments with Your Model

You can now run experiments with your custom model on **any** PDE system:

```bash
# Use your model with different PDE systems
python train_inverse.py --config-name=1dkdv model=yourmodel
python train_inverse.py --config-name=2dtf model=yourmodel
python train_inverse.py --config-name=2dns model=yourmodel

# Use model variants if you created them
python train_inverse.py --config-name=2drdk model=yourmodel_large

# Override parameters from command line
python train_inverse.py --config-name=2dtf model=yourmodel model.hidden_channels=96

# Combine multiple overrides
python train_inverse.py --config-name=2ddf model=yourmodel data.batch_size=16 model.encoder_layers=6
```

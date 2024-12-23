from typing import Tuple

import torch
from torch import nn

from src.dataset.intercropping.intercropping import Intercropping
from src.dataset.intercropping.utils.parameters import ExperimentalSite, CropManagementPractices, \
    IntercroppingDescriptors, Crops, get_one_hot_encoding


class PositiveLogPlusOne(nn.Module):
    """
    Custom activation function that computes log(x + 1).
    Ensures output is always defined (x + 1 > 0) and preserves small values better than raw log.
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-0.99999)
        x = self.relu(torch.log1p(x))
        return x


class IntercroppingMLP(nn.Module, Intercropping):
    """
    A Multi-Layer Perceptron (MLP) model for predicting intercropping outcomes.

    The neural network processes crop types, intercropping descriptors, site conditions,
    and management practices to predict two target variables related to intercropping performance.

    :param hidden_layers_sizes: Sizes of hidden layers, defaults to [128, 64]
    :param act_function: Activation function for hidden layers. Options: 'leakyrelu', 'relu', 'sigmoid', 'tanh', 'softmax'
    :param device: Device to run the model on (GPU if available, else CPU)

    :ivar hidden_layers: List of linear layers that form the hidden layers of the network
    :ivar output: Final linear layer producing the output predictions
    :ivar act: Activation function used in hidden layers
    :ivar output_act: PositiveLogPlusOne activation for output layer
    :ivar dropout: Dropout layer with p=0.2 for regularization
    :ivar device: Device where the model operates

    Input Features Structure:
        - One-hot encoded crop 1 features (118 dimensions)
        - One-hot encoded crop 2 features (118 dimensions)
        - Intercropping descriptor features (8 dimensions)
        - Site condition features (33 dimensions)
        - Crop management practice features (8 dimensions)
        Total: 285 dimensions

    Example:
         # Initialize model
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         model = IntercroppingMLP(
             hidden_layers_sizes=[256, 128, 64],
             act_function='leakyrelu',
             device=device
         )
         
         # Prepare inputs
         crop_1 = Crops.WHEAT
         crop_2 = Crops.MAIZE
         intercropping_desc = IntercroppingDescriptors(
             planting_density=100,
             row_spacing=30,
             planting_pattern="alternate rows"
         )
         site_conditions = ExperimentalSite(
             soil_type="clay",
             rainfall=800,
             temperature=25,
             latitude=45.5
         )
         management = CropManagementPractices(
             irrigation=True,
             fertilizer_amount=150
         )
         
         # Get predictions
         predictions = model.get_results(
             crop_1=crop_1,
             crop_2=crop_2,
             intercropping_description=intercropping_desc,
             site_conditions=site_conditions,
             crop_management_practices=management
         )

    Note:
        - Uses Kaiming initialization for weights
        - Applies dropout (p=0.2) after each hidden layer
        - Output uses PositiveLogPlusOne activation
        - All numerical inputs should be normalized
    """
    _INPUT_DIM = 285

    def __init__(self, hidden_layers_sizes: tuple[int,...] = None, act_function='LeakyReLU', device: torch.device = None):
        super().__init__()
        hidden_layers_sizes = list(hidden_layers_sizes) if hidden_layers_sizes else [128, 64]
        hidden_layers_sizes.insert(0, self._INPUT_DIM)
        self.hidden_layers = []
        for i in range(len(hidden_layers_sizes) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i + 1])
            )

        self.output = nn.Linear(64, 2)

        act_functions = {
            'leakyrelu': nn.LeakyReLU,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'softmax': nn.Softmax
        }
        if act_function.lower() not in act_functions.keys():
            raise ValueError("Activation function ", act_function, " is not allowed! Choose between: ",
                             list(act_functions.keys()))

        self.act = act_functions[act_function.lower()]()
        self.output_act = PositiveLogPlusOne()
        self.dropout = nn.Dropout(0.2)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor containing concatenated features:
                 - One-hot encoded crop 1 features (118 dimensions)
                 - One-hot encoded crop 2 features (118 dimensions)
                 - Intercropping descriptor features (8 dimensions)
                 - Site condition features (33 dimensions)
                 - Crop management practice features (6 dimensions)
        :type x: torch.Tensor of shape (batch_size, 283)
        :return: Predicted values for two target variables related to intercropping performance. Tensor of shape (batch_size,2)
        """
        for layer in self.hidden_layers:
            x = self.dropout(self.act(layer(x)))
        return self.output_act(self.output(x))

    def init_weights(self, seed: int = 42, hidden_activation: str = 'leaky_relu',
                     output_activation: str = 'linear') -> None:
        """
        Initialize model weights for a fully connected neural network.

        Args:
            seed (int): Random seed for reproducibility
            hidden_activation (str): Activation function used in hidden layers
            output_activation (str): Activation function used in output layer
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]

        # Initialize all hidden layers
        for layer in linear_layers[:-1]:  # All except last layer
            nn.init.kaiming_normal_(
                layer.weight,
                mode='fan_in',
                nonlinearity=hidden_activation
            )
            nn.init.zeros_(layer.bias)

        # Initialize output layer
        output_layer = linear_layers[-1]
        nn.init.kaiming_normal_(
            output_layer.weight,
            mode='fan_in',
            nonlinearity=output_activation
        )
        nn.init.zeros_(output_layer.bias)

    def encode_conditions(self, crop_1: Crops, crop_2: Crops,
                          intercropping_description: IntercroppingDescriptors,
                          site_conditions: ExperimentalSite,
                          crop_management_practices: CropManagementPractices):
        """
            Encodes all input conditions into a single tensor for model processing.

            :param crop_1: First crop in the intercropping system
            :param crop_2: Second crop in the intercropping system
            :param intercropping_description: Descriptors of the intercropping setup
            :param site_conditions: Environmental and site-specific conditions
            :param crop_management_practices: Agricultural management practices
            :return: Concatenated tensor of all encoded features

            The output tensor structure (285 dimensions):
                - Crop 1 one-hot encoding (118 dimensions)
                - Crop 2 one-hot encoding (118 dimensions)
                - Intercropping descriptors (8 dimensions)
                - Site conditions (33 dimensions)
                - Crop management practices (8 dimensions)

            Note:
                All input objects must implement a to_tensor() method that returns
                their tensor representation compatible with the expected dimensions.
            """
        crop_1_tensor = torch.tensor(get_one_hot_encoding(crop_1, Crops), device=self.device)
        crop_2_tensor = torch.tensor(get_one_hot_encoding(crop_2, Crops), device=self.device)
        intercropping_description_tensor = intercropping_description.to_tensor(self.device)
        site_conditions_tensor = site_conditions.to_tensor(self.device)
        crop_management_practices_tensor = crop_management_practices.to_tensor(self.device)
        final_encoding = torch.cat(
            [crop_1_tensor, crop_2_tensor, intercropping_description_tensor, site_conditions_tensor,
             crop_management_practices_tensor])
        return final_encoding

    def get_results(self, crop_1: Crops, crop_2: Crops, intercropping_description: IntercroppingDescriptors,
                    site_conditions: ExperimentalSite,
                    crop_management_practices: CropManagementPractices) \
            -> Tuple[float, float]:
        """
        See super class for documentation.
        """
        encoding = self.encode_conditions(crop_1, crop_2, intercropping_description, site_conditions,
                                          crop_management_practices)
        results = self.forward(encoding)
        return results.detach().cpu().tolist()


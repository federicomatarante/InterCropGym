import torch
import torch.nn as nn
from numpy import floating
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, Any

from src.dataset.intercropping.intercropping_augmenter import IntercroppingAugmenter


class TrainingStepHandler:
    """
    Handles training and validation steps for neural network models, including data preparation,
    augmentation, and training loop operations.
    
    This class provides utilities for managing the training process, including dataset preparation,
    data augmentation, batch processing, and validation steps.
    
    :param model: The neural network model to be trained
    :param device: The device to run the training on (cpu/cuda)
    :param criterion: Loss function to use (defaults to MSELoss)
    
    Examples:
        Basic usage:
         model = YourModel()
         handler = TrainingStepHandler(
             model=model,
             device='cuda',
             criterion=nn.MSELoss()
         )
         
         # Prepare datasets
         train_loader, val_loader = handler.prepare_datasets(
             train_dataset=train_data,
             val_dataset=val_data,
             batch_size=32
         )
        
        Training with data augmentation:
         augment_params = {
             'random_masking': {
                 'masking_sample_rate': 0.3,
                 'masking_probability': 0.15
             },
             'random_crops_swap': {
                 'crop_swap_sample_rate': 0.3
             }
         }
         train_loader, val_loader = handler.prepare_datasets(
             train_dataset=train_data,
             val_dataset=val_data,
             augment=True,
             augment_params=augment_params
         )
        
        Training step:
         optimizer = torch.optim.Adam(model.parameters())
         for batch in train_loader:
             loss = handler.training_step(
                 batch=batch,
                 optimizer=optimizer,
                 clip_grad=True,
                 max_norm=1.0
             )
        
        Validation:
         val_loss = handler.validation_step(val_loader)
    """
    def __init__(self, model, device, criterion=None):

        self.model = model
        self.device = device
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    def prepare_datasets(
            self,
            train_dataset: list,
            val_dataset: Optional[list] = None,
            batch_size: int = 32,
            augment: bool = True,
            augment_params: Dict = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepares training and validation datasets by applying augmentation if specified
        and creating appropriate DataLoader objects.

        :param train_dataset: Raw training dataset
        :param val_dataset: Optional raw validation dataset
        :param batch_size: Size of batches for DataLoader
        :param augment: Whether to apply data augmentation
        :param augment_params: Dictionary containing augmentation parameters
        :return: Tuple of (train_loader, val_loader) where val_loader may be None
        """
        # Handle data augmentation
        if augment  and augment_params is not None:
            train_dataset = self._augment_dataset(
                train_dataset,
                augment_params
            )

        # Prepare training data
        train_loader = self._prepare_single_dataset(
            train_dataset,
            batch_size,
            shuffle=True
        )

        # Prepare validation data if provided
        val_loader = None
        if val_dataset is not None:
            val_loader = self._prepare_single_dataset(
                val_dataset,
                batch_size,
                shuffle=False
            )

        return train_loader, val_loader

    def _augment_dataset(
            self,
            dataset: list,
            params: Dict
    ) -> list:
        """
        Applies specified augmentation techniques to the dataset.

        :param dataset: Original dataset to augment
        :param params: Dictionary containing augmentation parameters including:
            - masking_sample_rate: Rate of masking augmentation
            - masking_probability: Probability of masking
            - crop_swap_sample_rate: Rate of crop swap augmentation
        :return: Augmented dataset
        """
        augmenter = IntercroppingAugmenter(dataset)
        augmented_dataset = dataset.copy()

        # Apply masking augmentation
        if hasattr(augmenter, 'random_masking'):
            masked_data = augmenter.random_masking(
                params.get('masking_sample_rate', 0.3),
                params.get('masking_probability', 0.15)
            )
            augmented_dataset += masked_data

        # Apply crop swap augmentation
        if hasattr(augmenter, 'random_crops_swap'):
            swapped_data = augmenter.random_crops_swap(
                params.get('crop_swap_sample_rate', 0.3)
            )
            augmented_dataset += swapped_data

        return augmented_dataset

    def _prepare_single_dataset(
            self,
            dataset: list,
            batch_size: int,
            shuffle: bool = True
    ) -> DataLoader:
        """
        Prepares a single dataset for training or validation by encoding conditions
        and creating a DataLoader.

        :param dataset: Raw dataset to prepare
        :param batch_size: Size of batches for DataLoader
        :param shuffle: Whether to shuffle the dataset
        :return: DataLoader configured with the prepared dataset
        """
        # Encode input conditions
        x = torch.stack([
            self.model.encode_conditions(*conditions)
            for conditions, _ in dataset
        ])

        # Prepare target values
        y = torch.tensor(
            [[t1, t2] for _, (t1, t2) in dataset],
            device=self.device
        )

        # Create TensorDataset and DataLoader
        tensor_dataset = TensorDataset(x, y)
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            optimizer: torch.optim.Optimizer,
            clip_grad: bool = True,
            max_norm: float = 1.0
    ) -> float:
        """
        Executes a single training step including forward pass, loss calculation,
        backpropagation, and optimization.

        :param batch: Tuple containing input tensors and target values
        :param optimizer: Optimizer instance for updating model parameters
        :param clip_grad: Whether to apply gradient clipping
        :param max_norm: Maximum norm for gradient clipping
        :return: Loss value for this training step
        """
        inputs, targets = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping if enabled
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=max_norm
            )

        # Optimizer step
        optimizer.step()

        return loss.item()

    def validation_step(
            self,
            val_loader: DataLoader
    ) -> floating[Any]:
        """
        Performs validation on the provided validation dataset.

        :param val_loader: DataLoader containing validation data
        :return: Average validation loss across all batches
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_losses.append(loss.item())

        return np.mean(val_losses)

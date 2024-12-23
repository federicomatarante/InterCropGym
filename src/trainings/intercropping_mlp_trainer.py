import os
import random
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.dataset.intercropping.intercropping_mlp import IntercroppingMLP
from src.dataset.intercropping.readers.dataset_loader import DatasetLoader
from src.trainings.utils.training_step_handler import TrainingStepHandler
from src.utils.config_reader import ConfigReader
from src.trainings.utils.training_plotter import TrainingPlotter


class IntercroppingMLPTrainer:
    """
    A trainer class for managing the training process of the IntercroppingMLP model.

    Handles dataset splitting, model training, evaluation, checkpointing, and various 
    training configurations such as early stopping and learning rate scheduling.

    :param model: The IntercroppingMLP model instance to be trained
    :param config_reader: Configuration reader containing training parameters
    :param dataset_loader: Loader for the intercropping dataset

    Examples:
        Basic training workflow:
         from src.dataset.intercropping.intercropping_mlp import IntercroppingMLP
         from src.utils.ini_config_reader import INIConfigReader
         from src.dataset.intercropping.readers.dataset_loader import DatasetLoader
         
         # Initialize components
         model = IntercroppingMLP()
         config_reader = INIConfigReader("path/to/config.ini")
         dataset_loader = DatasetLoader("path/to/data.csv")
         
         # Create trainer and start training
         trainer = IntercroppingMLPTrainer(model, config_reader, dataset_loader)
         history = trainer.train(log_level=1, plot=True)
         
         # Evaluate the model
         metrics = trainer.evaluate(log_level=1)
         print(f"Test MSE: {metrics['mse_output1']:.4f}")

        Resume training from checkpoint:
         trainer = IntercroppingMLPTrainer(model, config_reader, dataset_loader)
         history = trainer.train(
             start_from_checkpoint="checkpoint_epoch_50.pth",
             log_level=2
         )

        Training with custom settings:
         trainer = IntercroppingMLPTrainer(model, config_reader, dataset_loader)
         # Training without augmentation and plots
         history = trainer.train(
             log_level=2,
             plot=False,
             augment=False
         )
         
         # Detailed evaluation
         metrics = trainer.evaluate(log_level=3)
         print(f"R2 Score: {metrics['r2_output1']:.4f}")

        Loading and using a trained model:
         trainer = IntercroppingMLPTrainer(model, config_reader, dataset_loader)
         checkpoint = trainer.load_checkpoint('best_model.pth')
         # Model is now ready for inference
         metrics = trainer.evaluate(log_level=1)
    """

    def __init__(self, model: IntercroppingMLP, config_reader: ConfigReader, dataset_loader: DatasetLoader):
        self.model = model
        self._load_config(config_reader)
        dataset = dataset_loader.load()
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(dataset)
        random.seed(self.seed)
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        self.plotter = TrainingPlotter()
        self.training_handler = TrainingStepHandler(
            model=model,
            device=model.device,
            criterion=nn.MSELoss()
        )

    def _split_dataset(self, dataset: list) -> tuple[list, list, list]:
        """
        Splits the input dataset into training, validation, and test sets based on configured ratios.
        :param dataset: Complete dataset to be split
        :return: Tuple containing (train_dataset, validation_dataset, test_dataset)
        """
        train_len, val_len, test_len = [int(data * len(dataset)) for data in self.data_split]
        random.shuffle(dataset)
        return (dataset[0:train_len],
                dataset[train_len: train_len + val_len],
                dataset[train_len + val_len: train_len + val_len + test_len])

    def _load_config(self, config_reader: ConfigReader) -> None:
        """
        Loads and validates all training configuration parameters from the config reader.

        Handles training parameters, checkpoint settings, early stopping criteria,
        dataset split ratios, and data augmentation parameters.

        :param config_reader: Configuration reader containing all training parameters
        :raises ValueError: If configuration format or values are invalid
        :raises KeyError: If required configuration parameters are missing
        :raises FileNotFoundError: If configuration file cannot be found
        """
        self.config_data = config_reader.config_data

        # Training parameters
        self.epochs = config_reader.get_param('training.epochs', v_type=int)
        self.batch_size = config_reader.get_param('training.batch_size', v_type=int)
        self.learning_rate = config_reader.get_param('training.learning_rate', v_type=float)
        self.weight_decay = config_reader.get_param('training.weight_decay', v_type=float)
        self.seed = config_reader.get_param("training.seed", v_type=int)

        # Checkpoint parameters
        self.save_frequency = config_reader.get_param('checkpoints.save_frequency', v_type=int)
        self.save_path = config_reader.get_param('checkpoints.save_path', v_type=str)
        self.log_path = config_reader.get_param('checkpoints.log_path', v_type=str)

        # Early stopping parameters
        self.early_stop_patience = config_reader.get_param('early_stopping.early_stop_patience', v_type=int)
        self.early_stop_min_improvement = config_reader.get_param(
            'early_stopping.early_stop_min_improvement', v_type=float)

        # Dataset parameters
        self.data_split = config_reader.get_param('dataset.data_split', v_type=tuple)

        if len(self.data_split) != 3:
            print(len(self.data_split))
            raise ValueError("Config file error: data split must be a tuple of 3 float values")
        try:
            self.data_split = tuple(float(split) for split in self.data_split)
        except TypeError as e:
            print(e)
            raise ValueError("Config file error: data split must be a tuple of 3 float values")
        if sum(self.data_split) >= 1:
            raise ValueError("Config file error: data split must sum up to 1")

        # Data augmentation parameters
        self.crop_swap_sample_rate = config_reader.get_param('data_augmentation.crop_swap_sample_rate', v_type=float)
        self.masking_sample_rate = config_reader.get_param('data_augmentation.masking_sample_rate', v_type=float)
        self.masking_probability = config_reader.get_param('data_augmentation.masking_probability', v_type=float)

    def _save_checkpoint(self, epoch: int, history: Dict, optimizer: torch.optim.Optimizer,
                         scheduler: ReduceLROnPlateau, loss: np.floating, is_best: bool = False) -> None:
        """
        Saves a checkpoint of the current training state.

        :param epoch: Current training epoch number
        :param history: Dictionary containing training history metrics
        :param optimizer: Current state of the optimizer
        :param scheduler: Current state of the learning rate scheduler
        :param loss: Current loss value
        :param is_best: Flag indicating if current model is the best performing so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'config': self.config_data,
            'history': history
        }

        # Save regular checkpoint
        if epoch % self.save_frequency == 0:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)

        # Save best model checkpoint
        if is_best:
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(checkpoint, best_model_path)

    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Loads a previously saved checkpoint file.

        :param checkpoint_file: Name of the checkpoint file to load
        :return: Dictionary containing the loaded checkpoint data
        :raises FileNotFoundError: If checkpoint file does not exist
        """
        checkpoint_path = os.path.join(self.save_path, checkpoint_file)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def evaluate(self, log_level: int = 0) -> Dict[str, float]:
        """
        Evaluates the model's performance on the test dataset.

        :param log_level: Controls logging verbosity:
            0 - Silent mode
            1 - Basic metrics at the end
            2 - Batch-level logging
            3 - Detailed metrics and predictions
        :return: Dictionary containing evaluation metrics including MSE, MAE, R2 scores
        :raises ValueError: If test dataset is empty
        """
        if not self.test_dataset:
            raise ValueError("Cannot test: test dataset is empty!")
        self.load_checkpoint('best_model.pth')
        self.model.eval()

        if log_level >= 1:
            print(f"Starting evaluation on {len(self.test_dataset)} samples...")

        # Prepare test data
        x_test = torch.stack([self.model.encode_conditions(*conditions) for conditions, _ in self.test_dataset])
        y_test = torch.tensor([[t1, t2] for _, (t1, t2) in self.test_dataset], device=self.model.device)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        if log_level >= 2:
            print(f"Data prepared: {len(test_loader)} batches of size {self.batch_size}")

        # Initialize metrics
        criterion = nn.MSELoss(reduction='none')
        total_mse = np.zeros(2)
        total_mae = np.zeros(2)
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_idx, (inputs, batch_targets) in enumerate(test_loader):
                outputs = self.model(inputs)

                # Calculate MSE and MAE
                mse = criterion(outputs, batch_targets)
                mae = torch.abs(outputs - batch_targets)
                total_mse = np.concatenate([mse.sum(dim=0).cpu().numpy(), total_mse])
                total_mae = np.concatenate([mae.sum(dim=0).cpu().numpy(), total_mae])

                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())

                if log_level >= 2:
                    batch_mse = mse.mean(dim=0).cpu().numpy()
                    print(f"Batch {batch_idx + 1}/{len(test_loader)}:")
                    print(f"  MSE: [{batch_mse[0]:.4f}, {batch_mse[1]:.4f}]")

        n_samples = len(self.test_dataset)
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate R-squared for each output
        r2_scores = []
        for i in range(2):  # For both outputs
            ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
            ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)

        metrics = {
            'mse_output1': (total_mse[0] / n_samples).item(),
            'mse_output2': (total_mse[1] / n_samples).item(),
            'mae_output1': (total_mae[0] / n_samples).item(),
            'mae_output2': (total_mae[1] / n_samples).item(),
            'r2_output1': r2_scores[0],
            'r2_output2': r2_scores[1],
            'rmse_output1': np.sqrt((total_mse[0] / n_samples).item()),
            'rmse_output2': np.sqrt((total_mse[1] / n_samples).item())
        }

        if log_level >= 1:
            print("\nEvaluation Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        if log_level >= 3:
            print("\nDetailed Statistics:")
            for i in range(2):
                output_predictions = predictions[:, i]
                output_targets = targets[:, i]
                print(f"\nOutput {i + 1}:")
                print(f"  Mean prediction: {np.mean(output_predictions):.4f}")
                print(f"  Mean target: {np.mean(output_targets):.4f}")
                print(f"  Std prediction: {np.std(output_predictions):.4f}")
                print(f"  Std target: {np.std(output_targets):.4f}")
                print(f"  Min prediction: {np.min(output_predictions):.4f}")
                print(f"  Max prediction: {np.max(output_predictions):.4f}")

        return metrics

    def train(
            self,
            log_level: int = 1,
            plot: bool = True,
            augment: bool = True,
            start_from_checkpoint: str = None
    ) -> Dict[str, Any]:
        """
        Trains the model using the configured parameters.

        :param log_level: Controls logging verbosity during training
        :param plot: Whether to plot training metrics in real-time
        :param augment: Whether to use data augmentation during training
        :param start_from_checkpoint: Optional checkpoint file to resume training from
        :return: Dictionary containing training history and metrics
        """
        # Initialize from checkpoint if provided
        start_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'learning_rates': []
        }

        if start_from_checkpoint:
            if log_level >= 1:
                print(f"Loading checkpoint: {start_from_checkpoint}")
            checkpoint = self.load_checkpoint(start_from_checkpoint)
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            history = checkpoint['history']

            # Initialize optimizer and scheduler with checkpoint state
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                          verbose=True if log_level >= 1 else False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.model.init_weights(self.seed)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                          verbose=True if log_level >= 1 else False)

        if augment and log_level > 1:
            print(
                f"Augmenting the dataset. "
                f"\nMasking Sample Rate: {self.masking_sample_rate} - Masking Probability {self.masking_probability} "
                f"- Crop Swap Sample Rate {self.crop_swap_sample_rate}")

        augment_params = {
            'random_masking': {
                'masking_sample_rate': self.masking_sample_rate,
                'masking_probability': self.masking_probability
            },
            'random_crops_swap': {
                'crop_swap_sample_rate': self.crop_swap_sample_rate
            }
        }

        train_loader, val_loader = self.training_handler.prepare_datasets(
            self.train_dataset,
            self.val_dataset,
            batch_size=self.batch_size,
            augment=augment,
            augment_params=augment_params
        )

        # Initialize plot if enabled
        if plot:
            self.plotter.initialize_plot()

        # Early stopping variables
        best_val_loss = history['best_val_loss']
        patience_counter = 0

        if log_level > 1:
            print(f"Starting training from epoch {start_epoch}")
            print(
                f"Train dataset size: {len(train_loader) * self.batch_size}. Val dataset size: {len(self.val_dataset)}\n"
                f"Device: {self.model.device}. Epochs: {self.epochs}. Batch size: {self.batch_size}")

        # Training loop
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            train_losses = []
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Training phase
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}',
                      leave=False,
                      disable=log_level == 0) as pbar:
                for batch in pbar:
                    loss = self.training_handler.training_step(
                        batch=batch,
                        optimizer=optimizer,
                        clip_grad=True,
                        max_norm=1.0
                    )
                    train_losses.append(loss)

                    if log_level >= 2:
                        pbar.set_postfix({
                            'train_loss': np.mean(train_losses),
                            'lr': current_lr
                        })
                    elif log_level == 1:
                        pbar.set_postfix({'train_loss': np.mean(train_losses)})

            epoch_train_loss = np.mean(train_losses)
            history['train_loss'].append(epoch_train_loss)

            # Validation phase
            if self.val_dataset:
                val_loss = self.training_handler.validation_step(val_loader)
                history['val_loss'].append(val_loss)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping check and checkpoint saving
                is_best = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    history['best_val_loss'] = val_loss
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1

                # Save checkpoint
                self._save_checkpoint(
                    history=history,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=val_loss,
                    is_best=is_best
                )

                if log_level >= 1:
                    print(f'\nEpoch {epoch + 1}/{self.epochs} - '
                          f'Train Loss: {epoch_train_loss:.4f} - '
                          f'Val Loss: {val_loss:.4f}')

                # Update plots if enabled
                if plot:
                    self.plotter.update_plots(history)

                # Check early stopping
                if patience_counter >= self.early_stop_patience:
                    if log_level >= 1:
                        print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
            else:
                if log_level >= 1:
                    print(f'Epoch {epoch + 1}/{self.epochs} - Train Loss: {epoch_train_loss:.4f}')

                # Update plots if enabled (training loss only)
                if plot:
                    self.plotter.update_plots(history)

            # Regular checkpoint saving
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=epoch_train_loss,
                    is_best=False,
                    history=history
                )

        # Close plot if enabled
        if plot:
            self.plotter.close_plot()

        # Load best model if validation was used
        if self.val_dataset:
            self.load_checkpoint('best_model.pth')
            if log_level >= 1:
                print(f'Loaded best model from epoch {history["best_epoch"]} '
                      f'with validation loss: {history["best_val_loss"]:.4f}')

        return history

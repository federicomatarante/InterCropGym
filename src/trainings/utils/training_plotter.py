from matplotlib import pyplot as plt
from sympy.physics.control.control_plots import matplotlib


class TrainingPlotter:
    """
    A class for real-time visualization of training metrics using matplotlib.

    Creates an interactive plot with two subplots showing the evolution of loss values
    and learning rate during model training. The plots are updated in real-time as
    training progresses.

    Examples:
        Basic usage:
        >>> plotter = TrainingPlotter()
        >>> plotter.initialize_plot()
        >>>
        >>> # During training loop
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.55, 0.45, 0.35],
        ...     'learning_rates': [0.001, 0.001, 0.0005]
        ... }
        >>> plotter.update_plots(history)
        >>>
        >>> # After training
        >>> plotter.close_plot()

        Custom training loop:
        >>> plotter = TrainingPlotter()
        >>> plotter.initialize_plot()
        >>>
        >>> training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        >>> for epoch in range(num_epochs):
        ...     # Training code here
        ...     training_history['train_loss'].append(train_loss)
        ...     training_history['val_loss'].append(val_loss)
        ...     training_history['learning_rates'].append(current_lr)
        ...     plotter.update_plots(training_history)
        >>>
        >>> plotter.close_plot()
    """
    def __init__(self):
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def initialize_plot(self):
        """
        Initializes the matplotlib plot with two subplots.

        Creates an interactive figure with two subplots:
        - Left subplot: Training and validation loss curves
        - Right subplot: Learning rate schedule
        """
        matplotlib.use('TkAgg')
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle('Training Progress')

    def update_plots(self, history):
        """
        Updates both plots with the latest training metrics.

        :param history: Dictionary containing training history with keys:
            - 'train_loss': List of training loss values
            - 'val_loss': List of validation loss values (optional)
            - 'learning_rates': List of learning rate values
        """
        if not self.fig:
            return

        # Plot losses
        self.ax1.clear()
        self.ax1.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            self.ax1.plot(history['val_loss'], label='Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_yscale('log')
        self.ax1.legend()
        self.ax1.grid(True)
        self.ax1.set_title('Loss Evolution')

        # Plot learning rate
        self.ax2.clear()
        self.ax2.plot(history['learning_rates'], 'g-')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Learning Rate')
        self.ax2.set_yscale('log')
        self.ax2.grid(True)
        self.ax2.set_title('Learning Rate Schedule')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def close_plot(self):
        """
        Closes the matplotlib plot and disables interactive mode.
        Should be called after training is complete to clean up resources.
        """
        if self.fig:
            plt.ioff()
            plt.close()

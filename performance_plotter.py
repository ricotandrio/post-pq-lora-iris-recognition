from lib import *

class PerformancePlotter:
  def __init__(
      self,
      epochs,
      train_losses,
      validation_losses,
      training_accuracies,
      validation_accuracies,
      cpu_usages,
      ram_usages,
      times
    ):

    self.epochs = epochs
    self.train_losses = train_losses
    self.validation_losses = validation_losses
    self.training_accuracies = training_accuracies
    self.validation_accuracies = validation_accuracies
    self.cpu_usages = cpu_usages
    self.ram_usages = ram_usages
    self.times = times

  def _plot(self, ax, x, y, label, color, marker='o'):
    ax.plot(x, y, label=label, marker=marker, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()

  def plot_performance(self):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    self._plot(axs[0, 0], self.epochs, self.train_losses, "Train Loss", "blue")
    self._plot(axs[0, 0], self.epochs, self.validation_losses, "Validation Loss", "orange")
    self._plot(axs[0, 1], self.epochs, self.training_accuracies, "Training Accuracy", "green")
    self._plot(axs[0, 1], self.epochs, self.validation_accuracies, "Validation Accuracy", "red")
    self._plot(axs[1, 0], self.epochs, self.cpu_usages, "CPU Usage", "purple")
    self._plot(axs[1, 0], self.epochs, self.ram_usages, "RAM Usage", "brown")
    self._plot(axs[1, 1], self.epochs, self.times, "Time", "pink")

    plt.tight_layout()
    plt.show()

from lib import *

class ModelDatasetBuilder:
  def __init__(
      self,
      get_model,
      num_classes=None,
      epoch=50,
      learning_rate=0.001,
      weight_decay=1e-2,
      step_size=10,
      gamma=0.1,
      batch_size=16,
      train_dataset=None,
      validation_dataset=None,
      test_dataset=None,
      quantized=False,
      lora=False
    ):
    self.num_classes = num_classes
    self.num_epochs = epoch
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.step_size = step_size
    self.gamma = gamma
    self.batch_size = batch_size
    self.train_dataset = train_dataset
    self.validation_dataset = validation_dataset
    self.test_dataset = test_dataset

    self.model = get_model()

    print(f"Model trainable parameter: {count_trainable_parameter(model=self.model)}")
    print(f"Model parameter: {count_total_parameter(model=self.model)}")

    if lora:
      apply_lora(self.model)

      print(f"Model trainable parameter (lora): {count_trainable_parameter(model=self.model)}")
      print(f"Model parameter (lora): {count_total_parameter(model=self.model)}")

    if quantized:
      self.device = torch.device("cpu")
    else:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def get_model_parameter_count(self):
    return sum(p.numel() for p in self.model.parameters())

  def train_model(self, output_path='mobilenet_iris.pth'):
    train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    # Add learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,            # Optimizer AdamW kita
      mode='min',           # Karena kita ingin menurunkan val_loss
      factor=0.5,           # Turunkan LR menjadi 50% awal karena kalau 10% berhubung dataset kita kecil kata GPT kurang ok
      patience=2,           # 2 Batas kesalahan berhubung patience kita 6 => Jadi Total toleransi = 1 + cooldown + 1 = 3 epoch gagal nanti baru LR nya diturunin
      threshold=0.001,      # Minimal perbedaan val_loss dianggap signifikan supaya ga stagnan
      threshold_mode='rel',
      cooldown=1,           # Diberi 1 epoch jeda sebelum evaluasi lagi
      min_lr=1e-6,          # Batas learning rate terendah di 1e-6 karena jangan terlalu kecil
      eps=1e-08
    )
    # Init model as training
    self.model = self.model.to(self.device)

    # Run model
    history = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 6

    for epoch in range(self.num_epochs):
      epoch_start_time = time.time()
      running_loss = 0.0
      correct = 0
      total = 0

      self.model.train()
      for i, batch in enumerate(train_loader):
        images, labels = batch

        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Accuracy
        preds = torch.argmax(outputs, dim=1)  # Get predicted class
        correct += (preds == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Total number of samples

      train_loss = running_loss / len(train_loader)
      train_accuracy = 100 * correct / total

      # Validation
      self.model.eval()
      val_loss = 0.0
      val_correct = 0
      val_total = 0

      with torch.no_grad():
        for images, labels in val_loader:
          images, labels = images.to(self.device), labels.to(self.device)
          outputs = self.model(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item()

          _, predicted = torch.max(outputs, 1)  # Get class with highest score
          val_total += labels.size(0)               # Update total number of labels
          val_correct += (predicted == labels).sum().item()  # Update correct predictions

      val_loss /= len(val_loader)
      val_accuracy = 100 * val_correct / val_total

      # Resource monitoring
      cpu_usage = psutil.cpu_percent()
      ram_usage = psutil.virtual_memory().used / 1024**2  # Convert to MB
      epoch_time = time.time() - epoch_start_time

      history.append({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "time": epoch_time
      })

      # Early stopping based on validation loss
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
      else:
        epochs_without_improvement += 1

      print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s, Toleration: {epochs_without_improvement}, Ram Usage: {ram_usage:.2f}, CPU Usage: {cpu_usage:.2f},")

      if scheduler:
        scheduler.step(val_loss)
        print(f"LR: {scheduler.get_last_lr()[0]}")

      if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    torch.save(self.model.state_dict(), output_path)

    performance_plotter = PerformancePlotter(
      epochs=[entry["epoch"] for entry in history],
      train_losses=[entry["train_loss"] for entry in history],
      validation_losses=[entry["validation_loss"] for entry in history],
      training_accuracies=[entry["train_accuracy"] for entry in history],
      validation_accuracies=[entry["validation_accuracy"] for entry in history],
      cpu_usages=[entry["cpu_usage"] for entry in history],
      ram_usages=[entry["ram_usage"] for entry in history],
      times=[entry["time"] for entry in history]
    )

    performance_plotter.plot_performance()

  def inference(self):
    test_data = self.test_dataset
    self.model.eval()

    predictions = {}
    correct = 0
    total = len(test_data)

    with torch.no_grad():
      for i, (image_tensor, label) in enumerate(test_data):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        label = label.to(self.device)

        outputs = self.model(image_tensor)
        pred = torch.argmax(outputs, dim=1).item()

        expected_class = label.item()

        is_correct = pred == expected_class

        if is_correct:
          correct += 1

    accuracy = (correct / total) * 100

    print(f"[!] Accuracy from inference in ModelDatasetBuilder {accuracy}%")

    return accuracy
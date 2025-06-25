from lib import *

class BaseIrisPMDS(Dataset):
  def __init__(self, data, data_dir, transform=None):
    self.data = data
    self.data_dir = data_dir
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_name, label = self.data[idx]
    img_path = os.path.join(self.data_dir, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    label = torch.tensor(label, dtype=torch.long)
    return image, label

class TrainIrisPMDS(BaseIrisPMDS):
  pass

class ValidationIrisPMDS(BaseIrisPMDS):
  pass

class TestIrisPMDS(BaseIrisPMDS):
  pass

class SplitIrisPMDS:
  def __init__(self, data_dir, transform=None, test_size=0.2, val_size=0.5, random_seed=42):
    self.data_dir = data_dir
    self.transform = transform
    self.data = defaultdict(list)

    for file_name in os.listdir(data_dir):
      if file_name.endswith(".jpg"):
        label = int(file_name.split("_")[0]) - 1
        self.data[label].append(file_name)

    self.train_data = []
    self.validation_data = []
    self.test_data = []

    for label, images in self.data.items():
      # First split into temp train+val and test
      train_images, test_images = train_test_split(
          images,
          test_size=test_size,
          random_state=random_seed
      )

      test_images, validation_images = train_test_split(
          test_images,
          test_size=val_size,
          random_state=random_seed
      )

      self.train_data.extend([(img, label) for img in train_images])
      self.validation_data.extend([(img, label) for img in validation_images])
      self.test_data.extend([(img, label) for img in test_images])

    # self.train_data = TRAIN_DATA
    # self.validation_data = VALIDATION_DATA
    # self.test_data = TEST_DATA

    self.train_dataset = TrainIrisPMDS(self.train_data, data_dir, transform)
    self.validation_dataset = ValidationIrisPMDS(self.validation_data, data_dir, transform)
    self.test_dataset = TestIrisPMDS(self.test_data, data_dir, transform)

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    num_pixels = 0

    for img_name, _ in self.train_data:
      img_path = os.path.join(self.data_dir, img_name)
      image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      image = np.array(image, dtype=np.float32) / 255.0

      pixel_sum += np.sum(image)
      pixel_squared_sum += np.sum(image ** 2)
      num_pixels += image.size  # total number of pixels

    self.mean = pixel_sum / num_pixels
    self.std = (pixel_squared_sum / num_pixels - self.mean ** 2) ** 0.5

  def get_train_dataset(self):
    return self.train_dataset

  def get_validation_dataset(self):
    return self.validation_dataset

  def get_test_dataset(self):
    return self.test_dataset

  def set_transform(self, transform):
    self.transform = transform
    self.train_dataset.transform = transform
    self.validation_dataset.transform = transform
    self.test_dataset.transform = transform

  def data_distribution(self):
    train_classes = defaultdict(int)
    validation_classes = defaultdict(int)
    test_classes = defaultdict(int)

    for _, label in self.train_dataset.data:
      train_classes[label] += 1

    for _, label in self.validation_dataset.data:
      validation_classes[label] += 1

    for _, label in self.test_dataset.data:
      test_classes[label] += 1

    print("Train Dataset Class Distribution:")
    for label, count in sorted(train_classes.items()):
      print(f"- Class {label}: {count} images")

    print("\nValidation Dataset Class Distribution:")
    for label, count in sorted(validation_classes.items()):
      print(f"- Class {label}: {count} images")

    print("\nTest Dataset Class Distribution:")
    for label, count in sorted(test_classes.items()):
      print(f"- Class {label}: {count} images")

    print("\n")

  def store_splitted_data(self, output_file="store_splitted_data.py"):
    with open(output_file, "w") as f:
      f.write("# Auto-generated train/test split file\n\n")

      f.write("TRAIN_DATA = [\n")
      for img, label in self.train_data:
        f.write(f"    ('{img}', {label}),\n")
      f.write("]\n\n")

      f.write("VALIDATION_DATA = [\n")
      for img, label in self.validation_data:
        f.write(f"    ('{img}', {label}),\n")
      f.write("]\n\n")

      f.write("TEST_DATA = [\n")
      for img, label in self.test_data:
        f.write(f"    ('{img}', {label}),\n")
      f.write("]\n")

    print(f"Split data saved to {output_file}")

  def get_test_data_loader(self, batch_size):
    return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

  def get_validation_data_loader(self, batch_size):
    return DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

  def get_train_data_loader(self, batch_size):
    return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

split_dataset = SplitIrisPMDS("./InputData", transform=None)

train_dataset = split_dataset.get_train_dataset()
test_dataset = split_dataset.get_test_dataset()

print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")
print(f"Total Dataset Size: {len(train_dataset) + len(test_dataset)}")

print(f"Calculated Mean: {split_dataset.mean:.4f}, Std: {split_dataset.std:.4f}")

split_dataset.data_distribution()
split_dataset.store_splitted_data()

print(f"Total Classes: {len(split_dataset.data)}")
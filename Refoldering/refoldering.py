import os
import shutil

def refoldering_data_cornea_iris_multimodal():
  current_path = os.getcwd() + '/Data_CORNEA_IRIS_Multimodal'

  def remove_cornea_folder():
    for folder in os.listdir(current_path):
      folder_path = os.path.join(current_path, folder)

      if os.path.isdir(folder_path):
        cornea_path = os.path.join(folder_path, "Cornea")
        iris_path = os.path.join(folder_path, "Iris")
        iris_l_path = os.path.join(iris_path, "L")
        iris_r_path = os.path.join(iris_path, "R")

        # Delete "Cornea" folder if it exists
        if os.path.exists(cornea_path):
          shutil.rmtree(cornea_path)
          print(f"Deleted: {cornea_path}")

        # Move Iris/L and Iris/R to the parent folder and remove Iris
        if os.path.exists(iris_l_path) and os.path.exists(iris_r_path):
          shutil.move(iris_l_path, folder_path)
          shutil.move(iris_r_path, folder_path)
          shutil.rmtree(iris_path)  # Remove the empty "Iris" folder
          print(f"Moved: {iris_l_path} and {iris_r_path} to {folder_path}")

  def move_images_to_root_folder():
    if not os.path.exists("./Image"):
      os.mkdir("./Image")

    for folder in os.listdir(current_path):
      folder_path = os.path.join(current_path, folder)

      iris_l_path = os.path.join(folder_path, "L")
      iris_r_path = os.path.join(folder_path, "R")

      for image in os.listdir(iris_l_path):
        image_path = os.path.join(iris_l_path, image)
        new_image_path = os.path.join("./Image", f"{folder}_L_{image}")
        shutil.move(image_path, new_image_path)

      for image in os.listdir(iris_r_path):
        image_path = os.path.join(iris_r_path, image)
        new_image_path = os.path.join("./Image", f"{folder}_R_{image}")
        shutil.move(image_path, new_image_path)

  def remove_empty_folders():
    for folder in os.listdir(current_path):
      folder_path = os.path.join(current_path, folder)

      left_path = os.path.join(folder_path, "L")
      right_path = os.path.join(folder_path, "R")

      if os.path.exists(left_path) and len(os.listdir(left_path)) == 0:
        os.rmdir(left_path)
        print(f"Deleted: {left_path}")

      if os.path.exists(right_path) and len(os.listdir(right_path)) == 0:
        os.rmdir(right_path)
        print(f"Deleted: {right_path}")

      os.rmdir(folder_path)
      print(f"Deleted: {folder_path}")
  remove_cornea_folder()
  move_images_to_root_folder()
  remove_empty_folders()

refoldering_data_cornea_iris_multimodal()
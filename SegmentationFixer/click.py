def draw_box_points(image, name='Image', num_points=4):
  import cv2
  
  def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      points = param['points'] 
      image = param['image']
      points.append((x, y))
      cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
      # Display the updated image
      if len(points) == 2:
        cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)
      cv2.imshow(param['name'], image)

  cv2.namedWindow(name)
  params = {'points': [], 'image': image, 'name': name}
  cv2.setMouseCallback(name, mouse_callback, params)

  print('Please draw center of iris, center of pupil, radius of iris, and radius of pupil on the image')

  cv2.imshow(name, image)

  while len(params['points']) < num_points:
    cv2.waitKey(1)
  
  return params['points']
  
# Main code for click.py
def main():
  import os
  import torch
  import cv2
  from output_coordinates import IMAGE_IRIS
  import pprint
  import shutil
  from folder_path import IMAGES, ONLY_RESULTS, ONLY_FAILED
  
  # Folder isi gambar original
  inputdir = IMAGES
  
  # Folder isi gambar output
  outputdir = ONLY_RESULTS
  
  # Folder isi gambar yang gagal
  faileddir = ONLY_FAILED
  
  os.makedirs(outputdir, exist_ok=True)
  
  # total file in failed
  print(f"Total in input_directory: {sum(1 for f in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, f))) - 1}")

  temp = IMAGE_IRIS
  
  for file in os.listdir(faileddir):
    print(f"Total in failed_output: {sum(1 for f in os.listdir(faileddir) if os.path.isfile(os.path.join(faileddir, f))) - 1}")
    print(f"Total in result_output: {sum(1 for f in os.listdir(outputdir) if os.path.isfile(os.path.join(outputdir, f))) - 1}")

    image_path = os.path.join(inputdir, file)
    print(image_path)
    # WARNING: This condition is failed on specific access of folder, please re-check
    if os.path.isfile(image_path) and os.path.exists(image_path) and image_path.endswith('.jpg'):
      # print(image_path)
      img = cv2.imread(image_path)
      
      points = draw_box_points(img)
      
      # print('Points:', points)
      
      pupil_center, iris_center, pupil_radius_point, iris_radius_point = points

      iris_radius = int(((iris_center[0] - iris_radius_point[0])**2 + (iris_center[1] - iris_radius_point[1])**2)**0.5)
      pupil_radius = int(((pupil_center[0] - pupil_radius_point[0])**2 + (pupil_center[1] - pupil_radius_point[1])**2)**0.5)
      
      # print('Iris Center:', iris_center)
      # print('Pupil Center:', pupil_center)
      # print('Iris Radius:', iris_radius)
      # print('Pupil Radius:', pupil_radius)
      
      temp[file] = (iris_center[0], iris_center[1], iris_radius, pupil_center[0], pupil_center[1], pupil_radius)
      with open('./SegmentationFixer/output_coordinates.py', 'w') as f:
        f.write("IMAGE_IRIS=")
        pprint.pprint(temp, stream=f)
        
      cv2.circle(img, iris_center, iris_radius, (0, 255, 0), 2)
      cv2.circle(img, pupil_center, pupil_radius, (0, 0, 255), 2)
      cv2.imshow('Iris and Pupil', img)
      cv2.waitKey(0)    
      
      cv2.destroyAllWindows()
      
      # move file to results_origin
      shutil.move(os.path.join(faileddir, file), os.path.join(outputdir, file))

# Move missing image
def move_missing_image():
  import os
  import shutil
  from folder_path import ONLY_IMAGE, ONLY_RESULTS, ONLY_FAILED
  
  # Input folder containing images
  inputdir = ONLY_IMAGE
  
  # Output folder for results
  outputdir = ONLY_RESULTS
  
  # Output folder for failed images (will recheck at this folder)
  faileddir = ONLY_FAILED
  
  os.makedirs(outputdir, exist_ok=True)
  
  # Total file in inputdir 
  input_files = {f for f in os.listdir(inputdir) if f.endswith('.jpg')}

  # Total file in outputdir
  result_files = {f for f in os.listdir(outputdir) if f.endswith('.jpg')}

  # Find files that are in inputdir but not in outputdir
  missing_files = input_files - result_files
  
  print("Files in only_img but not in results_origin:")
  for f in sorted(missing_files):
    print(f)
    shutil.move(f"{inputdir}/{f}", f"{faileddir}/{f}")

# Draw Circle based on output_coordinates.py
def draw_circles_based_on_output_coordinates():
  from output_coordinates import IMAGE_IRIS
  from folder_path import ONLY_IMAGE, ONLY_RESULTS
  import os
  import cv2
  
  # Input folder containing images
  inputdir = ONLY_IMAGE
  
  # Output folder for results
  outputdir = ONLY_RESULTS
  
  for file in os.listdir(outputdir):
  
    image_path = os.path.join(inputdir, file)
    if os.path.isfile(image_path) and os.path.exists(image_path) and image_path.endswith('.jpg'):
      print(image_path)
      img = cv2.imread(image_path)
      
      x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil = IMAGE_IRIS[file]

      cv2.circle(img, (x_iris, y_iris), r_iris, (0, 255, 0), 2)
      cv2.circle(img, (x_pupil, y_pupil), r_pupil, (0, 0, 255), 2)
      
      cv2.imshow('Iris and Pupil', img)
      
      cv2.imwrite(os.path.join(outputdir, file), img)

def fill_output_coordinates():
  import os
  import pprint
  
  temp_dict = {}
  
  for file in os.listdir('./SegmentationFixer/only_img'):
    if file.endswith('.jpg'):
      temp_dict[file] = (0, 0, 0, 0, 0, 0)
      
  with open('./SegmentationFixer/output_coordinates.py', 'w') as f:
    f.write("IMAGE_IRIS=")
    pprint.pprint(temp_dict, stream=f)

def move(src: str, dst: str, start_class: int=1, end_class: int=13):
  """
  Move files from src to dst based on their class.
  :param src: Source directory containing the files.
  :param dst: Destination directory where files will be moved.
  :param start_class: Starting class number (inclusive).
  :param end_class: Ending class number (inclusive).
  """
  import os
  import shutil

  # Create destination directory if it doesn't exist
  os.makedirs(dst, exist_ok=True)

  for file in os.listdir(src):
    # Check if the file is a directory
    if os.path.isfile(os.path.join(src, file)) and file.endswith('.jpg'):
      # Extract the class number from
      splitted_name = file.split('_')
      
      class_number = int(splitted_name[0])
      
      if class_number >= start_class and class_number <= end_class:
        shutil.copy(os.path.join(src, file), os.path.join(dst, file))

# Folder "Image" is the folder containing all images from the dataset.
# Folder "only_img" is the folder containing images from a specific class range as per the assignment.
# Folder "results_failed" is for images that failed processing.
# Folder "results_origin" is for successfully processed images and for output.
# This script is designed to help fixing with localizing the iris and pupil in images.  
if __name__ == '__main__':
  ## Uncomment the following line to move images from 'Image' to 'only_img' for classes 1 to 13
  # move('./Image', './only_img', start_class=1, end_class=13)
  
  ## For initialize output_coordinates.py
  # fill_output_coordinates()
  
  ## GUI Mode for fixing coordinates
  ### 1st click is iris_center
  ### 2nd click is pupil_center
  ### 3rd click is radius of iris
  ### 4th click is radius of pupil
  ### After 4 click, type any key on keyword for next
  main()
  
  ## For moving missing image from input to results so it will re-check at next main()
  # move_missing_image()
  
  # For updating image after fixing coordinates
  draw_circles_based_on_output_coordinates()
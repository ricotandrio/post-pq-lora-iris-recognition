# Post Pruning-Quantization with LoRA for Iris Recognition

This repository contains the experimental code and supporting modules from our research on **Optimizing MobileNet-Based Models for Iris Recognition** using **LoRA (Low-Rank Adaptation)**, **Pruning**, and **Quantization**.



## 📁 Project Structure

### `Refoldering/`
Contains code used to restructure the raw dataset into a unified format:  
```

ClassNumber_[L/R]_ID.extension

````
This ensures consistent naming for left/right eye classification.

### `SegmentationFixer/`
Manual correction tools for fixing segmentation results — especially those misidentified by the original Hough Transform-based segmentation pipeline.

### `Static_Data/`
Contains precomputed bounding boxes and iris coordinate data.  
Refer to the comments at the top of each file for source and format details.

### `lib.py`
Centralized import file listing all required libraries and dependencies used throughout the project.

### `requirements.txt`
Python dependencies required to run the project.  
Use the following to install:
```bash
pip install -r requirements.txt
````



## 📓 Notebooks

### `main.ipynb`

The main entry-point notebook, where all core functionalities are structured and organized into reusable modules.


## 📚 Dataset
This project uses the [Dataset](https://www-labs.iro.umontreal.ca/~labimage/IrisCorneaDataset/) from the Université de Montréal.


### `./Experiment`

Contains *`LORA_of_FINAL_MobileNetOptimization.ipynb`*, which is the original experimental notebook developed during research using **Google Colab**.
This version includes exploratory work, trial runs, and freezed results.

You can open the experiment notebook in Colab here:
*https://colab.research.google.com/drive/1nV0YMpPpMvOMRurS1avsyzETin-XIrAW?authuser=0#scrollTo=T9VgoGSNiy2x*


## 🛠 Notes

* Functions from the original Colab notebooks have been modularized and refactored into `.py` files for better readability and reuse.
* All experimentation was conducted on **MobileNetV3** and **MobileNetV4** models for iris recognition, with a focus on mobile optimization.
* Post-training pruning and quantization were applied **after** LoRA fine-tuning for minimal performance degradation.

## 🔧 Contact

For questions or collaboration inquiries, feel free to reach out via `.`
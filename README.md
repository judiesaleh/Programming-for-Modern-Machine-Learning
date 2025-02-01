# Computer Vision 3 - License Plate Recognition

## Project Description

This project is a part of the course "Programming for Modern Machine Learning" at the Ruhr-University Bochum. The goal of the project is to develop a license plate recognition system using computer vision techniques. Multiple approaches are implemented and compared.

## Project Structure

- `notebooks/`: Contains the notebooks used for the project.
- `dataset/`: Contains the dataset and the preprocessed data.
- `models/`: Contains the trained models.
- `src/`: Contains additional source code.


### Dataset: Chinese City Parking Dataset (CCPD)

The full dataset is availabe on https://github.com/detectRecog/CCPD.


The dataset consists of over 250.000 labeled images of chinese license plates. The license plates have white 7 characters on a blue background.
![04468151341-90_84-159 441_570 582-575 571_155 575_149 458_569 454-0_2_8_32_24_33_8-81-234](https://github.com/user-attachments/assets/f040e080-6377-4a36-afa3-6c437028fe6d)

The labels are stored in the `dataset/.../train/file_names.txt` file bzw. `dataset/.../train/labels.csv` file.

### Notebook 01: Data Exploration

This notebook contains the data exploration of the dataset. The dataset is presorted by the labels.
We use 2 different partititions of the 250.000 image dataset:

  1. We take the first 50.000 images with the highest area ratio, meaning the license plates take up the most space in the image. Then we apply 3 additional thresholds to sort out more images. We apply a max. `tilt_degree_deviation = 10`, `brightness_threshold = 70`, `blurriness_threshold = 100`. This way we assue that the image quality of the dataset is good.

    - about 9.500 images in total ( 7.250 for training, 900 for validation, 900 for testing)

  2. We take the first 100.000 images with the highest area ratio. Then we apply 3 additional thresholds to sort out more images. We apply a max. `tilt_degree_deviation = 10`, `brightness_threshold = 50`, `blurriness_threshold = 50`. This way we assue that the image quality of the dataset is still good, but we have a larger dataset.

    - about 37.000 images in total ( 29.600 for training, 3.700 for validation, 3.700 for testing)

Additionally, we create a csv file for each partition, which contains the filename, the area ratio, the tilt degree, the bounding box coordinates, the four vertices locations, the license plate number, the brightness and the blurriness and a `file_names.txt` file, which contains the filenames of the images in the partition.

The `01_data_exploration.ipynb` notebook also contains code to calculate the mean and standard deviation of the area of the license plate of the images, and the mean and standard deviation of the ratio of the license plate (width/height). These values were used in the `02_preprocessing.ipynb` notebook to create the preprocessed dataset.

### Notebook 02: Preprocessing

This notebook contains the preprocessing of the dataset. The preprocessing includes the following steps:

1.  Run each image through the HSV pipeline. The HSV pipeline tries to detect the license plate by using different HSV ranges, the image is run through a max. of 5 different HSV ranges.
  - Try to find fitting contours in the image.
  - If a contour is found, the accuracy of this contour is calculated with the values from the filename (for statistics) and returned.
  - If no contours are found, an accuracy of 0 is returned and the image is run through the next HSV range.

2.  If the accuracy of the detection is 0 after running through all HSV ranges, no contours are found. Those images are not used for training. (about 2% of the images)

3.  If the accuracy of the detection is below 20%, the image is used for training, but the label (filename) is updated. An accuracy of below 20 % means that the license plate is not detected correctly and max. 20% of the license plate area are in the cropped picture.
The filename consists of the license plate number, which is used for training. This number is updated to the encoding for "no character": "O_O_O_O_O_O_O".
(about 7% of the images)

4. The image is then cropped to the license plate (+ padding) and saved in the `dataset/preprocessed/...` folder.

5. The accuracy scores are saved and used for statistics.


The preprocessed dataset is used for the training of the models. For that is was splitted into training, validation and testing. And new csv files were created for each partition, which contains the filename, the area ratio, the tilt degree, the bounding box coordinates, the four vertices locations, the license plate number, the brightness and the blurriness and a `file_names.txt` file, which contains the filenames of the images in the partition. But the area ratio, bounding box coordinates, four vertices locations are not correct anymore.
![03719348659-90_89-156 465_539 578-541 574_173 573_177 452_545 453-0_0_27_19_30_24_29-129-115](https://github.com/user-attachments/assets/784ecb88-bdf8-4f42-acc1-8d85f3eefb3f)


## Additional Information

### Git LFS

Git Large File Storage (LFS) replaces large files with text pointers inside Git, while storing the file contents on a remote server. This is useful for versioning large files, especially binary files like images and datasets.


#### Setup

```bash
sudo apt-get install git-lfs   # Ubuntu/Debian
```

```bash
git lfs install
```

#### Use Git LFS

1. List LFS files without downloading them:
```bash
git lfs ls-files
```

2. Download all LFS files:
```bash
git lfs pull
```

3. Download specific files or patterns:

```bash
# Download specific file
git lfs pull --include="dataset/CCPD2019/specific_image.jpg"
```

```bash
# Download specific pattern
git lfs pull --include="dataset/CCPD2019/*.jpg"
```

```bash
# Download from specific folder
git lfs pull --include="dataset/CCPD2019/*"
```

```bash
# Download only files from latest commit
git lfs pull --recent
```

#### Clone repository without LFS files
```bash
git clone --no-checkout https://github.com/your/repo.git
cd repo
git lfs install
git checkout main
```

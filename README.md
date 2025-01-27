# Computer Vision - License Plate Recognition


## Dataset: Chinese City Parking Dataset (CCPD)

The full dataset is availabe on https://github.com/detectRecog/CCPD.


## Git LFS

Git Large File Storage (LFS) replaces large files with text pointers inside Git, while storing the file contents on a remote server. This is useful for versioning large files, especially binary files like images and datasets.


### Setup

```bash
sudo apt-get install git-lfs   # Ubuntu/Debian
```

```bash
git lfs install
```

### Use Git LFS

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

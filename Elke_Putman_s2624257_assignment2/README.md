# IPCV Assignment 2

This project is **Assignment 2** from the *Image Processing & Computer Vision* course.  
All functions are implemented in the `VideoProcessor` class (defined in `Processor_class.py`).  
The script takes an input video and produces a processed output video.

---

## Requirements

- **Python:** 3.8 or higher  
- **Required packages:**
  - `opencv-python`
  - `numpy`
  - `matplotlib`

## Usage
```bash
python main.py -i <input_video> -o <output_video>
```

## Example
Suppose you have a file called `example.mp4` you can run the following:
```bash
python main.py -i example.mp4 -o example_processed.mp4
```
## File structure
Below is the file structure shown required to run the `main.py` file

```bash
IPCV_Assignment_2/
├── main.py
├── Processor_class.py
├── Input_video.mp4
│
├── Templates_diver/
│   └── (template images for diver detection)
│
├── Templates_laptop/
│   └── (template images for laptop detection)
│
├── Templates_logo/
│   └── (template images for logo detection)
│
├── Templates_numbers/
│   └── (template images for number detection)
│
├── Templates_photo/
│   └── (template images for photo detection)
│
└── README.md




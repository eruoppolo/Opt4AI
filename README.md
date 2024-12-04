# Optimization For Artificial Intelligence - UniTS
## Exam project - Fall 2024 - Emanuele Ruoppolo

### Image generation using Genetic Algorithms
#### **1. Problem Definition**
The goal of this project is to evolve a low-resolution, pixel-art style image (e.g. 32x32, 64x64 o5 128x128 grid) that approximates a target image. The **target image** is a high-resolution image (e.g. a photograph or artwork) and the **output** is pixel-art image that captures the essence of the target image but is constrained by the low-resolution.
#### **2. Preprocessing**
In order to approximate a high resolution image with a lower resolution one we will define an appropriate fitness function. To use the fitness function properly we need to resize the target image into a lower-resolution version which serves as a guide for the generation. The image pixels values are normalized to 8bit integers (0-255) to ensure a common and proper format for the generation.

These operations are performed using the `Image` module from `Pillow` package.

``` python
from PIL import Image

def load_image(image_path:str , target_resolution:tuple) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_resolution, Image.Resampling.LANCZOS)
    return np.array(img)

```


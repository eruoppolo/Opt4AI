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

---

### **3. Initial Population Generation**
To generate an initial **random population** of individuals (images), we opted for a pixel based strategy. An individual is composed of randomly chosen pixel colors on a coloured background. This choice was preferred to a completely noised image since usually images can have a dominant color and this strategy could help in finding it a-priori without waiting it emerging from noise.
Regarding the **image format:** we decided to work squared images for simplicity. So each individual is a 3D array of size $n\times n$ where each cell holds a color divided in the three RGB channels.

### **4. Crossover and Mutations**
To enhance diversity we implemented six different types of recombination between the population individuals and one mutation strategy:

- **Crossover Mechanism:** Two parent individuals are comined to create offspring. The crossover can happen at a random point in the chromosome, swapping parts of the image between two parents.
  - **Single-Point Crossover:** The parents genome is splitted in half and  the left and right parts are exchanged between two parents:
    - **Horizontal**: the images are splitted in two horizontal bands and combined
    - **Vertical**: the images are splitted in two vertical bands and combined
    - **Diagonal**: the images are splitted in two parts on a random diagonal and combined
  - **Multi-Point Crossover:** The parents genome is splitted into multiple sections and randomly combined:
    - **Horizontal**: the images are splitted into horizontal bands, randomly selected, and combined
    - **Vertical**: the images are splitted into vertical bands, randomly selected, and combined
  - **Blending crossover**: The parents genome is summed by weighted on a random coefficient $\alpha\in[0,1]\to G=\alpha G_A+(1-\alpha)G_B$. The images are then blended on a random opacity parameter.
- **Mutations**: After each crossover, to maintain diversity and avoid premature convergence, the offspring genome gets some random mutations. A mutation conists in a random change in the color of a pixel. The amount of mutations is ruled by a **mutation rate** hyperparameter, which exponentially decreases over time: 

$$ \text{mutation rate}(t)=\text{mutation rate}(0)\cdot\exp(-\beta\cdot t)$$

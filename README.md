# Image-Super-Resolution-CNN
This is a Machine Learning mini-project that uses a Convolutional Neural Network (SRCNN) to improve the quality of low-resolution images.

The project compares the results of SRCNN with bicubic interpolation to see which method produces a better image.

## 1. Project Structure

```text
Image-Super-Resolution-CNN/
├── Data/
│   ├── train/              # Images used for training
│   └── test/               # Images used for testing
├── Results/
│   ├── srcnn.pth           # Trained model
│   └── comparison.png      # Comparison of the results
├── src/
│   ├── model.py            # Defines the SRCNN model
│   ├── train.py            # Trains the model
│   └── demo.py             # Tests the model and shows results
├── Requirements.txt
├── README.md
└── .gitignore
```

## 2. Requirements

The project uses Python and the following libraries:

- PyTorch
- Torchvision
- NumPy
- OpenCV
- Pillow
- Matplotlib
- scikit-image

A GPU is recommended for faster training, but the demo can also run on a CPU.

## 3. Setup Instructions

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Nirav6769/Image-Super-Resolution-CNN.git
cd Image-Super-Resolution-CNN
```

### Step 2: Install the required libraries

Run this command from the project folder:

```bash
python -m pip install -r Requirements.txt
```

### Step 3: Add the images

Put the training images inside `Data/train/` and the testing images inside `Data/test/`.

For this experiment, I used 20 training images and 5 test images.

The code creates low-resolution images by reducing their size and then enlarging them using bicubic interpolation.

## 4. Training the Model

To train the model, run:

```bash
python src/train.py
```

The script loads the training images, prepares them, and trains the SRCNN model.

It uses Mean Squared Error (MSE) to calculate the difference between the predicted image and the original image. The model updates its weights during training to reduce this error.

After training, the model is saved as:

```text
Results/srcnn.pth
```

I trained the model for 15 epochs using a Tesla T4 GPU on Google Colab.

**Note:** Training is not required every time you run the demo. The repository already contains the trained model weights.

## 5. Running the Demo

To test the trained model, run:

```bash
python src/demo.py
```

The script loads the trained model and selects an image from `Data/test/`.

It then:
1. Creates a bicubic-upsampled version of the image.
2. Uses SRCNN to reconstruct the image.
3. Calculates the PSNR for both results.
4. Displays the bicubic result, SRCNN result, and original image side by side.

The comparison is also saved as:

```text
Results/comparison.png
```

## 6. Results

The demo produced the following PSNR values for the test image:

| Method | PSNR |
|---|---:|
| Bicubic interpolation | 17.55 dB |
| SRCNN | 17.60 dB |

SRCNN performed slightly better than bicubic interpolation in this test, with an improvement of 0.05 dB.

The improvement is small, and the model was trained on a limited number of images. More training data and further experiments would be needed to improve the results.

The visual comparison can be found in `Results/comparison.png`.

## 7. SRCNN Architecture

The model has three convolutional layers:

- **First layer:** Extracts features from the input image using a 9 × 9 filter.
- **Second layer:** Learns relationships between the extracted features using a 5 × 5 filter.
- **Third layer:** Reconstructs the output image using another 5 × 5 filter.

ReLU activation is used after the first two layers.

The model works on the brightness (Y) channel of the image, while the colour channels are kept for the final reconstruction.

## 8. Evaluation

The project uses Peak Signal-to-Noise Ratio (PSNR) to compare the reconstructed images with the original image.

A higher PSNR generally means that the reconstructed image is closer to the original in terms of pixel values.

## 9. Limitations and Future Improvements

The project currently uses a small dataset of 20 training images and 5 test images. The model was also trained for only 15 epochs.

In the future, the project could be improved by:
- Using a larger dataset.
- Training for more epochs.
- Improving the training process.
- Testing on more images.
- Comparing additional image-quality metrics.

## 10. Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- Matplotlib
- scikit-image
- Google Colab

## 11. Author

**Nirav Choudhary**  
**Nimit S Jain**
PES University  
Computer Science and Engineering


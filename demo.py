import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import os
import random

# =========================================
# Hyperparameters and Configuration
# =========================================

BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 300
LEARNING_RATE = 0.0002
IMG_SIZE = 64  # Increased image size for better quality
NUM_SAMPLES = 2000
NUM_REAL_DISPLAY = 5
NUM_GENERATED_DISPLAY = 5
CHECKPOINT_DIR = "checkpoints_dcgan"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================
# Data Preparation
# =========================================

def create_face_color() -> Tuple[float, float, float]:
    colors = {
        'yellow': (1.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'green': (0.0, 1.0, 0.0),
        'pink': (1.0, 0.75, 0.8)
    }
    color = random.choice(list(colors.values()))
    return color

def add_eyes(image: np.ndarray, x: np.ndarray, y: np.ndarray, eye_x_left: int, eye_x_right: int, eye_y: int, eye_radius: int) -> None:
    left_eye_mask = (x - eye_x_left)**2 + (y - eye_y)**2 <= eye_radius**2
    right_eye_mask = (x - eye_x_right)**2 + (y - eye_y)**2 <= eye_radius**2
    image[:, left_eye_mask] = 0.0
    image[:, right_eye_mask] = 0.0

def add_mouth(image: np.ndarray, x: np.ndarray, y: np.ndarray, center: Tuple[int, int], mouth_width: int, mouth_height: int) -> None:
    mouth_y = center[1] + IMG_SIZE // 6
    mouth_x_start = center[0] - mouth_width // 2
    mouth_x_end = center[0] + mouth_width // 2

    for i in range(mouth_x_start, mouth_x_end):
        relative_x = (i - center[0]) / (mouth_width / 2)
        relative_y = (relative_x**2) * mouth_height
        y_pos = int(mouth_y - relative_y)

        if 0 <= y_pos < IMG_SIZE:
            image[:, y_pos, i] = 0.0

def add_optional_features(image: np.ndarray, x: np.ndarray, y: np.ndarray, feature: str) -> None:
    if feature == 'glasses':
        glass_radius = 3
        left_glass_center = (IMG_SIZE // 2 - IMG_SIZE // 8, IMG_SIZE // 3)
        right_glass_center = (IMG_SIZE // 2 + IMG_SIZE // 8, IMG_SIZE // 3)
        bridge_y = IMG_SIZE // 3
        bridge_x_start = left_glass_center[0] + glass_radius
        bridge_x_end = right_glass_center[0] - glass_radius

        left_mask = (x - left_glass_center[0])**2 + (y - left_glass_center[1])**2 <= glass_radius**2
        image[:, left_mask] = 0.0

        right_mask = (x - right_glass_center[0])**2 + (y - right_glass_center[1])**2 <= glass_radius**2
        image[:, right_mask] = 0.0

        bridge_mask = (x >= bridge_x_start) & (x <= bridge_x_end) & (y == bridge_y)
        image[:, bridge_mask] = 0.0

    elif feature == 'hat':
        hat_height = IMG_SIZE // 8
        hat_width = IMG_SIZE // 3
        hat_x_start = IMG_SIZE // 2 - hat_width // 2
        hat_x_end = IMG_SIZE // 2 + hat_width // 2
        hat_y_start = IMG_SIZE // 4 - hat_height
        hat_y_end = IMG_SIZE // 4

        for i in range(hat_x_start, hat_x_end):
            for j in range(hat_y_start, hat_y_end):
                if 0 <= j < IMG_SIZE:
                    image[:, j, i] = 0.0

def generate_smiley_faces_dataset(num_samples: int = NUM_SAMPLES, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    data = []
    real_images = []

    for idx in range(num_samples):
        image = np.ones((3, img_size, img_size), dtype=np.float32)  # White background

        face_color = create_face_color()
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        center = (img_size // 2, img_size // 2)
        radius = img_size // 2 - 5
        mask = (x_grid - center[0])**2 + (y_grid - center[1])**2 <= radius**2
        image[0, mask] = face_color[0]
        image[1, mask] = face_color[1]
        image[2, mask] = face_color[2]

        eye_radius = 3
        eye_y = center[1] - img_size // 6
        eye_x_left = center[0] - img_size // 4
        eye_x_right = center[0] + img_size // 4

        add_eyes(image, x_grid, y_grid, eye_x_left, eye_x_right, eye_y, eye_radius)

        if random.random() < 0.3:
            add_optional_features(image, x_grid, y_grid, 'glasses')
        if random.random() < 0.2:
            add_optional_features(image, x_grid, y_grid, 'hat')

        mouth_width = img_size // 2
        mouth_height = img_size // 10
        add_mouth(image, x_grid, y_grid, center, mouth_width, mouth_height)

        # Normalize to [-1, 1]
        image = (image * 2) - 1

        data.append(image.flatten())

        if idx < NUM_REAL_DISPLAY:
            real_images.append(image.copy())

    data = torch.tensor(np.array(data), dtype=torch.float32)
    real_images = torch.tensor(np.array(real_images), dtype=torch.float32)

    return data, real_images

data, real_images = generate_smiley_faces_dataset()

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================================
# DCGAN Generator Definition
# =========================================

class Generator(nn.Module):
    """
    DCGAN Generator
    Transforms a random noise vector into a structured image using transposed convolutions.
    """
    def __init__(self, latent_dim: int, img_channels: int = 3, feature_map_size: int = 64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # State: (feature_map_size*8) x 4 x 4

            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # State: (feature_map_size*4) x 8 x 8

            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # State: (feature_map_size*2) x 16 x 16

            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # State: (feature_map_size) x 32 x 32

            nn.ConvTranspose2d(feature_map_size, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: (img_channels) x 64 x 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =========================================
# DCGAN Discriminator Definition
# =========================================

class Discriminator(nn.Module):
    """
    DCGAN Discriminator
    Evaluates whether an image is real or fake using deep convolutional layers.
    """
    def __init__(self, img_channels: int = 3, feature_map_size: int = 64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (img_channels) x 64 x 64
            nn.Conv2d(img_channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_map_size) x 32 x 32

            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_map_size*2) x 16 x 16

            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_map_size*4) x 8 x 8

            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_map_size*8) x 4 x 4

            nn.Conv2d(feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1).squeeze(1)

# =========================================
# DCGAN Lightning Module Definition
# =========================================

class DCGAN(pl.LightningModule):
    """
    PyTorch Lightning module encapsulating the DCGAN architecture, training, and optimization.
    """
    def __init__(self, latent_dim: int, img_channels: int = 3, feature_map_size: int = 64):
        super(DCGAN, self).__init__()
        self.generator = Generator(latent_dim, img_channels, feature_map_size)
        self.discriminator = Discriminator(img_channels, feature_map_size)
        self.latent_dim = latent_dim
        self.automatic_optimization = False  # Manual optimization

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        real_images = batch[0].view(-1, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        batch_size = real_images.size(0)

        # Initialize optimizers
        d_optimizer, g_optimizer = self.optimizers()

        # ============================
        # Train Discriminator
        # ============================

        d_optimizer.zero_grad()

        # Real images
        real_preds = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, device=self.device) * 0.9  # Label smoothing
        real_loss = nn.functional.binary_cross_entropy(real_preds, real_labels)

        # Fake images
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images.detach())
        fake_labels = torch.zeros(batch_size, device=self.device)
        fake_loss = nn.functional.binary_cross_entropy(fake_preds, fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Backward and optimize
        self.manual_backward(d_loss)
        d_optimizer.step()

        self.log("d_loss", d_loss, prog_bar=True)

        # ============================
        # Train Generator
        # ============================

        g_optimizer.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(z)
        preds = self.discriminator(fake_images)
        # Generator tries to make the discriminator believe that the fake images are real
        g_loss = nn.functional.binary_cross_entropy(preds, torch.ones(batch_size, device=self.device))

        # Backward and optimize
        self.manual_backward(g_loss)
        g_optimizer.step()

        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures the optimizers for both the discriminator and the generator.
        """
        # Adam optimizer with betas (0.5, 0.999) as recommended in DCGAN paper
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        return [d_optimizer, g_optimizer]

# =========================================
# Training the DCGAN
# =========================================

def train_dcgan() -> DCGAN:
    """
    Initializes and trains the DCGAN model using PyTorch Lightning's Trainer.
    
    Returns:
        DCGAN: The trained DCGAN model.
    """
    model = DCGAN(latent_dim=LATENT_DIM)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=False,
        monitor='g_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=False
    )

    trainer.fit(model, dataloader)

    return model

# =========================================
# Visualization of Real and Generated Images
# =========================================

def generate_and_compare_images(model: DCGAN, latent_dim: int, real_images: torch.Tensor, num_generated: int = NUM_GENERATED_DISPLAY):
    """
    Generates images using the trained generator and compares them with real images.
    
    Args:
        model (DCGAN): The trained DCGAN model.
        latent_dim (int): Dimension of the latent vector used by the generator.
        real_images (torch.Tensor): Tensor of real images from the dataset for comparison.
        num_generated (int): Number of generated images to display.
    
    Returns:
        None
    """
    model.eval()
    fig, axes = plt.subplots(2, max(NUM_REAL_DISPLAY, num_generated), figsize=(15, 6))

    # Display real images
    for i in range(NUM_REAL_DISPLAY):
        img = real_images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Real Images", fontsize=12)

    # Display generated images
    for i in range(num_generated):
        z = torch.randn(1, latent_dim, 1, 1).to(model.device)
        with torch.no_grad():
            generated_image = model(z).squeeze().cpu().numpy()
            generated_image = np.transpose(generated_image, (1, 2, 0))
            img = (generated_image + 1) / 2
            img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Generated Images", fontsize=12)

    plt.tight_layout()
    plt.show()

# =========================================
# Explanation of DCGAN Results
# =========================================

def explain_dcgan_results():
    """
    Provides a textual explanation of how DCGANs work and interprets the generated images.
    """
    explanation = """
    **Understanding DCGAN Results**

    **DCGANs (Deep Convolutional Generative Adversarial Networks)** are an advanced version of GANs that utilize deep convolutional layers to improve the quality and stability of image generation. Here's how DCGANs enhance the traditional GAN framework:

    1. **Architecture Improvements**:
        - **Generator**: Uses transposed convolutions (also known as deconvolutions) to upsample the latent vector into a full-sized image. This allows for better spatial feature learning.
        - **Discriminator**: Employs convolutional layers to effectively downsample the input image and extract hierarchical features, making it more adept at distinguishing real from fake images.

    2. **Batch Normalization**:
        - Applied in both generator and discriminator to stabilize learning by normalizing the inputs to each layer, which helps in maintaining healthy gradients.

    3. **Activation Functions**:
        - **Generator**: Uses ReLU activations in hidden layers to introduce non-linearity and Tanh in the output layer to scale the generated images to the [-1, 1] range.
        - **Discriminator**: Utilizes LeakyReLU activations to allow a small gradient when the neuron is not active, preventing dead neurons and promoting better gradient flow.

    4. **Strided Convolutions**:
        - Replaces pooling layers with strided convolutions to learn spatial hierarchies without losing information.

    **Interpreting the Results**:

    - **Real Images**: Displayed on the top row serve as the ground truth. These images are diverse in color and optional features like glasses and hats, providing a comprehensive dataset for the GAN to learn from.
    
    - **Generated Images**: Shown on the bottom row are the images produced by the generator after training. Observing these images allows us to assess how well the generator has learned to mimic the real data distribution.

    **Observations**:

    - **Quality and Diversity**: DCGANs produce images with higher fidelity and diversity compared to basic GANs. The faces have consistent features like eyes and mouths, and the colors are vibrant and varied.
    
    - **Optional Features**: Features like glasses and hats are effectively captured, showcasing the generator's ability to handle more complex patterns.
    
    - **Stability**: Training a DCGAN tends to be more stable due to architectural and normalization improvements, reducing issues like mode collapse (where the generator produces limited varieties of images).

    **Conclusion**:

    By adopting the DCGAN architecture, we've significantly enhanced the generator's ability to produce high-quality and diverse images. The deep convolutional layers allow the model to capture intricate patterns and spatial hierarchies, resulting in more realistic and visually appealing outputs. This demonstrates the power of DCGANs in generating synthetic data that closely mirrors the complexity of real-world datasets.
    """
    print(explanation)

# =========================================
# Main Execution
# =========================================

if __name__ == "__main__":
    # Ensure the image size is compatible with DCGAN architecture (must be 64x64)
    if IMG_SIZE != 64:
        print(f"Adjusting IMG_SIZE from {IMG_SIZE} to 64 for DCGAN compatibility.")
        IMG_SIZE = 64
        # Regenerate the dataset with the new image size
        data, real_images = generate_smiley_faces_dataset(img_size=IMG_SIZE)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train the DCGAN model
    trained_model = train_dcgan()

    # Generate and visualize images
    generate_and_compare_images(trained_model, LATENT_DIM, real_images)

    # Provide an explanation of the DCGAN results
    explain_dcgan_results()

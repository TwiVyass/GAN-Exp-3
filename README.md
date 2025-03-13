# Deep Convolutional GAN (DCGAN) for Face Generation

This repository contains the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **CelebA dataset** to generate realistic face images.

## **Dataset Preprocessing**
### **1. Downloading the Dataset**
- The model uses the **CelebA dataset**.
- Ensure the dataset is placed in the correct directory: `celeba/`.
- If using Kaggle, you can download the dataset via:
  ```python
  import kagglehub
  from kagglehub import KaggleDatasetAdapter
  KaggleDatasetAdapter("jessicali9530/celeba-dataset").download()
  ```

### **2. Preprocessing Steps**
- Resize all images to **64x64** pixels.
- Convert images to tensors.
- Normalize pixel values to **[-1, 1]**.
  
```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

- Load dataset using **torchvision**:
  ```python
  from torchvision.datasets import ImageFolder
  from torch.utils.data import DataLoader

  dataset = ImageFolder(root="celeba", transform=transform)
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
  ```

## **Model Architecture**
The DCGAN consists of two main components:

### **1. Generator**
- Uses transposed convolution layers to upsample noise vectors into realistic images.
- Uses **BatchNorm** and **ReLU activations** for stability.

### **2. Discriminator**
- Uses convolution layers to classify real vs. fake images.
- Uses **LeakyReLU activations** and **Sigmoid activation** for final classification.

## **How to Train the Model**

### **1. Set Up the Training Environment**
- Ensure you have **PyTorch, torchvision, and matplotlib** installed:
  ```bash
  pip install torch torchvision matplotlib
  ```

### **2. Define Model and Optimizer**
- Initialize models and move to GPU if available:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  netG = Generator(nz=100, ngf=64, nc=3).to(device)
  netD = Discriminator(nc=3, ndf=64).to(device)
  ```
- Define the **loss function (BCELoss)** and optimizers:
  ```python
  criterion = nn.BCELoss()
  optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
  ```

### **3. Training Loop**
- The training loop includes:
  - Updating **Discriminator** with real and generated images.
  - Updating **Generator** based on Discriminator feedback.

```python
num_epochs = 50

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Discriminator
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)

        output_real = netD(real_images).view(-1, 1)
        loss_real = criterion(output_real, labels_real)
        loss_real.backward()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, labels_fake)
        loss_fake.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        output_fake = netD(fake_images).view(-1, 1)
        loss_G = criterion(output_fake, labels_real)
        loss_G.backward()
        optimizerG.step()

        # Print progress
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} | D Loss: {loss_real+loss_fake:.4f} | G Loss: {loss_G:.4f}")
```

### **4. Save Generated Images**
To visualize progress:
```python
with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()
    show_images(fake_images)
```

## **Testing the Model**
- After training, generate new images using:
  ```python
  noise = torch.randn(16, 100, 1, 1, device=device)
  with torch.no_grad():
      generated_images = netG(noise).detach().cpu()
  show_images(generated_images)
  ```

## **Expected Outputs**
- Initially, generated images appear as random noise.
- As training progresses, the model starts generating **more structured and realistic human faces**.
- **Discriminator Loss (D Loss)** should stabilize near **0.5**.
- **Generator Loss (G Loss)** should be relatively low (indicating improvement in generating realistic images).

### **Example Progression:**
- **Epoch 1**: Noisy, unstructured images.
- **Epoch 10-20**: Basic facial features start appearing.
- **Epoch 30+**: More realistic faces emerge.

## **Repository Structure**
```
GAN-Exp-3/
│── celeba/                  # Dataset directory
│── Generator.py             # Generator model
│── Discriminator.py         # Discriminator model
│── train.py                 # Training script
│── utils.py                 # Helper functions
│── README.md                # This file
│── GAN Lab_Experiment 3.ipynb  # Jupyter Notebook
```

---
This repository provides a fully functional **DCGAN** implementation for generating human faces. Feel free to improve and experiment with the architecture! 🚀


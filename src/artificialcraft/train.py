import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Hyperparameters
image_size = 64
batch_size = 128
nz = 100  # Latent vector size
num_epochs = 50
lr = 0.0002
beta1 = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

datadir = os.path.join(os.getcwd(), 'src', 'artificialcraft', 'data', 'dog')
dataset = ImageFolder(datadir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# Init
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# Labels
real_label = 1.
fake_label = 0.

# Fixed noise for sampling
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Load previously saved models (if they exist)
if os.path.exists('generator.pth') and os.path.exists('discriminator.pth'):
    G.load_state_dict(torch.load('generator.pth'))
    D.load_state_dict(torch.load('discriminator.pth'))
    print("Resuming training from saved weights.")

# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(tqdm(dataloader)):
        # === Train Discriminator ===
        D.zero_grad()
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = D(real_imgs)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        # Fake images
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_imgs = G(noise)
        label.fill_(fake_label)
        output = D(fake_imgs.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()
        lossD = lossD_real + lossD_fake

        # === Train Generator ===
        G.zero_grad()
        label.fill_(real_label)
        output = D(fake_imgs)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

    # Save sample images and models after each epoch
    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
    grid = make_grid(fake, padding=2, normalize=True)
    save_image(grid, f'output/fake_epoch_{epoch + 1:03d}.png')

    # Save models after each epoch
    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss_D: {lossD:.4f} Loss_G: {lossG:.4f}")

print("Training finished.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

# Assignment 3: GAN Trainer 

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Implements the GAN training loop for MNIST. 
    """
    generator, discriminator = model
    generator.to(device)
    discriminator.to(device)
    
    opt_g, opt_d = optimizer

    generator.train()
    discriminator.train()
    
    print(f"Starting GAN training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        for i, (real_images, _) in enumerate(data_loader):
            
            # Prepare Data
            real_images = real_images.to(device)
            # Scale real MNIST images to [-1, 1]
            real_images = (real_images * 2) - 1
            
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            
            # 3. Train the Discriminator 
            opt_d.zero_grad()
            
            # 3a. Train with real images
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, real_labels)
            loss_d_real.backward()

            # 3b. Train with fake images
            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            loss_d_fake.backward()
            
            opt_d.step()
            loss_d_total = loss_d_real + loss_d_fake
            
            # 4. Train the Generator
            opt_g.zero_grad()
            
            output_g = discriminator(fake_images)
            loss_g = criterion(output_g, real_labels)
            loss_g.backward()
            
            opt_g.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], "
                      f"Loss_D: {loss_d_total.item():.4f}, "
                      f"Loss_G: {loss_g.item():.4f}")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    print("GAN Training Finished.")
    return generator, discriminator

# Assignment 4: EBM Trainer & Helpers 

class Buffer:
    """ Replay buffer for EBM negative samples """
    def __init__(self, buffer_size=10000, buffer_data=None):
        if buffer_data is None:
            self.examples = []
        else:
            self.examples = buffer_data
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.examples)

    def push(self, new_examples):
        new_examples = new_examples.detach().cpu()
        for el in new_examples:
            if len(self.examples) < self.buffer_size:
                self.examples.append(el)
            else:
                # Replace a random old sample
                idx = torch.randint(0, self.buffer_size, (1,)).item()
                self.examples[idx] = el

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.examples), (batch_size,))
        return torch.stack([self.examples[i] for i in indices], dim=0)

def generate_negative_samples(model, buffer, batch_size, steps, step_size, noise, device, img_channels=3, img_size=32):
    """ 
    Generates negative samples using Langevin Dynamics (MCMC).
    This function performs gradient descent on the *image* to find low-energy regions.
    """
    # Initialize from buffer or random noise
    if len(buffer) > 0 and torch.rand(1) < 0.95:
        x_k = buffer.sample(batch_size).to(device)
    else:
        # Start from random noise in range [-1, 1]
        x_k = torch.rand(batch_size, img_channels, img_size, img_size, device=device) * 2 - 1 
    
    x_k.requires_grad = True # We need gradients w.r.t. the input
    
    for _ in range(steps):
        # Add noise
        z = torch.randn_like(x_k) * noise
        x_k.data.add_(z)
        
        # Get energy and gradient w.r.t. image
        model.zero_grad()
        energy = model(x_k).sum()
        energy.backward() # Calculates x_k.grad
        
        # Update image (descent)
        x_k.data.add_(-step_size * x_k.grad.data)
        
        # Clamp to valid image range [-1, 1]
        x_k.data.clamp_(-1, 1)
        
        x_k.grad.zero_() # Avoid gradient accumulation on the image

    return x_k.detach()

def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, 
              mcmc_steps=60, mcmc_step_size=10, mcmc_noise=0.005,
              samples_path="ebm_samples"):
    """
    Implements the EBM training loop for CIFAR-10.
    """
    model.to(device)
    model.train()
    
    # Create directory for saving samples
    os.makedirs(samples_path, exist_ok=True)
    
    # Initialize replay buffer
    buffer = Buffer(buffer_size=10000)
    
    print(f"Starting EBM training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        real_energy_avg = 0
        fake_energy_avg = 0
        
        for batch_idx, (real_imgs, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
            
            real_imgs = real_imgs.to(device)
            # Normalize CIFAR-10 images to [-1, 1] (assuming ToTensor() scaled to [0,1])
            real_imgs = (real_imgs * 2) - 1
            
            # 1. Generate Negative Samples
            fake_imgs = generate_negative_samples(
                model, buffer, real_imgs.shape[0], 
                mcmc_steps, mcmc_step_size, mcmc_noise, device
            )
            
            # Add new fakes to buffer
            buffer.push(fake_imgs)

            # 2. Calculate Loss (Contrastive Divergence)
            real_out = model(real_imgs)
            fake_out = model(fake_imgs)
            
            # Loss: Push real energy down, push fake energy up
            loss = real_out.mean() - fake_out.mean()
            train_loss += loss.item()
            real_energy_avg += real_out.mean().item()
            fake_energy_avg += fake_out.mean().item()

            # 3. Update Model Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time
        avg_loss = train_loss / len(data_loader)
        avg_real_e = real_energy_avg / len(data_loader)
        avg_fake_e = fake_energy_avg / len(data_loader)
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
              f"Loss: {avg_loss:.4f} | "
              f"E_real: {avg_real_e:.4f} | "
              f"E_fake: {avg_fake_e:.4f}")
        
        # Save a sample of generated images (denormalized to [0,1])
        sample_file = os.path.join(samples_path, f"ebm_samples_epoch_{epoch+1:02d}.png")
        save_image((fake_imgs.data + 1) / 2, sample_file)

    print("EBM Training Finished.")
    return model


# Assignment 4: Diffusion Trainer & Helpers

@torch.no_grad()
def get_normalizer(data_loader, device):
    """ Computes mean and std of the dataset for normalization. """
    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_samples = 0
    
    print("Calculating dataset mean and std...")
    for images, _ in tqdm(data_loader, desc="Calculating Stats"):
        images = images.to(device)
        batch_size = images.shape[0]
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size
        total_samples += batch_size
        
    mean /= total_samples
    std /= total_samples
    
    print(f"Calculated Stats -> Mean: {mean}, Std: {std}")
    return mean.reshape(1, 3, 1, 1), std.reshape(1, 3, 1, 1)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """ Saves a training checkpoint for the Diffusion model. """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.network.state_dict(),
        'ema_model_state_dict': model.ema_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'normalizer_mean': model.normalizer_mean,
        'normalizer_std': model.normalizer_std,
    }, path)
    print(f"Checkpoint saved to {path}")

def train_diffusion(model, train_loader, test_loader, optimizer, device='cpu', epochs=10, 
                    save_path="checkpoints", samples_path="diffusion_samples"):
    """
    Implements the Diffusion Model training loop for CIFAR-10.
    """
    model.to(device)
    loss_fn = F.l1_loss # L1 loss (Mean Absolute Error)
    
    # Create directories for saving models and samples
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(samples_path, exist_ok=True)

    # Calculate and set normalizer stats from the training data
    mean, std = get_normalizer(train_loader, device)
    model.set_normalizer(mean, std)
    
    print(f"Starting Diffusion training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for batch_idx, (real_imgs, _) in enumerate(progress_bar):
            
            real_imgs = real_imgs.to(device)
            
            # 1. Normalize images to [-1, 1] using dataset stats
            normalized_imgs = model.normalize(real_imgs)
            
            # 2. Sample random timesteps
            t = torch.randint(0, model.n_steps, (real_imgs.shape[0],), device=device)
            
            # 3. Add noise to images
            noisy_imgs, noise = model.add_noise(normalized_imgs, t)
            
            # 4. Predict noise with the UNet
            predicted_noise = model.network(noisy_imgs, t)
            
            # 5. Calculate loss
            loss = loss_fn(predicted_noise, noise)
            
            # 6. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 7. Update EMA model
            model.update_ema()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation Step  
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for real_imgs, _ in tqdm(test_loader, desc=f"Epoch {epoch+1} Val"):
                real_imgs = real_imgs.to(device)
                normalized_imgs = model.normalize(real_imgs)
                t = torch.randint(0, model.n_steps, (real_imgs.shape[0],), device=device)
                noisy_imgs, noise = model.add_noise(normalized_imgs, t)
                predicted_noise = model.network(noisy_imgs, t) # Use non-EMA for val loss
                loss = loss_fn(predicted_noise, noise)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_file = os.path.join(save_path, f"diffusion_epoch_{epoch+1:03d}.pth")
        save_checkpoint(model, optimizer, epoch+1, avg_train_loss, avg_val_loss, path=checkpoint_file)
        
        # Save generated sample
        samples = model.generate(num_images=10, use_ema=True) # Use EMA for generation
        sample_file = os.path.join(samples_path, f"diffusion_samples_epoch_{epoch+1:03d}.png")
        save_image(samples, sample_file, nrow=5)

    print("Diffusion Training Finished.")
    return model


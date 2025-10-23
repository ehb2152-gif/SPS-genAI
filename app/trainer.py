import torch
import torch.nn as nn
import time

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Implements the GAN training loop.
    
    Args:
        model (tuple): A tuple containing (generator, discriminator)
        data_loader (DataLoader): PyTorch DataLoader for MNIST dataset
        criterion (torch.nn.Module): Loss function, e.g., nn.BCEWithLogitsLoss
        optimizer (tuple): A tuple containing (optimizer_g, optimizer_d)
        device (str): 'cuda' or 'cpu'
        epochs (int): Number of training epochs
    """
    
    # 1. Unpack models and optimizers
    # model.py, get_model('GAN') returns two models
    generator, discriminator = model
    generator.to(device)
    discriminator.to(device)
    
    # The GAN training requires two optimizers
    opt_g, opt_d = optimizer

    # Set models to training mode
    generator.train()
    discriminator.train()
    
    print(f"Starting GAN training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        for i, (real_images, _) in enumerate(data_loader):
            
            # 2. Prepare Data 
            real_images = real_images.to(device)
            # Generator uses Tanh, which outputs [-1, 1]. Must scale the real MNIST images (which are [0, 1]) to the same [-1, 1] range.
            real_images = (real_images * 2) - 1
            
            batch_size = real_images.size(0)

            # Create labels for "real" (1) and "fake" (0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            
            # 3. Train the Discriminator 
            # Goal: Maximize log(D(x)) + log(1 - D(G(z)))
            
            opt_d.zero_grad()
            
            # 3a. Train with real images
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, real_labels)
            loss_d_real.backward()

            # 3b. Train with fake images
            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            
            # We .detach() fake_images to stop gradients from flowing back into the generator while training the discriminator.
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            loss_d_fake.backward()
            
            # Update discriminator weights
            opt_d.step()
            loss_d_total = loss_d_real + loss_d_fake

            
            # 4. Train the Generator
            # Goal: Maximize log(D(G(z)))
            
            opt_g.zero_grad()
            
            # Re-use the fake_images from above, but this time we do want the gradients to flow back.
            output_g = discriminator(fake_images)
            
            # "Lie" to the generator: tell it the fake images were "real" (use real_labels)
            loss_g = criterion(output_g, real_labels)
            loss_g.backward()
            
            # Update generator weights
            opt_g.step()

            # 5. Log Progress 
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], "
                      f"Loss_D: {loss_d_total.item():.4f}, "
                      f"Loss_G: {loss_g.item():.4f}")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    print("Training Finished.")
    
    # Return the trained models [cite: 33]
    return generator, discriminator
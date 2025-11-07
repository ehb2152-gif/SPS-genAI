import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

# Assignment 2 (FCNN) Model 
# Adapted for 3x32x32 CIFAR-10 images.
class FCNN(nn.Module):
    """
    A simple Fully-Connected Neural Network for 3x32x32 images.
    From Module 4, Practical 1.
    """
    def __init__(self, num_classes=10):
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Logits
        return x

# Assignment 2 (CNN) Model 
# Adapted for 3x32x32 CIFAR-10 images.
class CNN(nn.Module):
    """
    A basic Convolutional Neural Network for 3x32x32 images.
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 16 x 32 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16 x 16 x 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 32 x 16 x 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32 x 8 x 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Assignment 2 (EnhancedCNN) Model 
# Note: This model is for 3-channel, 64x64 images
class FinalCNN(nn.Module):
    """
    This is your EnhancedCNN model from Assignment 2 for CIFAR-10.
    Input: 3 x 64 x 64
    """
    def __init__(self, num_classes=10):
        super(FinalCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        # 32 features * 16 * 16 = 8192
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 16 * 16, 100), 
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x) # [batch, 16, 32, 32]
        x = self.conv_layer2(x) # [batch, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16) # Flatten to [batch, 8192]
        x = self.fc_layer(x)
        return x

# Assignment 3 (GAN) Models 
# Note: This Generator/Discriminator is for 1-channel 28x28 MNIST images.
class Generator(nn.Module):
    """
    GAN Generator Model (for 28x28 MNIST)
    Input: Noise vector of shape (BATCH_SIZE, 100)
    Output: Image of shape (BATCH_SIZE, 1, 28, 28)
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 7 * 7 * 128)
        self.conv_trans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_trans2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Outputs in range [-1, 1]
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        return x

class Discriminator(nn.Module):
    """
    GAN Discriminator Model (for 28x28 MNIST)
    Input: Image of shape (1, 28, 28)
    Output: Single logit (real/fake probability)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x

# Assignment 3 (VAE) Model 
# Original was for 1-channel FMNIST, in_channels/out_channels are now 3
class Encoder(nn.Module):
    """
    VAE Encoder
    """
    def __init__(self, in_channels=3, latent_dim=2):
        super(Encoder, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)   # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    VAE Decoder
    """
    def __init__(self, out_channels=3, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 4x4 -> 8x8
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 8x8 -> 16x16
        self.conv_trans3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1) # 16x16 -> 32x32
        
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 128, 4, 4)
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = torch.sigmoid(self.conv_trans3(x)) # To [0, 1] range
        return x

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)
    """
    def __init__(self, in_channels=3, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# Assignment 4 (EBM) Model 
class EBM(nn.Module):
    """
    Energy-Based Model (EBM)
    """
    def __init__(self, in_channels=3, features=32):
        super(EBM, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1), # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1), # 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1), # 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 4, features * 8, 4, 2, 1), # 4x4
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 8, 1, 4, 1, 0), # 1x1
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x).squeeze()


# Assignment 4 (Diffusion) Model & Helpers 

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w).transpose(-1, -2), qkv)
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(-1, -2).reshape(b, -1, h, w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w).transpose(-1, -2), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        out = torch.einsum("b h d e, b h n d -> b h n e", context, q)
        out = out.transpose(-1, -2).reshape(b, -1, h, w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class UNet(nn.Module):
    """
    The UNet model that learns to predict noise in the diffusion process.
    Input/Output is 3x32x32.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        
        dims = [features, *map(lambda m: features * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Time embedding
        time_dim = features * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(features),
            nn.Linear(features, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, features, 7, padding=3)
        
        # Downsampling blocks
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    PreNorm(dim_in, LinearAttention(dim_in)),
                    nn.Conv2d(dim_in, dim_out, 4, 2, 1) # Downsample
                ])
            )

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = PreNorm(mid_dim, Attention(mid_dim))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling blocks
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            # e.g., ind=0, (dim_in, dim_out) = (128, 256) -> upsample(256, 128)
            #       block_in = 128 + 128 = 256
            # e.g., ind=1, (dim_in, dim_out) = (64, 128) -> upsample(128, 64)
            #       block_in = 64 + 64 = 128
            # e.g., ind=2, (dim_in, dim_out) = (32, 64) -> upsample(64, 32)
            #       block_in = 32 + 32 = 64
            # e.g., ind=3, (dim_in, dim_out) = (32, 32) -> upsample(32, 32)
            #       block_in = 32 + 32 = 64
            
            # We need to find the channel size of the skip connection
            # h.pop() gives dims[-(ind+1)], e.g. 128, 64, 32, 32
            # The *other* h.pop() gives the same
            # The input to the ResnetBlock is dim_in (from upsample) + dim_in (from skip)
            
            skip_channels = dims[max(0, len(dims) - 2 - ind)] # This gets the skip channel size
            
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_in + skip_channels, dim_in, time_emb_dim=time_dim),
                    ResnetBlock(dim_in + skip_channels, dim_in, time_emb_dim=time_dim),
                    PreNorm(dim_in, LinearAttention(dim_in)),
                    nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1) 
                ])
            )
            
        # Final convolution
        self.conv_out = nn.Sequential(
            ResnetBlock(features * 2, features), # Input is features + features (from h)
            nn.Conv2d(features, out_channels, 1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.conv1(x)
        h = [x] # h = [ (B, 32, 32, 32) ]

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t); h.append(x) 
            x = block2(x, t); x = attn(x); h.append(x) 
            x = downsample(x) 
        
        # x is (B, 256, 2, 2)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # x is (B, 256, 2, 2)

        # Upsampling
        for block1, block2, attn, upsample in self.ups:
            # e.g., Loop 1: dim_in=128, dim_out=256
            x = upsample(x) # e.g., (B, 256, 2, 2) -> (B, 128, 4, 4)
            
            skip = h.pop() # (B, 128, 4, 4)
            x = torch.cat((x, skip), dim=1) # (B, 256, 4, 4)
            x = block1(x, t) # block1 is ResnetBlock(256, 128). x becomes (B, 128, 4, 4)
            
            skip = h.pop() # (B, 128, 4, 4)
            x = torch.cat((x, skip), dim=1) # (B, 256, 4, 4)
            x = block2(x, t) # block2 is ResnetBlock(256, 128). x becomes (B, 128, 4, 4)
            
            x = attn(x) # (B, 128, 4, 4)
        
        # x is now (B, 32, 32, 32). h.pop() is (B, 32, 32, 32)
        x = torch.cat((x, h.pop()), dim=1) # (B, 64, 32, 32)
        return self.conv_out(x) # conv_out takes 64, returns 3.


class DiffusionModel(nn.Module):
    """
    Wrapper class for the Diffusion UNet that handles the diffusion process
    (noise adding, denoising, sampling).
    """
    def __init__(self, network, n_steps=100, beta_min=1e-4, beta_max=0.02, device='cpu', image_size=32, in_channels=3):
        super().__init__()
        self.network = network.to(device)
        self.device = device
        self.n_steps = n_steps 
        self.image_size = image_size
        self.in_channels = in_channels
        self.betas = torch.linspace(beta_min, beta_max, n_steps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Default normalizer (will be overwritten by trainer)
        self.normalizer_mean = torch.tensor([0.5, 0.5, 0.5], device=device).reshape(1, 3, 1, 1)
        self.normalizer_std = torch.tensor([0.5, 0.5, 0.5], device=device).reshape(1, 3, 1, 1)
        
        # EMA (Exponential Moving Average) network for stable generation
        self.ema_network = self.copy_model(network)

    def copy_model(self, model):
        """ Helper to create a deep copy of the model for EMA """
        
        # Get dimensions from the existing model to ensure a perfect copy
        in_channels = model.conv1.in_channels
        out_channels = model.conv_out[1].out_channels
        features = model.conv1.out_channels
        
        # Re-calculate dim_mults from the model's layers
        dim_mults = []
        for down_block in model.downs:
            # The 4th item is the nn.Conv2d downsampler
            dim_mults.append(down_block[3].out_channels // features)

        new_model = UNet(
            in_channels=in_channels, 
            out_channels=out_channels, 
            features=features,
            dim_mults=tuple(dim_mults) # Pass the original dim_mults
        ).to(self.device)
        
        new_model.load_state_dict(model.state_dict())
        return new_model

    def set_normalizer(self, mean, std):
        """Sets the dataset-specific mean and std for normalization."""
        self.normalizer_mean = mean.reshape(1, 3, 1, 1).to(self.device)
        self.normalizer_std = std.reshape(1, 3, 1, 1).to(self.device)

    def normalize(self, x):
        """Normalizes images from [0, 1] to [-1, 1] using dataset stats."""
        return (x - self.normalizer_mean) / self.normalizer_std

    def denormalize(self, x):
        """Denormalizes images from [-1, 1] back to [0, 1]."""
        return x * self.normalizer_std + self.normalizer_mean

    def add_noise(self, x_0, t):
        """Adds t steps of noise to an image x_0."""
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, x_t, t, model_to_use):
        """Performs one step of denoising using the UNet model."""
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        alpha = self.alphas[t].reshape(-1, 1, 1, 1)
        
        predicted_noise = model_to_use(x_t, t)
        
        term1 = (x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        
        # Add noise back in if t > 0
        z = torch.randn_like(x_t, device=self.device)
        
        # Calculate sigma_t
        alpha_bar_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)[t] # [1.0, a_bar_0, a_bar_1, ...][t]
        beta_tilde = self.betas[t] * (1. - alpha_bar_prev) / (1. - alpha_bar.squeeze()) # Use vector indexing
        
        sigma_t = torch.sqrt(beta_tilde).reshape(-1, 1, 1, 1)
        term2 = sigma_t * z
        
        # Only add noise if not the last step
        x_t_minus_1 = term1 + (t > 0).float().reshape(-1, 1, 1, 1) * term2
        return x_t_minus_1

    @torch.no_grad()
    def generate(self, num_images=1, use_ema=True):
        """Generates new images by running the full reverse diffusion process."""
        self.eval()
        model_to_use = self.ema_network if use_ema else self.network
        
        # Start from pure noise
        x_t = torch.randn(num_images, self.in_channels, self.image_size, self.image_size, device=self.device)
        
        # Denoise step-by-step
        for t in reversed(range(self.n_steps)):
            ts = torch.full((num_images,), t, device=self.device, dtype=torch.long)
            x_t = self.denoise(x_t, ts, model_to_use)
            
        # Denormalize from [-1, 1] to [0, 1]
        x_t = self.denormalize(x_t)
        x_t = torch.clamp(x_t, 0, 1)
        self.train()
        return x_t

    def update_ema(self, decay=0.999):
        """Updates the EMA network weights."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# Factory Function (Fully Resolved) 

def get_model(model_name):
    """
    Factory function to return the appropriate model.
    """
    model = None
    
    # Assignment 2 Models 
    if model_name == 'FCNN':
        # 3-channel, 32x32 input
        model = FCNN()
    elif model_name == 'CNN':
        # 3-channel, 32x32 input
        model = CNN()
    elif model_name == 'EnhancedCNN':
        # 3-channel, 64x64 input
        model = FinalCNN()
        
    # Assignment 3 Models 
    elif model_name == 'GAN':
        # For training, we need both models (for 1-channel, 28x28 MNIST)
        model = (Generator(), Discriminator())
    elif model_name == 'Generator':
        # For inference/generation (1-channel, 28x28 MNIST)
        model = Generator()
    elif model_name == 'Discriminator':
        model = Discriminator()
    elif model_name == 'VAE':
        # 3-channel, 32x32 input
        model = VAE(in_channels=3)
        
    # Assignment 4 Models 
    elif model_name == 'EBM':
        # 3-channel, 32x32 input
        model = EBM(in_channels=3, features=32)
        
    elif model_name == 'DiffusionModel': 
        # Wrapper class containing 3-channel, 32x32 UNet
        unet_network = UNet(in_channels=3, out_channels=3, features=32) 
        model = DiffusionModel(network=unet_network, n_steps=100, device='cpu', image_size=32, in_channels=3) 
        
    if model is None:
        raise ValueError(f"Model name '{model_name}' not recognized.")
        
    return model
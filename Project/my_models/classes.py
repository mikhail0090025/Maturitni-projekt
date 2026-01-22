class ResidualBlock(nn.Module):
    """
    Основной вычислительный блок с residual connection.
    """
    def __init__(self, in_channels, width):
        super().__init__()
        # Используем GroupNorm вместо BatchNorm без scale и center
        self.norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        # Проекция для residual, если каналы не совпадают
        if in_channels != width:
            self.proj = nn.Conv2d(in_channels, width, kernel_size=1)
        else:
            self.proj = None

        self.in_channels = in_channels
        self.width = width

    def forward(self, x):
        residual = x
        if self.proj is not None:
            residual = self.proj(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = F.silu(x)  # Swish активация
        x = self.conv2(x)
        return x + residual

class DownBlock(nn.Module):
    """
    Понижающий блок с сохранением skip connections.
    """
    def __init__(self, in_channels, width, block_depth, emb_dim):
        super().__init__()
        # После конкатенации каналы = in_channels + emb_dim
        self.initial_conv = nn.Conv2d(in_channels + emb_dim, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(width, width) for _ in range(block_depth)])
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.width = width
        self.block_depth = block_depth
        self.emb_dim = emb_dim

    def forward(self, x, skips, emb):
        # Интерполяция emb до размера x
        height = x.shape[2]
        e = F.interpolate(emb, size=(height, height), mode='nearest')
        x = torch.cat([x, e], dim=1)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    """
    Повышающий блок с использованием skip connections.
    """
    def __init__(self, width, out_width, block_depth):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # После конкатенации со skip каналы удваиваются
        self.blocks = nn.ModuleList([ResidualBlock(width * 2, width if i < block_depth-1 else out_width) for i in range(block_depth)])

    def forward(self, x, skips):
        x = self.upsample(x)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x

class UNet(nn.Module):
    """
    U-Net для диффузионной модели.
    """
    def __init__(self, image_size, widths, block_depth, emb_dim=32):
        super().__init__()
        self.image_size = image_size
        self.emb_dim = emb_dim

        # Понижающие блоки
        self.downs = nn.ModuleList()
        in_channels = 3  # Начальные каналы от noisy_images
        for width in widths:
            self.downs.append(DownBlock(in_channels, width, block_depth, emb_dim))
            in_channels = width

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(widths[-1], widths[-1]) for _ in range(block_depth)]
        )

        # Повышающие блоки
        self.ups = nn.ModuleList()
        for i, width in enumerate(reversed(widths)):
            self.ups.append(UpBlock(width, list(reversed(widths))[i+1 if i+1 != len(widths) else i], block_depth))

        # Финальные слои
        final_in_channels = widths[0] + 3 + emb_dim  # x + noisy_images + emb
        self.final_conv1 = nn.Conv2d(final_in_channels, 32, kernel_size=1)
        self.final_conv2 = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.final_conv2.weight)
        nn.init.zeros_(self.final_conv2.bias)

        self.widths = widths

    def forward(self, noisy_images, noise_variances):
        emb = sinusoidal_embedding(noise_variances)
        skips = []
        x = noisy_images

        # Downsampling
        for down in self.downs:
            x = down(x, skips, emb)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for up in self.ups:
            x = up(x, skips)

        # Финальная обработка
        e = F.interpolate(emb, size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noisy_images, e], dim=1)
        x = self.final_conv1(x)
        x = F.silu(x)
        x = self.final_conv2(x)
        return x
    '''
    def __init__(self, image_size, widths, block_depth, emb_dim=32):
        super().__init__()
        self.image_size = image_size
        self.emb_dim = emb_dim

        # Понижающие блоки
        self.downs = nn.ModuleList()
        in_channels = 3  # Начальные каналы от noisy_images
        for width in widths[:-1]:
            self.downs.append(DownBlock(in_channels, width, block_depth, emb_dim))
            in_channels = width

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(widths[-1], widths[-1]) for _ in range(block_depth)]
        )

        # Повышающие блоки
        self.ups = nn.ModuleList()
        for i, width in enumerate(reversed(widths[:-1])):
            self.ups.append(UpBlock(width, list(reversed(widths[:-1]))[i+1 if i+1 != len(widths[:-1]) else i], block_depth))

        # Финальные слои
        final_in_channels = widths[0] + 3 + emb_dim  # x + noisy_images + emb
        self.final_conv1 = nn.Conv2d(final_in_channels, 32, kernel_size=1)
        self.final_conv2 = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.zeros_(self.final_conv2.weight)
        nn.init.zeros_(self.final_conv2.bias)

        self.widths = widths

        self.before_bottleneck = ResidualBlock(widths[-2], widths[-1])
        self.after_bottleneck = ResidualBlock(widths[-1], widths[-2])


    def forward(self, noisy_images, noise_variances):
        emb = sinusoidal_embedding(noise_variances)
        skips = []
        x = noisy_images

        # Downsampling
        for down in self.downs:
            x = down(x, skips, emb)


        # Bottleneck
        x = self.before_bottleneck(x)
        x = self.bottleneck(x)
        x = self.after_bottleneck(x)


        # Upsampling
        for up in self.ups:
            x = up(x, skips)

        # Финальная обработка
        e = F.interpolate(emb, size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noisy_images, e], dim=1)
        x = self.final_conv1(x)
        x = F.silu(x)
        x = self.final_conv2(x)
        return x
'''

from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io

class DiffusionModel(nn.Module):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        self.main_model = UNet(image_size, widths, block_depth)

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, x, t):
        return self.main_model(x, t)

    def denormalize(self, images, power=0.5):
        images = dataset_mean + images * dataset_std**power
        return images.clamp(0, 1)

    def diffusion_schedule(self, diffusion_times):
        start_angle = torch.acos(torch.tensor(max_signal_rate)) # Convert to tensor
        end_angle = torch.acos(torch.tensor(min_signal_rate))   # Convert to tensor

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates):
        pred_noises = self.forward(noisy_images, noise_rates**2)

        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images


    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise.to(device)  # Move initial noise to GPU

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = torch.ones((num_images, 1, 1, 1), device=device) - step * step_size # Move diffusion_times to GPU
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

            if step != diffusion_steps - 1:
                del pred_noises, pred_images, noise_rates, signal_rates, next_noise_rates, next_signal_rates, diffusion_times
            torch.cuda.empty_cache()

        return pred_images

    def generate(self, num_images, diffusion_steps, power=0.5):
        initial_noise = torch.randn((num_images, 3, image_size, image_size))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images, power)
        return generated_images

    def generate_gif(self, initial_noise, diffusion_steps, filename="diffusion_process.gif", power=0.5):
        """
        Generates a GIF of the reverse diffusion process with step number.
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise.to(device)
        frames = []

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = torch.ones((num_images, 1, 1, 1), device=device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

            # Denormalize and convert to numpy for plotting
            current_image = self.denormalize(pred_images.detach().cpu(), power=power).permute(0, 2, 3, 1).squeeze(0).numpy()

            # Create a Matplotlib figure and plot the image
            fig, ax = plt.subplots(figsize=(image_size/10, image_size/10), dpi=100) # Adjust figsize and dpi as needed
            ax.imshow(current_image)
            ax.set_title(f"Step {step+1}/{diffusion_steps}", fontsize=50)
            ax.axis("off")
            plt.tight_layout()

            # Save the plot to a buffer and then to a PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)

            plt.close(fig) # Close the figure to free memory

            if step != diffusion_steps - 1:
                del pred_noises, pred_images, noise_rates, signal_rates, next_noise_rates, next_signal_rates, diffusion_times
            torch.cuda.empty_cache()

        # Save frames as GIF
        if frames:
            for i in range(100):
                frames.append(frames[-1])
            frames[0].save(filename, save_all=True, append_images=frames, duration=100, loop=0)
            print(f"GIF saved successfully to {filename}")
        else:
            print("No frames were generated for the GIF.")


    def train_step(self, images):
        noises = torch.randn((batch_size, 3, image_size, image_size)).to(device) # Move noises to GPU

        diffusion_times = torch.rand(batch_size, 1, 1, 1, device=device) * (1.0 - 0.0) + 0.0 # Move diffusion_times to GPU
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        self.optimizer.zero_grad()
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
        noise_loss = nn.L1Loss()(pred_noises, noises)
        noise_loss.backward()
        self.optimizer.step()
        return noise_loss.item()

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, power=0.5):
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
            power=power
        ).detach().cpu().permute(0, 2, 3, 1)

        plt.figure(figsize=(num_cols * 2, num_rows * 2))

        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
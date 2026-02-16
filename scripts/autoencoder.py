import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from omni.models.autoencoder import Autoencoder
from omni.utils.dataset import TensorImageFolder
from omni.utils.device import get_device

device = get_device()
use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

# Create checkpoints directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize Weights & Biases
wandb.init(project="autoencoder", name="afhq_v2_baseline")

train_data = TensorImageFolder("./data/afhq_v2_preprocessed/train")
test_data = TensorImageFolder("./data/afhq_v2_preprocessed/test")

cpu_count = os.cpu_count() or 1
num_workers = min(8, cpu_count // 2)

train_loader = DataLoader(
    train_data,
    batch_size=128,  # you can go higher now
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=512,  # you can go higher now
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


step = 0

model = Autoencoder().to(device)

optimizer_ae = torch.optim.AdamW(model.parameters(), lr=8e-4)

criterion_recon = nn.MSELoss()

scaler = None  # not used for bf16

print(summary(model, (1, 3, 128, 128), device=device.type))

# Training configuration
total_steps = 5000
eval_interval = 500  # Run evaluation every N steps
checkpoint_interval = 1000  # Save checkpoint every N steps


# Create infinite iterator for training
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


train_iter = infinite_dataloader(train_loader)

step = 0
pbar = tqdm(range(total_steps), desc="Training")
first_eval = True

for _ in pbar:
    model.train()
    images, _ = next(train_iter)
    images = images.to(device, non_blocking=True)
    batch_size = images.size(0)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        images = images.float().div_(255.0)
        recon = model(images)

        # ==================
        # Autoencoder step
        # ==================
        optimizer_ae.zero_grad(set_to_none=True)

        recon = model(images)

        # Reconstruction loss
        loss_recon = criterion_recon(recon, images)

        loss_ae = loss_recon
        loss_ae.backward()
        optimizer_ae.step()

    step += 1

    loss_ae_item = loss_ae.item()
    psnr_item = peak_signal_noise_ratio(recon, images, data_range=1.0).item()
    ssim_item = structural_similarity_index_measure(
        recon, images, data_range=1.0
    ).item()

    pbar.set_postfix(
        {
            "loss_ae": loss_ae_item,
            "psnr": psnr_item,
            "ssim": ssim_item,
        }
    )

    # Log metrics to W&B
    wandb.log(
        {
            "train/loss_ae": loss_ae_item,
            "train/psnr": psnr_item,
            "train/ssim": ssim_item,
        },
        step=step,
    )

    # Save checkpoint
    if step % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"autoencoder_step_{step}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

    # --------------------
    # Eval
    # --------------------
    if step % eval_interval == 0:
        model.eval()

        total_loss_recon = 0
        eval_batch_count = 0

        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ),
        ):
            for batch_idx, (images, _) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                images = images.float().div_(255.0)
                recon = model(images)

                loss_recon = criterion_recon(recon, images)

                total_loss_recon += loss_recon.item() * images.size(0)

                # Log image grid on first eval batch
                if batch_idx == 0 and first_eval:
                    num_samples = min(16, images.size(0))
                    grid_images = torch.cat(
                        [images[:num_samples], recon[:num_samples]], dim=0
                    )
                    grid = make_grid(grid_images, nrow=num_samples)

                    # Convert grid to PIL Image for W&B
                    grid_pil = wandb.Image(
                        grid.permute(1, 2, 0).cpu().numpy(),
                        caption="Original (top) vs Reconstruction (bottom)",
                    )
                    wandb.log({"eval/reconstruction_grid": grid_pil}, step=step)
                    first_eval = False

        test_loss = total_loss_recon / len(test_loader.dataset)
        print(f"\nStep {step} - Test loss (recon): {test_loss:.6f}")

        # Log validation metrics to W&B
        wandb.log({"val/loss_recon": test_loss}, step=step)


# Save final model
final_checkpoint_path = os.path.join(checkpoint_dir, "autoencoder_final.pt")
torch.save(model.state_dict(), final_checkpoint_path)
print(f"Final model saved: {final_checkpoint_path}")

with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images[:32].to(device, non_blocking=True)
    images = images.float().div_(255.0)
    recon = model(images)


grid = make_grid(torch.cat([images.cpu(), recon.cpu()], dim=0), nrow=16)
plt.figure(figsize=(24, 16))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()

wandb.finish()
print("W&B run finished")

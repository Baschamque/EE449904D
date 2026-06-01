"""
First attempt in factorizing (separating) motion and shape. 
We use a canonical digit representation and a known (explicit) motion model for the trajectory.
MSE was promising (lower than any of the entangled models) even though the visual reconstruction was possibly as bad, but at least it looks more like actual digits now.
"""


import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST


H, W = 64, 64
T = 31
SPF = 0.25
M = max(1, int(H * W * SPF))
LATENT_DIM = 32
NOISE_LEVEL = 0.02
TRAIN_BATCH = 2048
STATIC_BATCH = 256
STATIC_EPOCHS = 400
RECON_STEPS = 300
DIGITS = list(range(10))

OMEGA = 2 * math.pi * 1.5
GAMMA = 1.5
DEFAULT_R = 20.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_span = torch.linspace(0.0, 1.0, T, device=device)


def decode_to_image(decoder_output: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(decoder_output)


def build_mask_normalizer(masks: torch.Tensor) -> torch.Tensor:
    return masks.sum(dim=(-1, -2)).clamp_min(1.0)


def forward_measurements(masks: torch.Tensor, frames: torch.Tensor, normalizer: torch.Tensor) -> torch.Tensor:
    if frames.dim() == 3:
        return torch.einsum("tmhw,thw->tm", masks, frames) / normalizer
    if frames.dim() == 4:
        return torch.einsum("tmhw,bthw->btm", masks, frames) / normalizer.unsqueeze(0)
    raise ValueError(f"Expected frames with 3 or 4 dims, got {frames.dim()}")


def pad_digit(img: torch.Tensor, H: int, W: int) -> torch.Tensor:
    pad_h = (H - 28) // 2
    pad_w = (W - 28) // 2
    return F.pad(img, (pad_w, pad_w, pad_h, pad_h))


def differentiable_translate(image: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    num_frames = tx.numel()
    img = image.view(1, 1, H, W).expand(num_frames, -1, -1, -1)
    theta = torch.zeros(num_frames, 2, 3, device=image.device, dtype=image.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * tx / W
    theta[:, 1, 2] = -2.0 * ty / H
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)


def render_spiral_from_canonical(image: torch.Tensor, phase: torch.Tensor, r_init: torch.Tensor) -> torch.Tensor:
    r_t = r_init * torch.exp(-GAMMA * t_span)
    angle_t = OMEGA * t_span + phase
    tx = r_t * torch.cos(angle_t)
    ty = r_t * torch.sin(angle_t)
    return differentiable_translate(image, tx, ty)


def load_centered_mnist_train(num_samples: int, digits: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    all_targets = dataset.targets
    images = []
    labels = []
    base_count = num_samples // len(digits)
    remainder = num_samples % len(digits)

    for digit in digits:
        digit_indices = torch.where(all_targets == digit)[0]
        take = base_count + (1 if digit < remainder else 0)
        print(f"   Found {len(digit_indices)} centered samples of digit '{digit}', taking {take}")
        perm = torch.randperm(len(digit_indices))[:take]
        for idx in digit_indices[perm]:
            img, _ = dataset[idx.item()]
            images.append(pad_digit(img, H, W).squeeze(0))
            labels.append(digit)

    return torch.stack(images, dim=0).to(device), torch.tensor(labels, device=device)


def load_centered_mnist_test(digits: list[int]) -> dict[int, torch.Tensor]:
    dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    all_targets = dataset.targets
    samples: dict[int, torch.Tensor] = {}
    for digit in digits:
        digit_indices = torch.where(all_targets == digit)[0]
        img, _ = dataset[digit_indices[0].item()]
        samples[digit] = pad_digit(img, H, W).squeeze(0).to(device)
    return samples


class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(4, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(4, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(4, 256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(),
            nn.Linear(512, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[:-2]
        z = self.net(x.reshape(-1, 1, H, W))
        return z.reshape(*s, LATENT_DIM)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.SiLU(),
        )

        def res_block(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.SiLU(),
            )

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.rb1 = res_block(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.rb2 = res_block(64)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.rb3 = res_block(32)
        self.up4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.rb4 = res_block(16)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = z.shape[:-1]
        x = self.fc(z.reshape(-1, LATENT_DIM)).reshape(-1, 256, 4, 4)
        x = self.up1(x)
        x = self.rb1(x) + x
        x = self.up2(x)
        x = self.rb2(x) + x
        x = self.up3(x)
        x = self.rb3(x) + x
        x = self.up4(x)
        x = self.rb4(x) + x
        return self.out(x).squeeze(1).reshape(*s, H, W)


def inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def reconstruct_digit_sequence(
    encoder: Encoder2D,
    decoder: Decoder2D,
    train_static: torch.Tensor,
    train_labels: torch.Tensor,
    digit: int,
    z_stats: dict[int, tuple[torch.Tensor, torch.Tensor]],
    masks: torch.Tensor,
    mask_norm: torch.Tensor,
    y_measured: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    z_mu, z_var = z_stats[digit]
    best_loss = float("inf")
    best_z = None

    with torch.no_grad():
        digit_indices = torch.where(train_labels == digit)[0]
        candidate_indices = digit_indices[: min(digit_indices.numel(), 256)]
        for i in range(0, candidate_indices.numel(), STATIC_BATCH):
            batch_indices = candidate_indices[i:i + STATIC_BATCH]
            x_cand = train_static[batch_indices]
            z_cand = encoder(x_cand.unsqueeze(1)).squeeze(1)
            canonical = decode_to_image(decoder(z_cand.unsqueeze(1)).squeeze(1))
            frames = []
            for j in range(canonical.size(0)):
                frames.append(render_spiral_from_canonical(canonical[j], torch.tensor(0.0, device=device), torch.tensor(DEFAULT_R, device=device)))
            x_est = torch.stack(frames, dim=0)
            y_est = forward_measurements(masks, x_est, mask_norm)
            losses = ((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            local_best = torch.argmin(losses)
            if losses[local_best].item() < best_loss:
                best_loss = losses[local_best].item()
                best_z = z_cand[local_best].clone()

    z = nn.Parameter(best_z)
    phase = nn.Parameter(torch.tensor(0.0, device=device))
    raw_r = nn.Parameter(torch.tensor(inverse_softplus(DEFAULT_R), device=device))
    optimizer = optim.Adam([z, phase, raw_r], lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECON_STEPS, eta_min=1e-4)
    criterion = nn.MSELoss()

    best_total = float("inf")
    best_frames = None
    best_phase = None
    best_r = None

    for step in range(RECON_STEPS):
        optimizer.zero_grad()
        canonical_logits = decoder(z.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        canonical = decode_to_image(canonical_logits)
        r_init = F.softplus(raw_r)
        frames = render_spiral_from_canonical(canonical, phase, r_init)
        y_est = forward_measurements(masks, frames, mask_norm)

        loss_meas = criterion(y_est, y_measured)
        loss_latent = 1e-3 * (((z - z_mu) ** 2) / z_var).mean()
        loss_motion = 1e-4 * (phase ** 2) + 1e-4 * ((r_init - DEFAULT_R) ** 2)
        loss_total = loss_meas + loss_latent + loss_motion

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([z, phase, raw_r], 1.0)
        optimizer.step()
        scheduler.step()

        if loss_total.item() < best_total:
            best_total = loss_total.item()
            best_frames = frames.detach().clone()
            best_phase = phase.detach().item()
            best_r = r_init.detach().item()

        if (step + 1) % 100 == 0:
            print(
                f"   Recon {step + 1:3d}/{RECON_STEPS} | Total={loss_total.item():.6f} "
                f"Meas={loss_meas.item():.6f} phase={phase.item():.3f} r={r_init.item():.3f}"
            )

    return best_frames, canonical.detach(), best_phase, best_r


def main():
    print("1. Loading All-Digit Centered MNIST Dataset...")
    x_train_static, x_train_labels = load_centered_mnist_train(TRAIN_BATCH, DIGITS)
    x_test_static_by_digit = load_centered_mnist_test(DIGITS)
    print(f"   Train static: {x_train_static.shape}, Test digits: {len(x_test_static_by_digit)}")

    print("\n2. Training Appearance Autoencoder...")
    encoder = Encoder2D().to(device)
    decoder = Decoder2D().to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STATIC_EPOCHS, eta_min=1e-5)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(STATIC_EPOCHS):
        idx = torch.randperm(x_train_static.size(0), device=device)[:STATIC_BATCH]
        x_mb = x_train_static[idx]
        optimizer.zero_grad()
        z = encoder(x_mb.unsqueeze(1)).squeeze(1)
        logits = decoder(z.unsqueeze(1)).squeeze(1)
        loss_recon = criterion_bce(logits, x_mb)
        loss_reg = 1e-4 * z.pow(2).mean()
        loss = loss_recon + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"   AE Epoch {epoch + 1:3d}/{STATIC_EPOCHS} | Recon BCE = {loss_recon.item():.5f}")

    with torch.no_grad():
        z_all = []
        z_labels = []
        for i in range(0, x_train_static.size(0), STATIC_BATCH):
            z_all.append(encoder(x_train_static[i:i + STATIC_BATCH].unsqueeze(1)).squeeze(1))
            z_labels.append(x_train_labels[i:i + STATIC_BATCH])
        z_all = torch.cat(z_all, dim=0)
        z_labels = torch.cat(z_labels, dim=0)
        z_stats = {}
        for digit in DIGITS:
            z_digit = z_all[z_labels == digit]
            z_stats[digit] = (z_digit.mean(dim=0), z_digit.var(dim=0) + 1e-4)

        sanity_img = x_train_static[:5]
        sanity_z = encoder(sanity_img.unsqueeze(1)).squeeze(1)
        sanity_recon = decode_to_image(decoder(sanity_z.unsqueeze(1)).squeeze(1))

    fig_sanity, axes_sanity = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes_sanity[0, i].imshow(sanity_img[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_sanity[0, i].set_title("GT")
        axes_sanity[0, i].axis("off")
        axes_sanity[1, i].imshow(sanity_recon[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_sanity[1, i].set_title("AE")
        axes_sanity[1, i].axis("off")
    plt.tight_layout()
    fig_sanity.savefig("appearance_sanity_check.png", dpi=150)
    plt.close(fig_sanity)
    print("   Saved 'appearance_sanity_check.png'.")

    print("\n3. Building Shared Measurement Operator...")
    masks = torch.randint(0, 2, (T, M, H, W), device=device).float()
    mask_norm = build_mask_normalizer(masks)

    print("\n4. Reconstructing All Digits with Separated Appearance and Motion...")
    results = {}
    criterion = nn.MSELoss()

    for digit in DIGITS:
        print(f"\n   Reconstructing digit {digit}...")
        canonical_gt = x_test_static_by_digit[digit]
        frames_gt = render_spiral_from_canonical(
            canonical_gt,
            torch.tensor(0.0, device=device),
            torch.tensor(DEFAULT_R, device=device),
        )
        y_clean = forward_measurements(masks, frames_gt, mask_norm)
        noise_std = NOISE_LEVEL * y_clean.std()
        y_measured = y_clean + torch.randn_like(y_clean) * noise_std

        frames_recon, canonical_recon, phase_hat, r_hat = reconstruct_digit_sequence(
            encoder,
            decoder,
            x_train_static,
            x_train_labels,
            digit,
            z_stats,
            masks,
            mask_norm,
            y_measured,
        )

        meas_mse = criterion(forward_measurements(masks, frames_recon, mask_norm), y_measured).item()
        img_mse = criterion(frames_recon, frames_gt).item()
        canon_mse = criterion(canonical_recon, canonical_gt).item()
        results[digit] = {
            "gt": frames_gt.detach().cpu().numpy(),
            "recon": frames_recon.detach().cpu().numpy(),
            "canonical_gt": canonical_gt.detach().cpu().numpy(),
            "canonical_recon": canonical_recon.detach().cpu().numpy(),
            "meas_mse": meas_mse,
            "img_mse": img_mse,
            "canon_mse": canon_mse,
            "phase": phase_hat,
            "r_init": r_hat,
        }
        print(
            f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | "
            f"Frame MSE = {img_mse:.6f} | Canonical MSE = {canon_mse:.6f} | "
            f"phase={phase_hat:.3f} r={r_hat:.3f}"
        )

    print("\n5. Saving Comparison Figure...")
    frames_to_show = [0, T // 2, T - 1]
    fig, axes = plt.subplots(len(DIGITS) * 2, len(frames_to_show) + 2, figsize=(12, 2.4 * len(DIGITS)))

    for row, digit in enumerate(DIGITS):
        axes[2 * row, 0].imshow(results[digit]["canonical_gt"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row, 0].set_title(f"{digit} Canon GT")
        axes[2 * row, 0].axis("off")
        axes[2 * row + 1, 0].imshow(results[digit]["canonical_recon"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row + 1, 0].set_title(f"{digit} Canon Rec")
        axes[2 * row + 1, 0].axis("off")

        for col, frame_idx in enumerate(frames_to_show, start=1):
            axes[2 * row, col].imshow(results[digit]["gt"][frame_idx], cmap="gray", vmin=0, vmax=1)
            axes[2 * row, col].set_title(f"{digit} GT t={frame_idx}")
            axes[2 * row, col].axis("off")
            axes[2 * row + 1, col].imshow(results[digit]["recon"][frame_idx], cmap="gray", vmin=0, vmax=1)
            axes[2 * row + 1, col].set_title(f"{digit} Rec t={frame_idx}")
            axes[2 * row + 1, col].axis("off")

        axes[2 * row, -1].axis("off")
        axes[2 * row + 1, -1].axis("off")
        axes[2 * row + 1, -1].text(
            0.0,
            0.5,
            f"meas={results[digit]['meas_mse']:.4g}\nframe={results[digit]['img_mse']:.4g}\ncanon={results[digit]['canon_mse']:.4g}",
            fontsize=9,
            va="center",
        )

    fig.suptitle("Factorized Appearance + Motion Reconstruction", fontsize=16)
    plt.tight_layout()
    fig.savefig("reconstruction_factorized_all_digits_t31.png", dpi=150)
    plt.close(fig)
    print("   Saved 'reconstruction_factorized_all_digits_t31.png'.")

    print("\n6. Saving Timelapse GIF...")
    fig_anim, axes_anim = plt.subplots(len(DIGITS), 2, figsize=(6, 2.1 * len(DIGITS)))
    im_pairs = []
    for row, digit in enumerate(DIGITS):
        axes_anim[row, 0].axis("off")
        axes_anim[row, 1].axis("off")
        axes_anim[row, 0].set_title(f"{digit} GT")
        axes_anim[row, 1].set_title(f"{digit} Recon")
        im_gt = axes_anim[row, 0].imshow(results[digit]["gt"][0], cmap="gray", vmin=0, vmax=1)
        im_rec = axes_anim[row, 1].imshow(results[digit]["recon"][0], cmap="gray", vmin=0, vmax=1)
        im_pairs.append((im_gt, im_rec))

    def update(frame_idx: int):
        artists = []
        for row, digit in enumerate(DIGITS):
            im_gt, im_rec = im_pairs[row]
            im_gt.set_array(results[digit]["gt"][frame_idx])
            im_rec.set_array(results[digit]["recon"][frame_idx])
            artists.extend([im_gt, im_rec])
        fig_anim.suptitle(f"Factorized Appearance + Motion - Frame {frame_idx + 1}/{T}", fontsize=16)
        return artists

    ani = animation.FuncAnimation(fig_anim, update, frames=T, interval=250, blit=False)
    ani.save("reconstruction_factorized_all_digits_t31.gif", writer="pillow")
    plt.close(fig_anim)
    print("   Saved 'reconstruction_factorized_all_digits_t31.gif'.")


if __name__ == "__main__":
    main()

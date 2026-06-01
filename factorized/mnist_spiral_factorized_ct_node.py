"""
This is a version of the factorized method. We changed the hand-coded explicit spiral motion (used in both factorized and exemplar) to a learned CT NODE for the motion.
Replacing the explicit motion with a learned NODE introduced motion uncertainty, so the appearance gets 'pulled' to compensate for it.
We can attribute this to the motion NODE error propagating into the appearance reconstruction which makes the visual reconstruction worse as a result.

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
TRAIN_PER_DIGIT = 128
TEST_PER_DIGIT = 1
NOISE_LEVEL = 0.02
STATIC_BATCH = 256
STATIC_EPOCHS = 350
MOTION_EPOCHS = 500
RECON_STEPS = 300
TOPK_EXEMPLARS = 16
DIGITS = list(range(10))

OMEGA = 2 * math.pi * 1.5
GAMMA = 1.5
R_MEAN = 20.0
R_STD = 2.0

LAMBDA_Z = 1e-3
LAMBDA_EX = 1.0
LAMBDA_TV = 5e-4
LAMBDA_SPARSE = 5e-4
LAMBDA_P0 = 1e-3
SOFTMIN_TEMP = 0.002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_span = torch.linspace(0.0, 1.0, T, device=device)


def pad_digit(img: torch.Tensor) -> torch.Tensor:
    pad_h = (H - 28) // 2
    pad_w = (W - 28) // 2
    return F.pad(img, (pad_w, pad_w, pad_h, pad_h))


def decode_to_image(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def build_mask_normalizer(masks: torch.Tensor) -> torch.Tensor:
    return masks.sum(dim=(-1, -2)).clamp_min(1.0)


def forward_measurements(masks: torch.Tensor, frames: torch.Tensor, normalizer: torch.Tensor) -> torch.Tensor:
    if frames.dim() == 3:
        return torch.einsum("tmhw,thw->tm", masks, frames) / normalizer
    if frames.dim() == 4:
        return torch.einsum("tmhw,bthw->btm", masks, frames) / normalizer.unsqueeze(0)
    raise ValueError(f"Expected frames with 3 or 4 dims, got {frames.dim()}")


def total_variation(image: torch.Tensor) -> torch.Tensor:
    return (image[:, 1:] - image[:, :-1]).abs().mean() + (image[1:, :] - image[:-1]).abs().mean()


def exemplar_softmin_distance(image: torch.Tensor, exemplars: torch.Tensor, temperature: float) -> torch.Tensor:
    distances = ((exemplars - image.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    return -temperature * torch.logsumexp(-distances / temperature, dim=0)


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


def render_from_positions(image: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    tx = positions[:, 0]
    ty = positions[:, 1]
    return differentiable_translate(image, tx, ty)


def spiral_positions(phase: torch.Tensor, r_init: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    r_t = r_init.unsqueeze(-1) * torch.exp(-GAMMA * times.unsqueeze(0))
    angle_t = OMEGA * times.unsqueeze(0) + phase.unsqueeze(-1)
    tx = r_t * torch.cos(angle_t)
    ty = r_t * torch.sin(angle_t)
    return torch.stack([tx, ty], dim=-1)


def odeint_rk4(func, z0, t):
    zs, z = [z0], z0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = func(t[i], z)
        k2 = func(t[i] + dt / 2, z + dt / 2 * k1)
        k3 = func(t[i] + dt / 2, z + dt / 2 * k2)
        k4 = func(t[i + 1], z + dt * k3)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        zs.append(z)
    return torch.stack(zs, dim=0)


def load_centered_mnist(train: bool, per_digit: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())
    targets = dataset.targets
    images = []
    labels = []
    split_name = "train" if train else "test"
    for digit in DIGITS:
        digit_indices = torch.where(targets == digit)[0]
        take = min(per_digit, len(digit_indices))
        print(f"   Found {len(digit_indices)} {split_name} samples of digit '{digit}', taking {take}")
        chosen = digit_indices[:take]
        for idx in chosen:
            img, _ = dataset[idx.item()]
            images.append(pad_digit(img).squeeze(0))
            labels.append(digit)
    return torch.stack(images, dim=0).to(device), torch.tensor(labels, device=device)


def sample_spiral_dataset(images: torch.Tensor, labels: torch.Tensor, fixed_motion: bool = False) -> dict[str, torch.Tensor]:
    num_samples = images.size(0)
    if fixed_motion:
        phase = torch.zeros(num_samples, device=device)
        r_init = torch.full((num_samples,), R_MEAN, device=device)
    else:
        phase = torch.rand(num_samples, device=device) * 2 * math.pi
        r_init = R_MEAN + R_STD * torch.randn(num_samples, device=device)
    positions = spiral_positions(phase, r_init, t_span)
    return {
        "canonical_images": images,
        "labels": labels,
        "phase": phase,
        "r_init": r_init,
        "positions": positions,
    }


class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(4, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(4, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(4, 256), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.SiLU(),
            nn.Linear(512, LATENT_DIM),
        )

    def forward(self, x):
        s = x.shape[:-2]
        z = self.net(x.reshape(-1, 1, H, W))
        return z.reshape(*s, LATENT_DIM)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4), nn.SiLU(),
        )

        def res_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(4, c), nn.SiLU(),
                nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(4, c), nn.SiLU(),
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

    def forward(self, z):
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


class MotionODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 128
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 2),
        )

    def forward(self, t, state):
        del t
        return self.net(state)


def select_topk_exemplars(
    digit_images: torch.Tensor,
    digit_positions: torch.Tensor,
    masks: torch.Tensor,
    mask_norm: torch.Tensor,
    y_measured: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 32
    losses = []
    for i in range(0, digit_images.size(0), batch_size):
        imgs = digit_images[i:i + batch_size]
        pos = digit_positions[i:i + batch_size]
        frames = []
        for j in range(imgs.size(0)):
            frames.append(render_from_positions(imgs[j], pos[j]))
        frame_batch = torch.stack(frames, dim=0)
        y_est = forward_measurements(masks, frame_batch, mask_norm)
        losses.append(((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2)))
    losses = torch.cat(losses, dim=0)
    values, indices = torch.topk(losses, k=min(topk, losses.numel()), largest=False)
    return digit_images[indices], digit_positions[indices], values


def main():
    print("1. Loading Centered MNIST...")
    train_images, train_labels = load_centered_mnist(train=True, per_digit=TRAIN_PER_DIGIT)
    test_images, test_labels = load_centered_mnist(train=False, per_digit=TEST_PER_DIGIT)
    train_data = sample_spiral_dataset(train_images, train_labels, fixed_motion=False)
    test_data = sample_spiral_dataset(test_images, test_labels, fixed_motion=True)

    print("\n2. Training Static Appearance Autoencoder...")
    encoder = Encoder2D().to(device)
    decoder = Decoder2D().to(device)
    opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
    sched_ae = optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=STATIC_EPOCHS, eta_min=1e-5)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(STATIC_EPOCHS):
        idx = torch.randperm(train_images.size(0), device=device)[:STATIC_BATCH]
        x_mb = train_images[idx]
        opt_ae.zero_grad()
        z = encoder(x_mb.unsqueeze(1)).squeeze(1)
        logits = decoder(z.unsqueeze(1)).squeeze(1)
        loss_recon = criterion_bce(logits, x_mb)
        loss = loss_recon + 1e-4 * z.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        opt_ae.step()
        sched_ae.step()
        if (epoch + 1) % 50 == 0:
            print(f"   AE Epoch {epoch + 1:3d}/{STATIC_EPOCHS} | Recon BCE = {loss_recon.item():.5f}")

    with torch.no_grad():
        z_train = encoder(train_images.unsqueeze(1)).squeeze(1)
        z_stats = {}
        p0_stats = {}
        for digit in DIGITS:
            mask = train_labels == digit
            z_digit = z_train[mask]
            p0_digit = train_data["positions"][mask, 0, :]
            z_stats[digit] = (z_digit.mean(dim=0), z_digit.var(dim=0) + 1e-4)
            p0_stats[digit] = (p0_digit.mean(dim=0), p0_digit.var(dim=0) + 1e-4)

        sanity_imgs = train_images[:5]
        sanity_z = encoder(sanity_imgs.unsqueeze(1)).squeeze(1)
        sanity_recon = decode_to_image(decoder(sanity_z.unsqueeze(1)).squeeze(1))

    fig_sanity, axes_sanity = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes_sanity[0, i].imshow(sanity_imgs[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_sanity[0, i].set_title("GT")
        axes_sanity[0, i].axis("off")
        axes_sanity[1, i].imshow(sanity_recon[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_sanity[1, i].set_title("AE")
        axes_sanity[1, i].axis("off")
    plt.tight_layout()
    fig_sanity.savefig("factorized_ct_node_appearance_sanity.png", dpi=150)
    plt.close(fig_sanity)
    print("   Saved 'factorized_ct_node_appearance_sanity.png'.")

    print("\n3. Training Continuous-Time Motion NODE...")
    motion_func = MotionODEFunc().to(device)
    opt_motion = optim.Adam(motion_func.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_motion = optim.lr_scheduler.CosineAnnealingLR(opt_motion, T_max=MOTION_EPOCHS, eta_min=1e-5)
    motion_batch = min(256, train_images.size(0))

    for epoch in range(MOTION_EPOCHS):
        idx = torch.randperm(train_images.size(0), device=device)[:motion_batch]
        pos_mb = train_data["positions"][idx]
        p0_mb = pos_mb[:, 0, :]
        opt_motion.zero_grad()
        pos_pred = odeint_rk4(motion_func, p0_mb, t_span).transpose(0, 1)
        loss_motion = F.mse_loss(pos_pred, pos_mb)
        loss_motion.backward()
        torch.nn.utils.clip_grad_norm_(motion_func.parameters(), 1.0)
        opt_motion.step()
        sched_motion.step()
        if (epoch + 1) % 50 == 0:
            print(f"   Motion Epoch {epoch + 1:3d}/{MOTION_EPOCHS} | Position MSE = {loss_motion.item():.5f}")

    print("\n4. Building Shared SPI Measurement Operator...")
    masks = torch.randint(0, 2, (T, M, H, W), device=device).float()
    mask_norm = build_mask_normalizer(masks)

    print("\n5. Online Reconstruction with Factorized Continuous-Time NODE...")
    results = {}
    criterion = nn.MSELoss()
    encoder.eval()
    decoder.eval()
    motion_func.eval()
    for p in list(encoder.parameters()) + list(decoder.parameters()) + list(motion_func.parameters()):
        p.requires_grad = False

    for digit in DIGITS:
        print(f"\n   Reconstructing digit {digit}...")
        test_idx = torch.where(test_labels == digit)[0][0]
        canonical_gt = test_data["canonical_images"][test_idx]
        positions_gt = test_data["positions"][test_idx]
        frames_gt = render_from_positions(canonical_gt, positions_gt)
        y_clean = forward_measurements(masks, frames_gt, mask_norm)
        noise_std = NOISE_LEVEL * y_clean.std()
        y_measured = y_clean + torch.randn_like(y_clean) * noise_std

        digit_mask = train_labels == digit
        digit_images = train_data["canonical_images"][digit_mask]
        digit_positions = train_data["positions"][digit_mask]
        topk_images, topk_positions, topk_losses = select_topk_exemplars(
            digit_images, digit_positions, masks, mask_norm, y_measured, TOPK_EXEMPLARS
        )
        with torch.no_grad():
            z_init = encoder(topk_images[0:1].unsqueeze(1)).squeeze(1).squeeze(0).clone()
        p0_init = topk_positions[0, 0, :].clone()

        z_mu, z_var = z_stats[digit]
        p0_mu, p0_var = p0_stats[digit]

        z_opt = nn.Parameter(z_init)
        p0_opt = nn.Parameter(p0_init)
        opt_recon = optim.Adam([z_opt, p0_opt], lr=1e-2)
        sched_recon = optim.lr_scheduler.CosineAnnealingLR(opt_recon, T_max=RECON_STEPS, eta_min=1e-4)

        best_total = float("inf")
        best_canonical = None
        best_frames = None
        best_positions = None

        for step in range(RECON_STEPS):
            opt_recon.zero_grad()
            canonical_logits = decoder(z_opt.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            canonical = decode_to_image(canonical_logits)
            positions_pred = odeint_rk4(motion_func, p0_opt, t_span)
            frames_pred = render_from_positions(canonical, positions_pred)
            y_est = forward_measurements(masks, frames_pred, mask_norm)

            loss_meas = criterion(y_est, y_measured)
            loss_z = (((z_opt - z_mu) ** 2) / z_var).mean()
            loss_p0 = (((p0_opt - p0_mu) ** 2) / p0_var).mean()
            loss_ex = exemplar_softmin_distance(canonical, topk_images, SOFTMIN_TEMP)
            loss_tv = total_variation(canonical)
            loss_sparse = canonical.mean()
            loss_total = (
                loss_meas
                + LAMBDA_Z * loss_z
                + LAMBDA_P0 * loss_p0
                + LAMBDA_EX * loss_ex
                + LAMBDA_TV * loss_tv
                + LAMBDA_SPARSE * loss_sparse
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([z_opt, p0_opt], 1.0)
            opt_recon.step()
            sched_recon.step()

            if loss_total.item() < best_total:
                best_total = loss_total.item()
                best_canonical = canonical.detach().clone()
                best_frames = frames_pred.detach().clone()
                best_positions = positions_pred.detach().clone()

            if (step + 1) % 100 == 0:
                print(
                    f"   Recon {step + 1:3d}/{RECON_STEPS} | Total={loss_total.item():.6f} "
                    f"Meas={loss_meas.item():.6f} Ex={loss_ex.item():.6f}"
                )

        meas_mse = criterion(forward_measurements(masks, best_frames, mask_norm), y_measured).item()
        frame_mse = criterion(best_frames, frames_gt).item()
        canon_mse = criterion(best_canonical, canonical_gt).item()
        pos_mse = criterion(best_positions, positions_gt).item()
        results[digit] = {
            "canonical_gt": canonical_gt.detach().cpu().numpy(),
            "canonical_recon": best_canonical.detach().cpu().numpy(),
            "gt": frames_gt.detach().cpu().numpy(),
            "recon": best_frames.detach().cpu().numpy(),
            "meas_mse": meas_mse,
            "frame_mse": frame_mse,
            "canon_mse": canon_mse,
            "pos_mse": pos_mse,
            "init_meas_mse": topk_losses[0].item(),
        }
        print(
            f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | "
            f"Frame MSE = {frame_mse:.6f} | Canonical MSE = {canon_mse:.6f} | "
            f"Position MSE = {pos_mse:.6f}"
        )

    print("\n6. Saving Comparison Figure...")
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
            f"meas={results[digit]['meas_mse']:.4g}\nframe={results[digit]['frame_mse']:.4g}\ncanon={results[digit]['canon_mse']:.4g}\npos={results[digit]['pos_mse']:.4g}",
            fontsize=8,
            va="center",
        )
    fig.suptitle("Factorized Continuous-Time NODE Reconstruction", fontsize=16)
    plt.tight_layout()
    fig.savefig("reconstruction_factorized_ct_node_t31.png", dpi=150)
    plt.close(fig)
    print("   Saved 'reconstruction_factorized_ct_node_t31.png'.")

    print("\n7. Saving Timelapse GIF...")
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

    def update(frame_idx):
        artists = []
        for row, digit in enumerate(DIGITS):
            im_gt, im_rec = im_pairs[row]
            im_gt.set_array(results[digit]["gt"][frame_idx])
            im_rec.set_array(results[digit]["recon"][frame_idx])
            artists.extend([im_gt, im_rec])
        fig_anim.suptitle(f"Factorized CT-NODE Timelapse - Frame {frame_idx + 1}/{T}", fontsize=16)
        return artists

    ani = animation.FuncAnimation(fig_anim, update, frames=T, interval=250, blit=False)
    ani.save("reconstruction_factorized_ct_node_t31.gif", writer="pillow")
    plt.close(fig_anim)
    print("   Saved 'reconstruction_factorized_ct_node_t31.gif'.")


if __name__ == "__main__":
    main()

"""
This is the original version.
We use one shared latent NODE to represent the entire dynamics (both shape and motion). The CT part works well but the reconstruction blurs the digits into a very ugly and incomprehensible blob.
Fits measurements well (numbers-wise good) but visually it wasn't doing a very good job reconstructing all digits.
"""

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

# ==========================================
# 0. Global Parameters
# ==========================================
H, W = 64, 64
T = 31
SPF = 0.25
M = max(1, int(H * W * SPF))
LATENT_DIM = 32
NOISE_LEVEL = 0.02
TRAIN_BATCH = 1024
MINI_BATCH = 16
DIGITS = list(range(10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_span = torch.linspace(0, 1, T).to(device)


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


# ==========================================
# 1. Data Loading
# ==========================================
print("1. Loading All-Digit Spiral MNIST Dataset...")


def render_spiral_sequence(img, T, H, W, phase=None, r_init=None):
    pad_h = (H - 28) // 2
    pad_w = (W - 28) // 2
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h))

    r_traj = 20.0
    omega = 2 * np.pi * 1.5
    gamma = 1.5
    t = torch.linspace(0, 1, T)

    if phase is None:
        phase = (torch.rand(1) * 2 * np.pi).item()
    if r_init is None:
        r_init = (r_traj + (torch.randn(1) * 2.0)).item()

    frames = []
    for t_step in t:
        r_t = r_init * torch.exp(-gamma * t_step)
        angle_t = omega * t_step + phase
        tx = r_t * torch.cos(angle_t)
        ty = r_t * torch.sin(angle_t)
        translated_img = TF.affine(
            img_padded,
            angle=0.0,
            translate=[float(tx), float(ty)],
            scale=1.0,
            shear=0.0,
        )
        frames.append(translated_img.squeeze(0))

    return torch.stack(frames, dim=0)


def generate_spiral_mnist_dataset(num_samples, T, H, W, digits=DIGITS):
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

    seqs = []
    all_targets = dataset.targets
    base_count = num_samples // len(digits)
    remainder = num_samples % len(digits)

    for digit in digits:
        digit_indices = torch.where(all_targets == digit)[0]
        take = base_count + (1 if digit < remainder else 0)
        print(f"   Found {len(digit_indices)} samples of digit '{digit}', taking {take}")

        perm = torch.randperm(len(digit_indices))[:take]
        selected_indices = digit_indices[perm]

        for idx in selected_indices:
            img, _ = dataset[idx.item()]
            seqs.append(render_spiral_sequence(img, T, H, W))

    return torch.stack(seqs, dim=0).to(device)


x_train_gt = generate_spiral_mnist_dataset(TRAIN_BATCH, T, H, W)


def generate_test_samples(T, H, W, digits=DIGITS):
    dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    all_targets = dataset.targets
    samples = {}
    for digit in digits:
        digit_indices = torch.where(all_targets == digit)[0]
        img, _ = dataset[digit_indices[0].item()]
        samples[digit] = render_spiral_sequence(img, T, H, W, phase=0.0, r_init=20.0).to(device)
    return samples


x_test_gt_by_digit = generate_test_samples(T, H, W)

print(f"   Train: {x_train_gt.shape}, Test digits: {len(x_test_gt_by_digit)}")

# ==========================================
# 2. Forward Model
# ==========================================
print(f"2. Forward Model: SPF={SPF * 100}%, M={M}")
A = torch.randint(0, 2, (T, M, H, W)).float().to(device)
mask_norm = build_mask_normalizer(A)


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

    def forward(self, x):
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

        def res_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1),
                nn.GroupNorm(4, c),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, 1, 1),
                nn.GroupNorm(4, c),
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


class LatentODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, LATENT_DIM),
        )

    def forward(self, t, z):
        del t
        return self.net(z)


encoder = Encoder2D().to(device)
decoder = Decoder2D().to(device)
ode_func = LatentODEFunc().to(device)


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


# ==========================================
# 4. Offline Training
# ==========================================
print("\n4. Offline Training (Scheme A: Decoupled AE and ODE)...")

criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
_AMP = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Phase 4.1: Static AE Pre-training
# ---------------------------------------------------------
print("   [Phase 4.1] Static AE Pre-training (Consistent BCE + Sigmoid Decoding)...")
x_train_static = x_train_gt.reshape(-1, H, W)
STATIC_BATCH = 256
STATIC_EPOCHS = 500

opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
sched_ae = optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=STATIC_EPOCHS, eta_min=1e-5)

for epoch in range(STATIC_EPOCHS):
    idx = torch.randperm(x_train_static.size(0), device=device)[:STATIC_BATCH]
    x_mb = x_train_static[idx]

    opt_ae.zero_grad()
    with torch.amp.autocast(device_type=_AMP, enabled=torch.cuda.is_available()):
        z = encoder(x_mb.unsqueeze(1)).squeeze(1)
        logits = decoder(z.unsqueeze(1)).squeeze(1)
        loss_recon = criterion_bce(logits, x_mb)
        loss_latent_reg = 1e-4 * torch.mean(z**2)
        loss = loss_recon + loss_latent_reg

    scaler.scale(loss).backward()
    scaler.unscale_(opt_ae)
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
    scaler.step(opt_ae)
    scaler.update()
    sched_ae.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"      AE Epoch {epoch + 1:3d}/{STATIC_EPOCHS} | "
            f"Recon BCE = {loss_recon.item():.5f} | Total = {loss.item():.5f}"
        )

# ==========================================
# Sanity Check: Static AE Reconstruction
# ==========================================
with torch.no_grad():
    test_static_img = x_train_static[:5]
    test_z = encoder(test_static_img.unsqueeze(1)).squeeze(1)
    test_recon = decode_to_image(decoder(test_z.unsqueeze(1)).squeeze(1))

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(test_static_img[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title("GT")
        axes[0, i].axis("off")
        axes[1, i].imshow(test_recon[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("AE Recon")
        axes[1, i].axis("off")
    plt.suptitle("Sanity Check: Static AE Reconstruction")
    plt.tight_layout()
    plt.savefig("ae_sanity_check.png")
    plt.close()
print("   [Sanity Check] Saved 'ae_sanity_check.png'.")

# ---------------------------------------------------------
# Phase 4.2: Dynamic ODE Training
# ---------------------------------------------------------
print("\n   [Phase 4.2] Dynamic ODE Training (Frozen AE Manifold)...")

encoder.eval()
decoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
for p in decoder.parameters():
    p.requires_grad = False

opt_dyn = optim.Adam(ode_func.parameters(), lr=1e-3, weight_decay=1e-5)
sched_dyn = optim.lr_scheduler.CosineAnnealingLR(opt_dyn, T_max=700, eta_min=1e-5)

DYN_EPOCHS = 700
for epoch in range(DYN_EPOCHS):
    idx = torch.randperm(TRAIN_BATCH, device=device)[:MINI_BATCH]
    x_mb = x_train_gt[idx]

    opt_dyn.zero_grad()
    with torch.amp.autocast(device_type=_AMP, enabled=torch.cuda.is_available()):
        with torch.no_grad():
            z_seq = encoder(x_mb)

        z0 = z_seq[:, 0, :]
        z_pred = odeint_rk4(ode_func, z0, t_span).transpose(0, 1)

        loss_latent = criterion_mse(z_pred, z_seq)

        logits_pred = decoder(z_pred)
        x_pred = decode_to_image(logits_pred)
        loss_pred_img = criterion_mse(x_pred, x_mb)

        loss = loss_latent * 10.0 + loss_pred_img

    scaler.scale(loss).backward()
    scaler.unscale_(opt_dyn)
    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
    scaler.step(opt_dyn)
    scaler.update()
    sched_dyn.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"      DYN Epoch {epoch + 1:3d}/{DYN_EPOCHS} | Total = {loss.item():.5f} "
            f"(Img_Pred = {loss_pred_img.item():.5f}, Latent = {loss_latent.item():.5f})"
        )

print("   Offline Training Complete.")

# ==========================================
# 5. Online Reconstruction
# ==========================================
print("\n5. Online Reconstruction (Eq. 7) with Smart Init for All Digits...")
for model in [encoder, decoder, ode_func]:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

def reconstruct_sequence(y_measured):
    print("   Searching for best initialization candidate...")
    best_init_loss = float("inf")
    best_z_init = None

    with torch.no_grad():
        for i in range(0, 256, MINI_BATCH):
            x_cand = x_train_gt[i : i + MINI_BATCH]
            z_cand = encoder(x_cand)
            z0_cand = z_cand[:, 0, :]
            z_rollout = odeint_rk4(ode_func, z0_cand, t_span).transpose(0, 1)
            x_est = decode_to_image(decoder(z_rollout))
            y_est = forward_measurements(A, x_est, mask_norm)
            losses = ((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2))

            local_best = torch.argmin(losses)
            local_best_loss = losses[local_best].item()
            if local_best_loss < best_init_loss:
                best_init_loss = local_best_loss
                best_z_init = z_cand[local_best].clone()

    print(f"   Found init candidate with Measurement MSE: {best_init_loss:.6f}")

    print("   Phase 1: Optimizing z0 (Strict ODE Manifold)...")
    z0_opt = nn.Parameter(best_z_init[0].clone())
    opt_z0 = optim.Adam([z0_opt], lr=1e-2)
    sched_z0 = optim.lr_scheduler.CosineAnnealingLR(opt_z0, T_max=200, eta_min=1e-4)

    for _ in range(200):
        opt_z0.zero_grad()
        z_ode = odeint_rk4(ode_func, z0_opt, t_span)
        x_est = decode_to_image(decoder(z_ode.unsqueeze(0))).squeeze(0)
        y_est = forward_measurements(A, x_est, mask_norm)
        loss_meas = criterion_mse(y_est, y_measured)

        loss_meas.backward()
        opt_z0.step()
        sched_z0.step()

    print("   Phase 2: Optimizing full Z sequence (Eq. 7)...")
    with torch.no_grad():
        z_opt_init = odeint_rk4(ode_func, z0_opt.detach(), t_span)

    z_opt = nn.Parameter(z_opt_init.clone())
    opt_recon = optim.Adam([z_opt], lr=5e-3)
    sched_recon = optim.lr_scheduler.CosineAnnealingLR(opt_recon, T_max=300, eta_min=1e-5)

    lambda_ode = 1.0
    best_recon_loss = float("inf")
    best_z = z_opt.detach().clone()

    for epoch in range(300):
        opt_recon.zero_grad()

        x_est = decode_to_image(decoder(z_opt.unsqueeze(0))).squeeze(0)
        y_est = forward_measurements(A, x_est, mask_norm)
        loss_meas = criterion_mse(y_est, y_measured)

        z_ode = odeint_rk4(ode_func, z_opt[0], t_span)
        loss_ode = criterion_mse(z_opt, z_ode)

        loss_total = loss_meas + lambda_ode * loss_ode

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([z_opt], 0.5)
        opt_recon.step()
        sched_recon.step()

        if loss_total.item() < best_recon_loss:
            best_recon_loss = loss_total.item()
            best_z = z_opt.detach().clone()

        if (epoch + 1) % 100 == 0:
            print(
                f"   Recon {epoch + 1:3d}/300 | Total={loss_total.item():.6f} "
                f"Meas={loss_meas.item():.6f} ODE={loss_ode.item():.6f}"
            )

    with torch.no_grad():
        x_recon = decode_to_image(decoder(best_z.unsqueeze(0))).squeeze(0)
    return x_recon

# ==========================================
# 6. Visualization
# ==========================================
results = {}

for digit in DIGITS:
    print(f"\n   Reconstructing digit {digit}...")
    x_test_gt = x_test_gt_by_digit[digit]
    y_clean = forward_measurements(A, x_test_gt, mask_norm)
    noise_std = NOISE_LEVEL * y_clean.std()
    y_measured = y_clean + torch.randn_like(y_clean) * noise_std

    x_recon_t = reconstruct_sequence(y_measured)
    y_final = forward_measurements(A, x_recon_t, mask_norm)
    meas_mse = criterion_mse(y_final, y_measured).item()
    img_mse = criterion_mse(x_recon_t, x_test_gt).item()

    results[digit] = {
        "gt": x_test_gt.detach().cpu().numpy(),
        "recon": x_recon_t.detach().cpu().numpy(),
        "meas_mse": meas_mse,
        "img_mse": img_mse,
    }
    print(f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | Image MSE = {img_mse:.6f}")

print("\nFinal Results by Digit:")
for digit in DIGITS:
    print(
        f"   Digit {digit}: Measurement MSE = {results[digit]['meas_mse']:.6f}, "
        f"Image MSE = {results[digit]['img_mse']:.6f}"
    )

frames_to_show = [0, T // 2, T - 1]
fig_static, axes_static = plt.subplots(len(DIGITS) * 2, len(frames_to_show), figsize=(10, 2.4 * len(DIGITS)))
for row, digit in enumerate(DIGITS):
    x_gt = results[digit]["gt"]
    x_recon = results[digit]["recon"]
    for col, frame_idx in enumerate(frames_to_show):
        axes_static[2 * row, col].imshow(x_gt[frame_idx], cmap="gray", vmin=0, vmax=1)
        axes_static[2 * row, col].set_title(f"{digit} GT t={frame_idx}")
        axes_static[2 * row, col].axis("off")
        axes_static[2 * row + 1, col].imshow(x_recon[frame_idx], cmap="gray", vmin=0, vmax=1)
        axes_static[2 * row + 1, col].set_title(f"{digit} Recon t={frame_idx}")
        axes_static[2 * row + 1, col].axis("off")
fig_static.suptitle(f"All-Digit SPI-NODE Reconstruction (SPF={SPF * 100:.0f}%)", fontsize=14)
plt.tight_layout()
fig_static.savefig("reconstruction_all_digits_t31.png", dpi=150)
plt.close(fig_static)

print("Saved 'reconstruction_all_digits_t31.png'.")

print("Saving 'reconstruction_all_digits_t31.gif'...")
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


def update_all_digits(frame_idx: int):
    artists = []
    for row, digit in enumerate(DIGITS):
        im_gt, im_rec = im_pairs[row]
        im_gt.set_array(results[digit]["gt"][frame_idx])
        im_rec.set_array(results[digit]["recon"][frame_idx])
        artists.extend([im_gt, im_rec])
    fig_anim.suptitle(f"All-Digit SPI-NODE - Frame {frame_idx + 1}/{T}", fontsize=16)
    return artists


ani_all = animation.FuncAnimation(fig_anim, update_all_digits, frames=T, interval=250, blit=False)
ani_all.save("reconstruction_all_digits_t31.gif", writer="pillow")
plt.close(fig_anim)
print("Saved 'reconstruction_all_digits_t31.gif'.")

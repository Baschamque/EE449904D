"""
This is a version of the entangled method. We introduce an additional method of metric-learning style latent representation. I did this to try to separate the different digit classes so that they would be hopefully more strongly separated in the entangled latent space.
We use a more discriminative latent geometry so that the reconstruction would stop showing the class-average problem (like all the other attempts, but this one is conceptually stronger.
Latent organization improved a little but reconstruction quality is still lower than the non-entangled methods (and MSE is somehow much much higher than the other three methods by quite a big margin).
Ultimately this and the other three previous techniques showed that using one latent CT model to explain both shape and form is difficult.
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
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.datasets import MNIST


H, W = 64, 64
T = 31
SPF = 0.25
M = max(1, int(H * W * SPF))
LATENT_DIM = 48
FEATURE_DIM = 128
NOISE_LEVEL = 0.02
TRAIN_SEQS = 1024
MINI_BATCH = 40
EPOCHS = 600
DIGITS = list(range(10))

LAMBDA_RECON_DIRECT = 1.0
LAMBDA_RECON_ROLL = 1.0
LAMBDA_LATENT = 0.5
LAMBDA_CLASS = 0.2
LAMBDA_REG = 1e-4
LAMBDA_METRIC = 0.2

PROTO_RECON_STEPS = 300
LAMBDA_PROTO = 1e-3
LAMBDA_CLASS_RECON = 0.05
LAMBDA_TV = 5e-4
LAMBDA_SPARSE = 5e-4
METRIC_MARGIN = 0.8

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


def total_variation_batch(frames: torch.Tensor) -> torch.Tensor:
    return (
        (frames[..., 1:, :] - frames[..., :-1, :]).abs().mean()
        + (frames[..., :, 1:] - frames[..., :, :-1]).abs().mean()
    )


def render_spiral_sequence(img: torch.Tensor, phase: float | None = None, r_init: float | None = None) -> torch.Tensor:
    pad_h = (H - 28) // 2
    pad_w = (W - 28) // 2
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h))

    omega = 2 * math.pi * 1.5
    gamma = 1.5
    base_r = 20.0
    t = torch.linspace(0.0, 1.0, T)

    if phase is None:
        phase = (torch.rand(1) * 2 * math.pi).item()
    if r_init is None:
        r_init = (base_r + 2.0 * torch.randn(1)).item()

    frames = []
    for t_step in t:
        r_t = r_init * torch.exp(-gamma * t_step)
        angle_t = omega * t_step + phase
        tx = r_t * torch.cos(angle_t)
        ty = r_t * torch.sin(angle_t)
        translated = TF.affine(
            img_padded,
            angle=0.0,
            translate=[float(tx), float(ty)],
            scale=1.0,
            shear=0.0,
        )
        frames.append(translated.squeeze(0))
    return torch.stack(frames, dim=0)


def generate_spiral_mnist_dataset(num_samples: int, digits: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    targets = dataset.targets
    seqs = []
    labels = []

    base_count = num_samples // len(digits)
    remainder = num_samples % len(digits)
    for digit in digits:
        digit_indices = torch.where(targets == digit)[0]
        take = base_count + (1 if digit < remainder else 0)
        print(f"   Found {len(digit_indices)} train samples of digit '{digit}', taking {take}")
        perm = torch.randperm(len(digit_indices))[:take]
        for idx in digit_indices[perm]:
            img, _ = dataset[idx.item()]
            seqs.append(render_spiral_sequence(img))
            labels.append(digit)
    return torch.stack(seqs, dim=0).to(device), torch.tensor(labels, device=device)


def generate_test_sequences(digits: list[int]) -> dict[int, torch.Tensor]:
    dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    targets = dataset.targets
    samples = {}
    for digit in digits:
        digit_indices = torch.where(targets == digit)[0]
        img, _ = dataset[digit_indices[0].item()]
        samples[digit] = render_spiral_sequence(img, phase=0.0, r_init=20.0).to(device)
    return samples


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(4, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(4, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1), nn.GroupNorm(4, 128), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, FEATURE_DIM),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4), nn.SiLU(),
        )

        def res_block(c: int) -> nn.Sequential:
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


class LatentODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, LATENT_DIM),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(z)


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


class SequenceNODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.gru = nn.GRU(FEATURE_DIM, FEATURE_DIM, batch_first=True)
        self.frame_latent_head = nn.Linear(FEATURE_DIM, LATENT_DIM)
        self.z0_head = nn.Linear(FEATURE_DIM, LATENT_DIM)
        self.class_head = nn.Linear(LATENT_DIM, len(DIGITS))
        self.decoder = Decoder2D()
        self.ode_func = LatentODEFunc()

    def encode_sequence_features(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = frames.shape
        feats = self.frame_encoder(frames.reshape(-1, 1, H, W))
        return feats.reshape(b, t, FEATURE_DIM)

    def encode_sequence(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_sequence_features(frames)
        encoded_latents = self.frame_latent_head(features)
        _, h_n = self.gru(features)
        z0 = self.z0_head(h_n[-1])
        return encoded_latents, z0

    def rollout(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return odeint_rk4(self.ode_func, z0, times).transpose(0, 1)

    def forward(self, frames: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_latents, z0 = self.encode_sequence(frames)
        rollout_latents = self.rollout(z0, times)
        direct_logits = self.decoder(encoded_latents)
        rollout_logits = self.decoder(rollout_latents)
        z0_logits = self.class_head(z0)
        rollout_class_logits = self.class_head(rollout_latents.reshape(-1, LATENT_DIM)).reshape(
            rollout_latents.shape[0], rollout_latents.shape[1], len(DIGITS)
        )
        return {
            "encoded_latents": encoded_latents,
            "z0": z0,
            "rollout_latents": rollout_latents,
            "direct_logits": direct_logits,
            "rollout_logits": rollout_logits,
            "z0_logits": z0_logits,
            "rollout_class_logits": rollout_class_logits,
        }


def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + 0.05 * F.l1_loss(
        decode_to_image(logits), targets
    )


def metric_loss(z: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    z_norm = F.normalize(z, dim=1)
    distances = torch.cdist(z_norm, z_norm, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff = ~same
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    same = same & ~eye

    positive = distances[same]
    negative = distances[diff]
    if positive.numel() == 0 or negative.numel() == 0:
        return torch.tensor(0.0, device=labels.device)

    pos_loss = positive.pow(2).mean()
    neg_loss = F.relu(margin - negative).pow(2).mean()
    return pos_loss + neg_loss


print("1. Loading All-Digit Spiral MNIST Dataset...")
x_train_gt, y_train = generate_spiral_mnist_dataset(TRAIN_SEQS, DIGITS)
x_test_gt_by_digit = generate_test_sequences(DIGITS)
print(f"   Train: {x_train_gt.shape}, Test digits: {len(x_test_gt_by_digit)}")

print("\n2. Training Metric-Learning Entangled NODE...")
model = SequenceNODEModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

for epoch in range(EPOCHS):
    idx = torch.randperm(TRAIN_SEQS, device=device)[:MINI_BATCH]
    x_mb = x_train_gt[idx]
    y_mb = y_train[idx]

    outputs = model(x_mb, t_span)
    direct_logits = outputs["direct_logits"]
    rollout_logits = outputs["rollout_logits"]
    encoded_latents = outputs["encoded_latents"]
    rollout_latents = outputs["rollout_latents"]

    loss_direct = reconstruction_loss(direct_logits, x_mb)
    loss_roll = reconstruction_loss(rollout_logits, x_mb)
    loss_latent = F.mse_loss(rollout_latents, encoded_latents.detach())
    loss_class = F.cross_entropy(outputs["z0_logits"], y_mb)
    repeated_labels = y_mb.unsqueeze(1).expand(-1, T).reshape(-1)
    loss_class = loss_class + F.cross_entropy(
        outputs["rollout_class_logits"].reshape(-1, len(DIGITS)),
        repeated_labels,
    )
    loss_metric = metric_loss(outputs["z0"], y_mb, METRIC_MARGIN)
    loss_reg = outputs["z0"].pow(2).mean() + encoded_latents.pow(2).mean()

    loss = (
        LAMBDA_RECON_DIRECT * loss_direct
        + LAMBDA_RECON_ROLL * loss_roll
        + LAMBDA_LATENT * loss_latent
        + LAMBDA_CLASS * loss_class
        + LAMBDA_METRIC * loss_metric
        + LAMBDA_REG * loss_reg
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            pred_labels = outputs["z0_logits"].argmax(dim=1)
            acc = (pred_labels == y_mb).float().mean().item()
        print(
            f"   Epoch {epoch + 1:3d}/{EPOCHS} | Total={loss.item():.5f} "
            f"Direct={loss_direct.item():.5f} Roll={loss_roll.item():.5f} "
            f"Lat={loss_latent.item():.5f} Cls={loss_class.item():.5f} "
            f"Met={loss_metric.item():.5f} Acc={acc:.3f}"
        )

with torch.no_grad():
    z0_train = []
    for i in range(0, TRAIN_SEQS, MINI_BATCH):
        _, z0_batch = model.encode_sequence(x_train_gt[i:i + MINI_BATCH])
        z0_train.append(z0_batch)
    z0_train = torch.cat(z0_train, dim=0)
    class_stats = {}
    for digit in DIGITS:
        z_digit = z0_train[y_train == digit]
        class_stats[digit] = (z_digit.mean(dim=0), z_digit.var(dim=0) + 1e-4)

    sanity_imgs = x_train_gt[:2, [0, T // 2, T - 1]]
    sanity_roll = decode_to_image(model.decoder(model.rollout(z0_train[:2], t_span)[:, [0, T // 2, T - 1]]))

fig_sanity, axes_sanity = plt.subplots(4, 3, figsize=(6, 8))
for row in range(2):
    for col in range(3):
        axes_sanity[2 * row, col].imshow(sanity_imgs[row, col].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_sanity[2 * row, col].set_title("GT")
        axes_sanity[2 * row, col].axis("off")
        axes_sanity[2 * row + 1, col].imshow(
            sanity_roll[row, col].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes_sanity[2 * row + 1, col].set_title("Rollout")
        axes_sanity[2 * row + 1, col].axis("off")
plt.tight_layout()
fig_sanity.savefig("entangled_metric_sanity.png", dpi=150)
plt.close(fig_sanity)
print("   Saved 'entangled_metric_sanity.png'.")

print("\n3. Shared SPI Measurement Operator...")
A = torch.randint(0, 2, (T, M, H, W), device=device).float()
mask_norm = build_mask_normalizer(A)

print("\n4. Online Reconstruction with Metric-Learning Entangled NODE...")
results = {}
model.eval()
for p in model.parameters():
    p.requires_grad = False

with torch.no_grad():
    z0_candidates = []
    class_candidates = []
    for i in range(0, TRAIN_SEQS, MINI_BATCH):
        x_cand = x_train_gt[i:i + MINI_BATCH]
        _, z0_cand = model.encode_sequence(x_cand)
        z0_candidates.append(z0_cand)
        class_candidates.append(y_train[i:i + MINI_BATCH])
    z0_candidates = torch.cat(z0_candidates, dim=0)
    class_candidates = torch.cat(class_candidates, dim=0)

for digit in DIGITS:
    print(f"\n   Reconstructing digit {digit}...")
    x_test_gt = x_test_gt_by_digit[digit]
    y_clean = forward_measurements(A, x_test_gt, mask_norm)
    noise_std = NOISE_LEVEL * y_clean.std()
    y_measured = y_clean + torch.randn_like(y_clean) * noise_std

    best_init_loss = float("inf")
    best_z0 = None
    best_class = None

    with torch.no_grad():
        for i in range(0, z0_candidates.size(0), MINI_BATCH):
            z0_cand = z0_candidates[i:i + MINI_BATCH]
            z_roll = model.rollout(z0_cand, t_span)
            x_est = decode_to_image(model.decoder(z_roll))
            y_est = forward_measurements(A, x_est, mask_norm)
            losses = ((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            local_idx = torch.argmin(losses)
            local_loss = losses[local_idx].item()
            if local_loss < best_init_loss:
                best_init_loss = local_loss
                best_z0 = z0_cand[local_idx].clone()
                best_class = int(class_candidates[i + local_idx].item())

    z_mu, z_var = class_stats[best_class]
    z0_opt = nn.Parameter(best_z0.clone())
    opt_z = optim.Adam([z0_opt], lr=1e-2)
    sched_z = optim.lr_scheduler.CosineAnnealingLR(opt_z, T_max=PROTO_RECON_STEPS, eta_min=1e-4)
    best_total = float("inf")
    best_frames = None

    for step in range(PROTO_RECON_STEPS):
        opt_z.zero_grad()
        z_roll = model.rollout(z0_opt.unsqueeze(0), t_span).squeeze(0)
        logits = model.decoder(z_roll.unsqueeze(0)).squeeze(0)
        x_est = decode_to_image(logits)
        y_est = forward_measurements(A, x_est, mask_norm)

        loss_meas = F.mse_loss(y_est, y_measured)
        loss_proto = (((z0_opt - z_mu) ** 2) / z_var).mean()
        class_targets = torch.full((T,), best_class, dtype=torch.long, device=device)
        loss_class = F.cross_entropy(model.class_head(z_roll), class_targets)
        loss_tv = total_variation_batch(x_est)
        loss_sparse = x_est.mean()
        loss_total = (
            loss_meas
            + LAMBDA_PROTO * loss_proto
            + LAMBDA_CLASS_RECON * loss_class
            + LAMBDA_TV * loss_tv
            + LAMBDA_SPARSE * loss_sparse
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([z0_opt], 1.0)
        opt_z.step()
        sched_z.step()

        if loss_total.item() < best_total:
            best_total = loss_total.item()
            best_frames = x_est.detach().clone()

        if (step + 1) % 100 == 0:
            print(
                f"   Recon {step + 1:3d}/300 | Total={loss_total.item():.6f} "
                f"Meas={loss_meas.item():.6f} Proto={loss_proto.item():.6f} Cls={loss_class.item():.6f}"
            )

    meas_mse = F.mse_loss(forward_measurements(A, best_frames, mask_norm), y_measured).item()
    img_mse = F.mse_loss(best_frames, x_test_gt).item()
    pred_class = int(model.class_head(z0_opt.unsqueeze(0)).argmax(dim=1).item())
    results[digit] = {
        "gt": x_test_gt.detach().cpu().numpy(),
        "recon": best_frames.detach().cpu().numpy(),
        "meas_mse": meas_mse,
        "img_mse": img_mse,
        "init_class": best_class,
        "final_class": pred_class,
    }
    print(
        f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | "
        f"Image MSE = {img_mse:.6f} | Init class = {best_class} | Final class = {pred_class}"
    )

print("\n5. Saving Comparison Figure...")
frames_to_show = [0, T // 2, T - 1]
fig, axes = plt.subplots(len(DIGITS) * 2, len(frames_to_show), figsize=(10, 2.4 * len(DIGITS)))
for row, digit in enumerate(DIGITS):
    x_gt = results[digit]["gt"]
    x_rec = results[digit]["recon"]
    for col, frame_idx in enumerate(frames_to_show):
        axes[2 * row, col].imshow(x_gt[frame_idx], cmap="gray", vmin=0, vmax=1)
        axes[2 * row, col].set_title(f"{digit} GT t={frame_idx}")
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].imshow(x_rec[frame_idx], cmap="gray", vmin=0, vmax=1)
        axes[2 * row + 1, col].set_title(f"{digit} Rec t={frame_idx}")
        axes[2 * row + 1, col].axis("off")
fig.suptitle("Metric-Learning Entangled Continuous-Time NODE Reconstruction", fontsize=14)
plt.tight_layout()
fig.savefig("reconstruction_entangled_metric_t31.png", dpi=150)
plt.close(fig)
print("   Saved 'reconstruction_entangled_metric_t31.png'.")

print("   Saving 'reconstruction_entangled_metric_t31.gif'...")
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


def update_metric(frame_idx: int):
    artists = []
    for row, digit in enumerate(DIGITS):
        im_gt, im_rec = im_pairs[row]
        im_gt.set_array(results[digit]["gt"][frame_idx])
        im_rec.set_array(results[digit]["recon"][frame_idx])
        artists.extend([im_gt, im_rec])
    fig_anim.suptitle(f"Entangled NODE + Metric Learning - Frame {frame_idx + 1}/{T}", fontsize=16)
    return artists


ani = animation.FuncAnimation(fig_anim, update_metric, frames=T, interval=250, blit=False)
ani.save("reconstruction_entangled_metric_t31.gif", writer="pillow")
plt.close(fig_anim)
print("   Saved 'reconstruction_entangled_metric_t31.gif'.")

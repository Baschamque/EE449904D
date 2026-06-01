"""
This a version of the factorized method. Using the strongest appearance prior we got from the exemplar model, we put this into the factorized CT NODE.
Basically nothing is made explicitly now, the shape is handled using exemplars and the motion is handled using our CT NODE.

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
TRAIN_PER_DIGIT = 256
TEST_PER_DIGIT = 1
MOTION_TRAIN_SAMPLES = 2048
MOTION_BATCH = 256
MOTION_EPOCHS = 500
TOPK_EXEMPLARS = 16
RECON_STEPS = 400
NOISE_LEVEL = 0.02
DIGITS = list(range(10))

OMEGA = 2 * math.pi * 1.5
GAMMA = 1.5
R_MEAN = 20.0
R_STD = 2.0
DEFAULT_PHASE = 0.0
DEFAULT_R = 20.0

LAMBDA_EXEMPLAR = 2.0
LAMBDA_RESID = 5e-3
LAMBDA_STATE = 1e-3
LAMBDA_TV = 5e-4
LAMBDA_SPARSE = 5e-4
SOFTMIN_TEMP = 0.002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_span = torch.linspace(0.0, 1.0, T, device=device)


def pad_digit(img: torch.Tensor) -> torch.Tensor:
    pad_h = (H - 28) // 2
    pad_w = (W - 28) // 2
    return F.pad(img, (pad_w, pad_w, pad_h, pad_h))


def inverse_sigmoid(image: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    image = image.clamp(eps, 1.0 - eps)
    return torch.log(image / (1.0 - image))


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
    return differentiable_translate(image, positions[:, 0], positions[:, 1])


def spiral_state_from_params(phase: torch.Tensor, r_init: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    times = times.unsqueeze(0)
    r_t = r_init.unsqueeze(-1) * torch.exp(-GAMMA * times)
    dr_t = -GAMMA * r_t
    angle_t = OMEGA * times + phase.unsqueeze(-1)
    cos_t = torch.cos(angle_t)
    sin_t = torch.sin(angle_t)
    x = r_t * cos_t
    y = r_t * sin_t
    vx = dr_t * cos_t - OMEGA * r_t * sin_t
    vy = dr_t * sin_t + OMEGA * r_t * cos_t
    return torch.stack([x, y, vx, vy], dim=-1)


def render_spiral_from_canonical(image: torch.Tensor, phase: torch.Tensor, r_init: torch.Tensor) -> torch.Tensor:
    states = spiral_state_from_params(phase.unsqueeze(0), r_init.unsqueeze(0), t_span).squeeze(0)
    return render_from_positions(image, states[:, :2])


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


def sample_motion_dataset(num_samples: int) -> torch.Tensor:
    phase = torch.rand(num_samples, device=device) * 2 * math.pi
    r_init = R_MEAN + R_STD * torch.randn(num_samples, device=device)
    return spiral_state_from_params(phase, r_init, t_span)


class MotionODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 128
        self.net = nn.Sequential(
            nn.Linear(4, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 4),
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(state)


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


def select_topk_exemplars(
    exemplars: torch.Tensor,
    state_traj: torch.Tensor,
    masks: torch.Tensor,
    mask_norm: torch.Tensor,
    y_measured: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 32
    losses = []
    for i in range(0, exemplars.size(0), batch_size):
        batch = exemplars[i:i + batch_size]
        frames = []
        for j in range(batch.size(0)):
            frames.append(render_from_positions(batch[j], state_traj[:, :2]))
        frame_batch = torch.stack(frames, dim=0)
        y_est = forward_measurements(masks, frame_batch, mask_norm)
        losses.append(((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2)))
    losses = torch.cat(losses, dim=0)
    values, indices = torch.topk(losses, k=min(topk, losses.numel()), largest=False)
    return exemplars[indices], values


def main():
    print("1. Loading Centered MNIST Appearance Bank...")
    train_images, train_labels = load_centered_mnist(train=True, per_digit=TRAIN_PER_DIGIT)
    test_images, test_labels = load_centered_mnist(train=False, per_digit=TEST_PER_DIGIT)
    print(f"   Train bank: {train_images.shape}, Test images: {test_images.shape}")

    print("\n2. Training Continuous-Time Motion NODE...")
    motion_train = sample_motion_dataset(MOTION_TRAIN_SAMPLES)
    motion_func = MotionODEFunc().to(device)
    opt_motion = optim.Adam(motion_func.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_motion = optim.lr_scheduler.CosineAnnealingLR(opt_motion, T_max=MOTION_EPOCHS, eta_min=1e-5)

    for epoch in range(MOTION_EPOCHS):
        idx = torch.randperm(MOTION_TRAIN_SAMPLES, device=device)[:MOTION_BATCH]
        states_mb = motion_train[idx]
        s0_mb = states_mb[:, 0, :]
        opt_motion.zero_grad()
        states_pred = odeint_rk4(motion_func, s0_mb, t_span).transpose(0, 1)
        loss_motion = F.mse_loss(states_pred, states_mb)
        loss_motion.backward()
        torch.nn.utils.clip_grad_norm_(motion_func.parameters(), 1.0)
        opt_motion.step()
        sched_motion.step()
        if (epoch + 1) % 50 == 0:
            print(f"   Motion Epoch {epoch + 1:3d}/{MOTION_EPOCHS} | State MSE = {loss_motion.item():.6f}")

    with torch.no_grad():
        s0_all = motion_train[:, 0, :]
        s0_mu = s0_all.mean(dim=0)
        s0_var = s0_all.var(dim=0) + 1e-4

    print("\n3. Building Dynamic SPI Measurement Operator...")
    masks = torch.randint(0, 2, (T, M, H, W), device=device).float()
    mask_norm = build_mask_normalizer(masks)

    print("\n4. Hybrid Exemplar Appearance + CT-NODE Motion Reconstruction...")
    results = {}
    motion_func.eval()
    for p in motion_func.parameters():
        p.requires_grad = False

    gt_state = spiral_state_from_params(
        torch.tensor([DEFAULT_PHASE], device=device),
        torch.tensor([DEFAULT_R], device=device),
        t_span,
    ).squeeze(0)

    for digit in DIGITS:
        print(f"\n   Reconstructing digit {digit}...")
        gt_image = test_images[test_labels == digit][0]
        gt_frames = render_from_positions(gt_image, gt_state[:, :2])
        y_clean = forward_measurements(masks, gt_frames, mask_norm)
        noise_std = NOISE_LEVEL * y_clean.std()
        y_measured = y_clean + torch.randn_like(y_clean) * noise_std

        exemplars = train_images[train_labels == digit]
        topk_exemplars, init_losses = select_topk_exemplars(exemplars, gt_state, masks, mask_norm, y_measured, TOPK_EXEMPLARS)

        alpha_logits = nn.Parameter(torch.zeros(topk_exemplars.size(0), device=device))
        residual = nn.Parameter(torch.zeros(H, W, device=device))
        s0_opt = nn.Parameter(gt_state[0].clone())
        optimizer = optim.Adam([alpha_logits, residual, s0_opt], lr=5e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECON_STEPS, eta_min=5e-4)

        best_total = float("inf")
        best_image = None
        best_frames = None
        best_init = None

        for step in range(RECON_STEPS):
            optimizer.zero_grad()
            weights = torch.softmax(alpha_logits, dim=0)
            base_image = (weights.view(-1, 1, 1) * topk_exemplars).sum(dim=0)
            canonical = torch.sigmoid(inverse_sigmoid(base_image) + residual)
            state_pred = odeint_rk4(motion_func, s0_opt.unsqueeze(0), t_span).squeeze(1)
            frames_pred = render_from_positions(canonical, state_pred[:, :2])
            y_est = forward_measurements(masks, frames_pred, mask_norm)

            loss_meas = F.mse_loss(y_est, y_measured)
            loss_ex = exemplar_softmin_distance(canonical, topk_exemplars, SOFTMIN_TEMP)
            loss_resid = residual.pow(2).mean()
            loss_state = (((s0_opt - s0_mu) ** 2) / s0_var).mean()
            loss_tv = total_variation(canonical)
            loss_sparse = canonical.mean()
            loss_total = (
                loss_meas
                + LAMBDA_EXEMPLAR * loss_ex
                + LAMBDA_RESID * loss_resid
                + LAMBDA_STATE * loss_state
                + LAMBDA_TV * loss_tv
                + LAMBDA_SPARSE * loss_sparse
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([alpha_logits, residual, s0_opt], 1.0)
            optimizer.step()
            scheduler.step()

            if loss_total.item() < best_total:
                best_total = loss_total.item()
                best_image = canonical.detach().clone()
                best_frames = frames_pred.detach().clone()
                best_init = base_image.detach().clone()

            if (step + 1) % 100 == 0:
                print(
                    f"   Recon {step + 1:3d}/{RECON_STEPS} | Total={loss_total.item():.6f} "
                    f"Meas={loss_meas.item():.6f} Ex={loss_ex.item():.6f} State={loss_state.item():.6f}"
                )

        meas_mse = F.mse_loss(forward_measurements(masks, best_frames, mask_norm), y_measured).item()
        frame_mse = F.mse_loss(best_frames, gt_frames).item()
        canon_mse = F.mse_loss(best_image, gt_image).item()
        init_canon_mse = F.mse_loss(best_init, gt_image).item()
        results[digit] = {
            "canonical_gt": gt_image.detach().cpu().numpy(),
            "canonical_init": best_init.detach().cpu().numpy(),
            "canonical_recon": best_image.detach().cpu().numpy(),
            "gt": gt_frames.detach().cpu().numpy(),
            "recon": best_frames.detach().cpu().numpy(),
            "meas_mse": meas_mse,
            "frame_mse": frame_mse,
            "canon_mse": canon_mse,
            "init_canon_mse": init_canon_mse,
            "init_meas_loss": float(init_losses[0].item()),
        }
        print(
            f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | "
            f"Frame MSE = {frame_mse:.6f} | Canonical MSE = {canon_mse:.6f}"
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
            f"meas={results[digit]['meas_mse']:.4g}\nframe={results[digit]['frame_mse']:.4g}\ncanon={results[digit]['canon_mse']:.4g}",
            fontsize=9,
            va="center",
        )

    fig.suptitle("Hybrid Exemplar Appearance + Factorized CT-NODE Motion", fontsize=16)
    plt.tight_layout()
    fig.savefig("reconstruction_factorized_ct_node_exemplar_t31.png", dpi=150)
    plt.close(fig)
    print("   Saved 'reconstruction_factorized_ct_node_exemplar_t31.png'.")

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
        fig_anim.suptitle(f"Hybrid CT-NODE Timelapse - Frame {frame_idx + 1}/{T}", fontsize=16)
        return artists

    ani = animation.FuncAnimation(fig_anim, update, frames=T, interval=250, blit=False)
    ani.save("reconstruction_factorized_ct_node_exemplar_t31.gif", writer="pillow")
    plt.close(fig_anim)
    print("   Saved 'reconstruction_factorized_ct_node_exemplar_t31.gif'.")


if __name__ == "__main__":
    main()

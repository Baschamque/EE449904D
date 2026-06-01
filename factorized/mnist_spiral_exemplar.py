"""
This is a version of the factorized method. We use a much stronger appearance prior by reconstructing the canonical digit image directly.
For this we assumed/constrained two things:
1. That the appearance is static (therefore we can get away with reconstructing just one canonical appearance iamge)
2. That appearance should stay close to real digits (rather than generating from weak latent decoder, we regularize it directely towards the real MNIST exemplars)
Appearance stayed much sharper and much closer to the real digit (some minor changes are expected due to forward process).
"""


import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST


H, W = 64, 64
T = 31
SPF = 0.25
M = max(1, int(H * W * SPF))
NOISE_LEVEL = 0.02
TRAIN_PER_DIGIT = 256
TOPK_EXEMPLARS = 16
RECON_STEPS = 400
DIGITS = list(range(10))

OMEGA = 2 * math.pi * 1.5
GAMMA = 1.5
DEFAULT_R = 20.0
DEFAULT_PHASE = 0.0

LAMBDA_EXEMPLAR = 2.0
LAMBDA_TV = 5e-4
LAMBDA_SPARSE = 5e-4
LAMBDA_MOTION = 1e-4
SOFTMIN_TEMP = 0.002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_span = torch.linspace(0.0, 1.0, T, device=device)


def build_mask_normalizer(masks: torch.Tensor) -> torch.Tensor:
    return masks.sum(dim=(-1, -2)).clamp_min(1.0)


def forward_measurements(masks: torch.Tensor, frames: torch.Tensor, normalizer: torch.Tensor) -> torch.Tensor:
    if frames.dim() == 3:
        return torch.einsum("tmhw,thw->tm", masks, frames) / normalizer
    if frames.dim() == 4:
        return torch.einsum("tmhw,bthw->btm", masks, frames) / normalizer.unsqueeze(0)
    raise ValueError(f"Expected frames with 3 or 4 dims, got {frames.dim()}")


def pad_digit(img: torch.Tensor) -> torch.Tensor:
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


def inverse_sigmoid(image: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    image = image.clamp(eps, 1.0 - eps)
    return torch.log(image / (1.0 - image))


def inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    return (image[:, 1:] - image[:, :-1]).abs().mean() + (image[1:, :] - image[:-1, :]).abs().mean()


def exemplar_softmin_distance(image: torch.Tensor, exemplars: torch.Tensor, temperature: float) -> torch.Tensor:
    distances = ((exemplars - image.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    return -temperature * torch.logsumexp(-distances / temperature, dim=0)


def load_centered_mnist_split(train: bool, per_digit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())
    targets = dataset.targets
    images = []
    labels = []
    for digit in DIGITS:
        digit_indices = torch.where(targets == digit)[0]
        take = len(digit_indices) if per_digit is None else min(per_digit, len(digit_indices))
        print(f"   Found {len(digit_indices)} {'train' if train else 'test'} samples of digit '{digit}', taking {take}")
        chosen = digit_indices[:take]
        for idx in chosen:
            img, _ = dataset[idx.item()]
            images.append(pad_digit(img).squeeze(0))
            labels.append(digit)
    return torch.stack(images, dim=0).to(device), torch.tensor(labels, device=device)


def get_first_test_per_digit(test_images: torch.Tensor, test_labels: torch.Tensor) -> dict[int, torch.Tensor]:
    samples = {}
    for digit in DIGITS:
        first_idx = torch.where(test_labels == digit)[0][0]
        samples[digit] = test_images[first_idx]
    return samples


def select_topk_exemplars(
    exemplars: torch.Tensor,
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
            frames.append(
                render_spiral_from_canonical(
                    batch[j],
                    torch.tensor(DEFAULT_PHASE, device=device),
                    torch.tensor(DEFAULT_R, device=device),
                )
            )
        frame_batch = torch.stack(frames, dim=0)
        y_est = forward_measurements(masks, frame_batch, mask_norm)
        losses.append(((y_est - y_measured.unsqueeze(0)) ** 2).mean(dim=(1, 2)))
    losses = torch.cat(losses, dim=0)
    values, indices = torch.topk(losses, k=min(topk, losses.numel()), largest=False)
    return exemplars[indices], values


def reconstruct_digit_sequence(
    exemplars: torch.Tensor,
    masks: torch.Tensor,
    mask_norm: torch.Tensor,
    y_measured: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
    topk_exemplars, exemplar_losses = select_topk_exemplars(exemplars, masks, mask_norm, y_measured, TOPK_EXEMPLARS)
    init_image = topk_exemplars[0]

    canon_logits = torch.nn.Parameter(inverse_sigmoid(init_image))
    phase = torch.nn.Parameter(torch.tensor(DEFAULT_PHASE, device=device))
    raw_r = torch.nn.Parameter(torch.tensor(inverse_softplus(DEFAULT_R), device=device))
    optimizer = optim.Adam([canon_logits, phase, raw_r], lr=8e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECON_STEPS, eta_min=5e-4)

    best_total = float("inf")
    best_image = None
    best_frames = None
    best_phase = DEFAULT_PHASE
    best_r = DEFAULT_R

    for step in range(RECON_STEPS):
        optimizer.zero_grad()
        canonical = torch.sigmoid(canon_logits)
        r_init = F.softplus(raw_r)
        frames = render_spiral_from_canonical(canonical, phase, r_init)
        y_est = forward_measurements(masks, frames, mask_norm)

        loss_meas = F.mse_loss(y_est, y_measured)
        loss_exemplar = exemplar_softmin_distance(canonical, topk_exemplars, SOFTMIN_TEMP)
        loss_tv = total_variation(canonical)
        loss_sparse = canonical.mean()
        loss_motion = (phase - DEFAULT_PHASE) ** 2 + (r_init - DEFAULT_R) ** 2

        loss_total = (
            loss_meas
            + LAMBDA_EXEMPLAR * loss_exemplar
            + LAMBDA_TV * loss_tv
            + LAMBDA_SPARSE * loss_sparse
            + LAMBDA_MOTION * loss_motion
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([canon_logits, phase, raw_r], 1.0)
        optimizer.step()
        scheduler.step()

        if loss_total.item() < best_total:
            best_total = loss_total.item()
            best_image = canonical.detach().clone()
            best_frames = frames.detach().clone()
            best_phase = phase.detach().item()
            best_r = r_init.detach().item()

        if (step + 1) % 100 == 0:
            print(
                f"   Recon {step + 1:3d}/{RECON_STEPS} | Total={loss_total.item():.6f} "
                f"Meas={loss_meas.item():.6f} Ex={loss_exemplar.item():.6f} "
                f"phase={phase.item():.3f} r={r_init.item():.3f}"
            )

    return best_image, best_frames, topk_exemplars[0], best_phase, best_r, exemplar_losses[0].item()


def main():
    print("1. Loading Centered MNIST Exemplar Bank...")
    train_images, train_labels = load_centered_mnist_split(train=True, per_digit=TRAIN_PER_DIGIT)
    test_images, test_labels = load_centered_mnist_split(train=False, per_digit=1)
    test_by_digit = get_first_test_per_digit(test_images, test_labels)
    print(f"   Train bank: {train_images.shape}, Test digits: {len(test_by_digit)}")

    print("\n2. Building Shared Measurement Operator...")
    masks = torch.randint(0, 2, (T, M, H, W), device=device).float()
    mask_norm = build_mask_normalizer(masks)

    print("\n3. Reconstructing All Digits with Direct Image + Exemplar Prior...")
    results = {}

    for digit in DIGITS:
        print(f"\n   Reconstructing digit {digit}...")
        canonical_gt = test_by_digit[digit]
        frames_gt = render_spiral_from_canonical(
            canonical_gt,
            torch.tensor(DEFAULT_PHASE, device=device),
            torch.tensor(DEFAULT_R, device=device),
        )
        y_clean = forward_measurements(masks, frames_gt, mask_norm)
        noise_std = NOISE_LEVEL * y_clean.std()
        y_measured = y_clean + torch.randn_like(y_clean) * noise_std

        exemplars = train_images[train_labels == digit]
        canonical_recon, frames_recon, exemplar_init, phase_hat, r_hat, init_meas_loss = reconstruct_digit_sequence(
            exemplars,
            masks,
            mask_norm,
            y_measured,
        )

        meas_mse = F.mse_loss(forward_measurements(masks, frames_recon, mask_norm), y_measured).item()
        frame_mse = F.mse_loss(frames_recon, frames_gt).item()
        canon_mse = F.mse_loss(canonical_recon, canonical_gt).item()
        exemplar_mse = F.mse_loss(exemplar_init, canonical_gt).item()

        results[digit] = {
            "canonical_gt": canonical_gt.detach().cpu().numpy(),
            "canonical_recon": canonical_recon.detach().cpu().numpy(),
            "canonical_init": exemplar_init.detach().cpu().numpy(),
            "gt": frames_gt.detach().cpu().numpy(),
            "recon": frames_recon.detach().cpu().numpy(),
            "meas_mse": meas_mse,
            "frame_mse": frame_mse,
            "canon_mse": canon_mse,
            "exemplar_mse": exemplar_mse,
            "phase": phase_hat,
            "r_init": r_hat,
            "init_meas_loss": init_meas_loss,
        }
        print(
            f"   Digit {digit} | Measurement MSE = {meas_mse:.6f} | "
            f"Frame MSE = {frame_mse:.6f} | Canonical MSE = {canon_mse:.6f} | "
            f"Exemplar MSE = {exemplar_mse:.6f} | phase={phase_hat:.3f} r={r_hat:.3f}"
        )

    print("\n4. Saving Comparison Figure...")
    frames_to_show = [0, T // 2, T - 1]
    fig, axes = plt.subplots(len(DIGITS) * 2, len(frames_to_show) + 3, figsize=(14, 2.4 * len(DIGITS)))

    for row, digit in enumerate(DIGITS):
        axes[2 * row, 0].imshow(results[digit]["canonical_gt"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row, 0].set_title(f"{digit} Canon GT")
        axes[2 * row, 0].axis("off")

        axes[2 * row + 1, 0].imshow(results[digit]["canonical_init"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row + 1, 0].set_title(f"{digit} Init Ex")
        axes[2 * row + 1, 0].axis("off")

        axes[2 * row, 1].imshow(results[digit]["canonical_gt"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row, 1].set_title(f"{digit} Canon GT")
        axes[2 * row, 1].axis("off")

        axes[2 * row + 1, 1].imshow(results[digit]["canonical_recon"], cmap="gray", vmin=0, vmax=1)
        axes[2 * row + 1, 1].set_title(f"{digit} Canon Rec")
        axes[2 * row + 1, 1].axis("off")

        for col, frame_idx in enumerate(frames_to_show, start=2):
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

    fig.suptitle("Direct Canonical Image Reconstruction with Exemplar Prior", fontsize=16)
    plt.tight_layout()
    fig.savefig("reconstruction_exemplar_all_digits_t31.png", dpi=150)
    plt.close(fig)
    print("   Saved 'reconstruction_exemplar_all_digits_t31.png'.")

    print("\n5. Saving Timelapse GIF...")
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
        fig_anim.suptitle(f"Exemplar-Prior SPI Timelapse - Frame {frame_idx + 1}/{T}", fontsize=16)
        return artists

    ani = animation.FuncAnimation(fig_anim, update, frames=T, interval=250, blit=False)
    ani.save("reconstruction_exemplar_timelapse_t31.gif", writer="pillow")
    plt.close(fig_anim)
    print("   Saved 'reconstruction_exemplar_timelapse_t31.gif'.")


if __name__ == "__main__":
    main()

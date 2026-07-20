import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


dataset, info = tfds.load('oxford_iiit_pet:4.0.0', with_info=True)
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from skimage import measure
from scipy.spatial.distance import directed_hausdorff


for sample in dataset['train'].take(1):
    img_tf  = sample['image']
    mask_tf = sample['segmentation_mask']

img_np  = img_tf.numpy().astype(np.float32) / 255.0
mask_np = mask_tf.numpy().squeeze()


mask_gt =np.where((mask_np == 1) | (mask_np == 3), 1, 0)


img_gray   = rgb2gray(img_np)


rows, cols = np.where(mask_gt)
r_c    = (rows.min() + rows.max()) / 2
c_c    = (cols.min() + cols.max()) / 2
radius = max(rows.max() - rows.min(),
             cols.max() - cols.min()) / 2 * 0.75

s    = np.linspace(0, 2 * np.pi, 400)
init = np.array([r_c + radius * np.sin(s),
                 c_c + radius * np.cos(s)]).T
init[:, 0] = np.clip(init[:, 0], 1, img_gray.shape[0] - 2)
init[:, 1] = np.clip(init[:, 1], 1, img_gray.shape[1] - 2)


img_smooth = gaussian(img_gray, sigma=4, preserve_range=False)

iter_steps = [1, 3, 6, 10, 15, 25, 40, 60, 90, 130, 200, 350, 600, 1000, 2500]
print("Calcul des snapshots...")
snapshots = [
    active_contour(
        img_smooth,
        init.copy(),
        alpha=0.01,
        beta=0.1,
        gamma=0.001,
        w_line=0,
        w_edge=5,
        max_num_iter=n
    )
    for n in iter_steps
]
print("Fait.")


def contour_to_mask(contour, shape):
    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(contour[:, 0], contour[:, 1], shape)
    mask[rr, cc] = True
    return mask

def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-8)

def hausdorff_dist(c_pred, c_gt):
    return max(directed_hausdorff(c_pred, c_gt)[0],
               directed_hausdorff(c_gt, c_pred)[0])


contours_gt = measure.find_contours(mask_gt.astype(float), 0.5)
gt_contour  = max(contours_gt, key=len)


dice_vals, iou_vals, haus_vals = [], [], []
for snap in snapshots:
    m = contour_to_mask(snap, img_gray.shape)
    dice_vals.append(dice_score(m, mask_gt))
    iou_vals.append(iou_score(m, mask_gt))
    haus_vals.append(hausdorff_dist(snap, gt_contour))

final      = snapshots[-1]
mask_final = contour_to_mask(final, img_gray.shape)
print("=" * 45)
print(f"  Dice           : {dice_score(mask_final, mask_gt):.4f}")
print(f"  IoU            : {iou_score(mask_final, mask_gt):.4f}")
print(f"  Hausdorff (px) : {hausdorff_dist(final, gt_contour):.2f}")
print("=" * 45)


fig_final, ax_final = plt.subplots(figsize=(7, 7))
ax_final.imshow(img_gray, cmap='gray')
ax_final.plot(init[:, 1], init[:, 0], '--r', lw=1.5, label='Initial')
ax_final.plot(gt_contour[:, 1], gt_contour[:, 0], '-g', lw=1.5, label='GT')
ax_final.plot(snapshots[-1][:, 1], snapshots[-1][:, 0], '-b', lw=2.5, label='Snake Final')
ax_final.set_title(f"Segmentation Finale (Dice: {dice_score(mask_final, mask_gt):.4f})")
ax_final.legend(loc='lower right')
ax_final.axis('off')
plt.show()

fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle("Évolution des métriques — Oxford IIIT Pet", fontsize=13)

for ax_, vals, label, color in zip(
        axes,
        [dice_vals, iou_vals, haus_vals],
        ["Dice", "IoU", "Hausdorff (px)"],
        ['b', 'g', 'r']):
    ax_.plot(iter_steps, vals, f'o-{color}', lw=2)
    ax_.set_title(label)
    ax_.set_xlabel("Itérations (log)")
    ax_.set_xscale('log')
    ax_.grid(alpha=0.4)
    if label != "Hausdorff (px)":
        ax_.set_ylim(0, 1.05)

plt.tight_layout()
plt.show(block=True)

IMG_SIZE = 128
train_data = dataset['train']

def preprocess_for_eval(sample):
    # Redimensionnement identique à Chan-Vese
    image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(
        sample["segmentation_mask"],
        (IMG_SIZE, IMG_SIZE),
        method='nearest'
    )

    image = tf.cast(image, tf.float32) / 255.0

    # Classe 1 = animal, classe 3 = bordure 
    mask_gt = tf.cast((mask == 1) | (mask == 3), tf.float32).numpy().squeeze()

    return image.numpy(), mask_gt


def run_snake_fast(img_np):
    img_gray = rgb2gray(img_np)

    # Prétraitement
    img_smooth = gaussian(img_gray, sigma=3, preserve_range=False)

    # Initialisation cercle centré
    r_c, c_c = IMG_SIZE / 2, IMG_SIZE / 2
    radius = (IMG_SIZE / 2) * 0.7

    s = np.linspace(0, 2 * np.pi, 400)

    init = np.array([
        r_c + radius * np.sin(s),
        c_c + radius * np.cos(s)
    ]).T

    init[:, 0] = np.clip(init[:, 0], 1, IMG_SIZE - 2)
    init[:, 1] = np.clip(init[:, 1], 1, IMG_SIZE - 2)

    # Snake
    snap = active_contour(
        img_smooth,
        init,
        alpha=0.01,
        beta=0.1,
        gamma=0.001,
        w_line=0,
        w_edge=5,
        max_num_iter=400
    )

    return contour_to_mask(snap, img_gray.shape)


print("\n" + "="*50)
print("DÉBUT DE L'ÉVALUATION SUR 50 IMAGES (Format 128x128)")
print("="*50)

results_eval = []

print("Calcul en cours sur 50 images...")

for i, sample in enumerate(train_data.take(50)):

    img, gt = preprocess_for_eval(sample)

    pred_mask = run_snake_fast(img)

    d = dice_score(pred_mask, gt)
    j = iou_score(pred_mask, gt)

    contours_gt = measure.find_contours(gt.astype(float), 0.5)
    gt_contour = max(contours_gt, key=len)

    contours_pred = measure.find_contours(pred_mask.astype(float), 0.5)

    if len(contours_pred) > 0:
        pred_contour = max(contours_pred, key=len)

        haus = hausdorff_dist(pred_contour, gt_contour)

        diag = np.sqrt(
            pred_mask.shape[0]**2 +
            pred_mask.shape[1]**2
        )

        haus_norm = haus / diag

    else:
        haus_norm = 1.0

    # Sauvegarde
    results_eval.append((d, j, haus_norm))

    if (i + 1) % 10 == 0:
        print(f" -> Images traitées : {i+1}/50")


dice_mean = np.mean([r[0] for r in results_eval])
iou_mean = np.mean([r[1] for r in results_eval])
haus_mean = np.mean([r[2] for r in results_eval])


print("\n" + "-" * 50)
print("RÉSULTATS FINAUX SUR 50 IMAGES (SNAKE)")
print(f"Dice moyen                : {dice_mean:.4f}")
print(f"IoU moyen                 : {iou_mean:.4f}")
print(f"Hausdorff normalisé moyen : {haus_mean:.4f}")
print("-" * 50)


sample = next(iter(train_data.take(1)))

# Prétraitement
img, gt = preprocess_for_eval(sample)

# Prédiction Snake
pred_mask = run_snake_fast(img)


dice = dice_score(pred_mask, gt)
iou  = iou_score(pred_mask, gt)


img_gray = rgb2gray(img)

# Contour Ground Truth
contours_gt = measure.find_contours(gt.astype(float), 0.5)
gt_contour = max(contours_gt, key=len)

# Contour prédit
contours_pred = measure.find_contours(pred_mask.astype(float), 0.5)
pred_contour = max(contours_pred, key=len)

# Hausdorff brut
haus = hausdorff_dist(pred_contour, gt_contour)

# Normalisation par la diagonale
diag = np.sqrt(img_gray.shape[0]**2 + img_gray.shape[1]**2)
haus_norm = haus / diag


print("=" * 50)
print("Résultats sur la 1ère image")
print(f"Dice                 : {dice:.4f}")
print(f"IoU                  : {iou:.4f}")
print(f"Hausdorff normalisé  : {haus_norm:.4f}")
print("=" * 50)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Image originale
axes[0].imshow(img)
axes[0].set_title("Image")
axes[0].axis("off")

# Ground Truth
axes[1].imshow(gt, cmap='gray')
axes[1].set_title("Ground Truth")
axes[1].axis("off")

# Prédiction Snake
axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title(
    f"Snake\nDice={dice:.3f} | IoU={iou:.3f} | H={haus_norm:.3f}"
)
axes[2].axis("off")

plt.show()

# --- Bloc supplémentaire : test isolé avec un cercle fixe (rayon/centre en dur) ---
for sample in train_data.take(1):
    img_tf = sample['image']

img = img_tf.numpy().astype(np.float32) / 255.0
img_gray = rgb2gray(img)

s = np.linspace(0, 2 * np.pi, 400)
r = 220 + 210 * np.sin(s)
c = 290 + 210 * np.cos(s)
init = np.array([r, c]).T

snake = active_contour(
    gaussian(img_gray, sigma=3, preserve_range=False),
    init,
    alpha=0.01,
    beta=0.01,
    gamma=0.001,
)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_gray, cmap='gray')
ax.plot(init[:, 1], init[:, 0], '--r', lw=2, label='Initial')
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=2, label='Snake')
ax.set_xticks([])
ax.set_yticks([])
ax.legend()
plt.show()

for sample in train_data.take(1):
    mask_tf = sample['segmentation_mask']

mask = mask_tf.numpy().squeeze()
# Classe 1 = animal, classe 3 = bordure 
gt = (mask == 1) | (mask == 3)

pred_mask = np.zeros(img_gray.shape, dtype=bool)
rr, cc = polygon(snake[:, 0], snake[:, 1], img_gray.shape)
pred_mask[rr, cc] = True

dice = dice_score(pred_mask, gt)
iou  = iou_score(pred_mask, gt)

print("=" * 40)
print(f"Dice : {dice:.4f}")
print(f"IoU  : {iou:.4f}")
print("=" * 40)

plt.figure(figsize=(6, 6))
plt.imshow(pred_mask, cmap='gray')
plt.title(f"Prediction\nDice={dice:.3f} | IoU={iou:.3f}")
plt.axis('off')
plt.show()

haus = hausdorff_dist(snake, gt_contour)
diag = np.sqrt(img_gray.shape[0]**2 + img_gray.shape[1]**2)
haus_norm = haus / diag

print(f"Hausdorff : {haus:.2f} px")
print(f"Hausdorff normalisé : {haus_norm:.4f}")

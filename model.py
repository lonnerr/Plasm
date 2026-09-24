
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATASET_PATH = "lj_dataset/dataset.npz"

SEED = 42

BATCH_SIZE = 512
EPOCHS = 200

LEARNING_RATE = 1e-3

# Размеры сети
HIDDEN_SIZE = 128

# Доля данных
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ============================================================
# SEED
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используем GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Используем CPU")


# ============================================================
# ЗАГРУЗКА DATASET
# ============================================================

print()
print("=" * 60)
print("ЗАГРУЗКА DATASET")
print("=" * 60)

data = np.load(DATASET_PATH)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# SHUFFLE
# ============================================================

N = len(X)

indices = np.random.permutation(N)

X = X[indices]
Y = Y[indices]


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

train_end = int(N * TRAIN_RATIO)
val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]


print()
print("Размеры:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# НОРМАЛИЗАЦИЯ
# ============================================================

# Очень важно:
#
# статистики считаем ТОЛЬКО по train.
#
# Нельзя использовать test/validation при вычислении
# параметров нормализации.

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

Y_mean = Y_train.mean(axis=0)
Y_std = Y_train.std(axis=0)


# Защита от деления на 0
X_std[X_std < 1e-8] = 1.0
Y_std[Y_std < 1e-8] = 1.0


def normalize_X(x):
    return (x - X_mean) / X_std


def normalize_Y(y):
    return (y - Y_mean) / Y_std


def denormalize_Y(y):
    return y * Y_std + Y_mean


X_train_norm = normalize_X(X_train)
X_val_norm = normalize_X(X_val)
X_test_norm = normalize_X(X_test)

Y_train_norm = normalize_Y(Y_train)
Y_val_norm = normalize_Y(Y_val)
Y_test_norm = normalize_Y(Y_test)


# ============================================================
# TORCH DATASETS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_norm,
    dtype=torch.float32,
)

Y_train_tensor = torch.tensor(
    Y_train_norm,
    dtype=torch.float32,
)

X_val_tensor = torch.tensor(
    X_val_norm,
    dtype=torch.float32,
)

Y_val_tensor = torch.tensor(
    Y_val_norm,
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_norm,
    dtype=torch.float32,
)

Y_test_tensor = torch.tensor(
    Y_test_norm,
    dtype=torch.float32,
)


train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor,
)

val_dataset = TensorDataset(
    X_val_tensor,
    Y_val_tensor,
)

test_dataset = TensorDataset(
    X_test_tensor,
    Y_test_tensor,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

class ForceModel(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        output_size=9,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                output_size,
            ),
        )

    def forward(self, x):
        return self.network(x)


model = ForceModel(
    input_size=6,
    hidden_size=HIDDEN_SIZE,
    output_size=9,
)

model = model.to(device)


print()
print("=" * 60)
print("MODEL")
print("=" * 60)

print(model)


# ============================================================
# LOSS
# ============================================================

# Huber loss менее чувствителен к редким очень большим силам,
# чем MSE.

criterion = nn.HuberLoss(
    delta=1.0
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING
# ============================================================

train_losses = []
val_losses = []


print()
print("=" * 60)
print("TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_samples = 0

    for batch_X, batch_Y in train_loader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = criterion(
            prediction,
            batch_Y,
        )

        loss.backward()

        optimizer.step()

        batch_size = batch_X.shape[0]

        train_loss_sum += (
            loss.item()
            * batch_size
        )

        train_samples += batch_size


    train_loss = (
        train_loss_sum
        / train_samples
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_samples = 0

    with torch.no_grad():

        for batch_X, batch_Y in val_loader:

            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            prediction = model(batch_X)

            loss = criterion(
                prediction,
                batch_Y,
            )

            batch_size = batch_X.shape[0]

            val_loss_sum += (
                loss.item()
                * batch_size
            )

            val_samples += batch_size


    val_loss = (
        val_loss_sum
        / val_samples
    )


    train_losses.append(train_loss)
    val_losses.append(val_loss)


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:3d}/{EPOCHS} | "
            f"train = {train_loss:.6f} | "
            f"val = {val_loss:.6f}"
        )


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("TEST")
print("=" * 60)

model.eval()

predictions_norm = []

with torch.no_grad():

    for batch_X, _ in test_loader:

        batch_X = batch_X.to(device)

        prediction = model(batch_X)

        predictions_norm.append(
            prediction.cpu().numpy()
        )


predictions_norm = np.concatenate(
    predictions_norm,
    axis=0,
)


# Возвращаем физические единицы
predictions = denormalize_Y(
    predictions_norm
)


# ============================================================
# METRICS
# ============================================================

error = predictions - Y_test

absolute_error = np.abs(error)

squared_error = error**2


mae = absolute_error.mean()

rmse = np.sqrt(
    squared_error.mean()
)


print(f"MAE  = {mae:.6e}")
print(f"RMSE = {rmse:.6e}")


# ============================================================
# MAE ДЛЯ КАЖДОЙ КОМПОНЕНТЫ
# ============================================================

component_mae = np.mean(
    np.abs(
        predictions - Y_test
    ),
    axis=0,
)


print()
print("MAE по компонентам:")

names = [
    "Fx1", "Fy1", "Fz1",
    "Fx2", "Fy2", "Fz2",
    "Fx3", "Fy3", "Fz3",
]

for name, value in zip(
    names,
    component_mae,
):

    print(
        f"{name:>3s}: "
        f"{value:.6e}"
    )


# ============================================================
# ПРОВЕРКА F1 + F2 + F3 = 0
# ============================================================

predictions_3d = predictions.reshape(
    -1,
    3,
    3,
)

true_forces_3d = Y_test.reshape(
    -1,
    3,
    3,
)


predicted_total_force = (
    predictions_3d.sum(axis=1)
)

true_total_force = (
    true_forces_3d.sum(axis=1)
)


predicted_total_error = np.linalg.norm(
    predicted_total_force,
    axis=1,
)

true_total_error = np.linalg.norm(
    true_total_force,
    axis=1,
)


print()
print("=" * 60)
print("ФИЗИЧЕСКАЯ ПРОВЕРКА")
print("=" * 60)

print(
    "Истинные силы:"
)

print(
    f"mean |F1+F2+F3| = "
    f"{true_total_error.mean():.6e}"
)

print(
    f"max  |F1+F2+F3| = "
    f"{true_total_error.max():.6e}"
)


print()
print(
    "Предсказанные силы:"
)

print(
    f"mean |F1+F2+F3| = "
    f"{predicted_total_error.mean():.6e}"
)

print(
    f"max  |F1+F2+F3| = "
    f"{predicted_total_error.max():.6e}"
)


# ============================================================
# СОХРАНЕНИЕ МОДЕЛИ
# ============================================================

model_path = "lj_force_model.pt"

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "X_mean":
            X_mean,

        "X_std":
            X_std,

        "Y_mean":
            Y_mean,

        "Y_std":
            Y_std,

        "input_size":
            6,

        "output_size":
            9,

        "hidden_size":
            HIDDEN_SIZE,
    },
    model_path,
)


print()
print(f"Модель сохранена: {model_path}")


# ============================================================
# ГРАФИК LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses,
    label="Train",
)

plt.plot(
    val_losses,
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Huber loss")
plt.title("Training history")
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# TRUE VS PREDICTED
# ============================================================

# Для первого анализа возьмём Fx1.

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    Y_test[:, 0],
    predictions[:, 0],
    s=4,
    alpha=0.3,
)

min_value = min(
    Y_test[:, 0].min(),
    predictions[:, 0].min(),
)

max_value = max(
    Y_test[:, 0].max(),
    predictions[:, 0].max(),
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
)

plt.xlabel("True Fx1")
plt.ylabel("Predicted Fx1")
plt.title("True vs predicted Fx1")

plt.tight_layout()

plt.show()


# ============================================================
# НЕСКОЛЬКО РЕАЛЬНЫХ ПРИМЕРОВ
# ============================================================

print()
print("=" * 60)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("=" * 60)


for i in range(5):

    print()
    print(f"Example {i + 1}")

    print("Input:")
    print(X_test[i])

    print()
    print("True forces:")

    print(
        Y_test[i].reshape(
            3,
            3,
        )
    )

    print()
    print("Predicted forces:")

    print(
        predictions[i].reshape(
            3,
            3,
        )
    )

    print()
    print(
        "Absolute error:"
    )

    print(
        np.abs(
            predictions[i]
            - Y_test[i]
        ).reshape(
            3,
            3,
        )
    )

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

N_SAMPLES = 100_000

# Параметры потенциала Леннарда–Джонса
SIGMA = 1.0
EPSILON = 1.0

# Диапазон всех попарных расстояний
R_MIN = 0.9 * SIGMA
R_MAX = 2.5 * SIGMA

# Seed для воспроизводимости
SEED = 42

# Папка для результатов
OUTPUT_DIR = Path("lj_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# СЛУЧАЙНЫЙ ЕДИНИЧНЫЙ ВЕКТОР
# ============================================================

def random_unit_vector(rng):
    """
    Возвращает случайный единичный вектор в 3D.

    Направление равномерно распределено по поверхности сферы.
    """
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


# ============================================================
# СИЛА ЛЕННАРДА–ДЖОНСА МЕЖДУ ДВУМЯ ЧАСТИЦАМИ
# ============================================================

def lj_pair_force(p_i, p_j, epsilon=1.0, sigma=1.0):
    """
    Возвращает силу, действующую на частицу i
    со стороны частицы j.

    p_i, p_j:
        координаты частиц, shape = (3,)

    Возвращает:
        force, shape = (3,)
    """

    # Вектор от j к i
    r_vector = p_i - p_j

    # Расстояние между частицами
    r = np.linalg.norm(r_vector)

    sr6 = (sigma / r) ** 6

    force = (
        24.0
        * epsilon
        / r**2
        * (2.0 * sr6**2 - sr6)
        * r_vector
    )

    return force


# ============================================================
# СИЛЫ ДЛЯ ТРЁХ ЧАСТИЦ
# ============================================================

def calculate_forces(positions, epsilon=1.0, sigma=1.0):
    """
    positions:

        [
            [x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3]
        ]

    shape = (3, 3)

    Возвращает:

        [
            [Fx1, Fy1, Fz1],
            [Fx2, Fy2, Fz2],
            [Fx3, Fy3, Fz3]
        ]

    shape = (3, 3)
    """

    forces = np.zeros((3, 3), dtype=np.float64)

    # Пара 1-2
    f12 = lj_pair_force(
        positions[0],
        positions[1],
        epsilon,
        sigma,
    )

    forces[0] += f12
    forces[1] -= f12

    # Пара 1-3
    f13 = lj_pair_force(
        positions[0],
        positions[2],
        epsilon,
        sigma,
    )

    forces[0] += f13
    forces[2] -= f13

    # Пара 2-3
    f23 = lj_pair_force(
        positions[1],
        positions[2],
        epsilon,
        sigma,
    )

    forces[1] += f23
    forces[2] -= f23

    return forces


# ============================================================
# ГЕНЕРАЦИЯ КОНФИГУРАЦИИ
# ============================================================

def generate_configuration(
    rng,
    r_min=R_MIN,
    r_max=R_MAX,
):
    """
    Генерирует конфигурацию из трёх частиц.

    Частица 1 фиксирована:

        p1 = (0, 0, 0)

    r12 и r13 выбираются равномерно из [r_min, r_max].

    Затем выбирается r23 так, чтобы три расстояния
    могли образовать треугольник.

    После этого весь треугольник случайно ориентируется
    в 3D.

    Возвращает:

        positions.shape == (3, 3)
    """

    # --------------------------------------------------------
    # Выбираем расстояния r12 и r13
    # --------------------------------------------------------

    while True:

        r12 = rng.uniform(
            r_min,
            r_max,
        )

        r13 = rng.uniform(
            r_min,
            r_max,
        )

        # Для существования треугольника:

        # |r12 - r13| <= r23 <= r12 + r13

        lower = max(
            r_min,
            abs(r12 - r13),
        )

        upper = min(
            r_max,
            r12 + r13,
        )

        if lower < upper:

            r23 = rng.uniform(
                lower,
                upper,
            )

            break

    # --------------------------------------------------------
    # Случайная ориентация в 3D
    # --------------------------------------------------------

    # Направление от частицы 1 к частице 2
    e1 = random_unit_vector(rng)

    # Получаем случайный вектор
    temp = random_unit_vector(rng)

    # Делаем его перпендикулярным e1
    e2 = (
        temp
        - np.dot(temp, e1) * e1
    )

    e2 /= np.linalg.norm(e2)

    # --------------------------------------------------------
    # Угол между r12 и r13
    #
    # Закон косинусов:
    #
    # r23^2 = r12^2 + r13^2
    #         - 2*r12*r13*cos(theta)
    # --------------------------------------------------------

    cos_theta = (
        r12**2
        + r13**2
        - r23**2
    ) / (
        2.0 * r12 * r13
    )

    # Защита от ошибок округления
    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    sin_theta = np.sqrt(
        1.0 - cos_theta**2
    )

    # Случайный азимут
    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    # Третье направление
    e3 = (
        cos_theta * e1
        +
        sin_theta
        * (
            np.cos(phi) * e2
            +
            np.sin(phi)
            * np.cross(e1, e2)
        )
    )

    # --------------------------------------------------------
    # Координаты частиц
    # --------------------------------------------------------

    # Первая частица всегда в начале координат
    p1 = np.zeros(3)

    # Вторая частица
    p2 = r12 * e1

    # Третья частица
    p3 = r13 * e3

    positions = np.array(
        [
            p1,
            p2,
            p3,
        ],
        dtype=np.float64,
    )

    return positions


# ============================================================
# ГЕНЕРАЦИЯ DATASET
# ============================================================

print("=" * 60)
print("ГЕНЕРАЦИЯ DATASET")
print("=" * 60)

print(f"Количество конфигураций: {N_SAMPLES:,}")
print(f"sigma: {SIGMA}")
print(f"epsilon: {EPSILON}")
print(f"Диапазон расстояний: [{R_MIN}, {R_MAX}]")
print()


# ------------------------------------------------------------
# X
#
# Только координаты частиц 2 и 3.
#
# particle 1 всегда:
#
# (0, 0, 0)
#
# Поэтому:
#
# X = [x2,y2,z2,x3,y3,z3]
#
# ------------------------------------------------------------

X = np.empty(
    (N_SAMPLES, 6),
    dtype=np.float64,
)


# ------------------------------------------------------------
# Y
#
# Силы всех трёх частиц:
#
# Y =
#
# [Fx1,Fy1,Fz1,
#  Fx2,Fy2,Fz2,
#  Fx3,Fy3,Fz3]
#
# ------------------------------------------------------------

Y = np.empty(
    (N_SAMPLES, 9),
    dtype=np.float64,
)


# Попарные расстояния
#
# distances[:, 0] = r12
# distances[:, 1] = r13
# distances[:, 2] = r23

distances = np.empty(
    (N_SAMPLES, 3),
    dtype=np.float64,
)


# Модули сил
#
# force_norms[:, 0] = |F1|
# force_norms[:, 1] = |F2|
# force_norms[:, 2] = |F3|

force_norms = np.empty(
    (N_SAMPLES, 3),
    dtype=np.float64,
)


# ------------------------------------------------------------
# Основной цикл
# ------------------------------------------------------------

for n in range(N_SAMPLES):

    # Генерируем геометрию
    positions = generate_configuration(rng)

    # Считаем силы
    forces = calculate_forces(
        positions,
        epsilon=EPSILON,
        sigma=SIGMA,
    )

    # --------------------------------------------------------
    # Входные данные
    #
    # Не сохраняем particle 1,
    # потому что она всегда (0,0,0).
    # --------------------------------------------------------

    X[n] = positions[1:].reshape(-1)

    # --------------------------------------------------------
    # Целевые значения
    # --------------------------------------------------------

    Y[n] = forces.reshape(-1)

    # --------------------------------------------------------
    # Расстояния
    # --------------------------------------------------------

    distances[n, 0] = np.linalg.norm(
        positions[0] - positions[1]
    )

    distances[n, 1] = np.linalg.norm(
        positions[0] - positions[2]
    )

    distances[n, 2] = np.linalg.norm(
        positions[1] - positions[2]
    )

    # --------------------------------------------------------
    # Модули сил
    # --------------------------------------------------------

    force_norms[n] = np.linalg.norm(
        forces,
        axis=1,
    )

    # Прогресс
    if (n + 1) % 10_000 == 0:

        print(
            f"Создано: "
            f"{n + 1:,} / {N_SAMPLES:,}"
        )


# ============================================================
# ПРОВЕРКА РАССТОЯНИЙ
# ============================================================

print()
print("=" * 60)
print("ПРОВЕРКА РАССТОЯНИЙ")
print("=" * 60)

print(
    f"Минимальное расстояние: "
    f"{distances.min():.6f}"
)

print(
    f"Максимальное расстояние: "
    f"{distances.max():.6f}"
)

assert distances.min() >= R_MIN - 1e-12
assert distances.max() <= R_MAX + 1e-12


# ============================================================
# ПРОВЕРКА СОХРАНЕНИЯ ИМПУЛЬСА
# ============================================================

print()
print("=" * 60)
print("ПРОВЕРКА СИЛ")
print("=" * 60)

forces_3d = Y.reshape(
    -1,
    3,
    3,
)

# F_total = F1 + F2 + F3

total_force = forces_3d.sum(
    axis=1
)

total_force_error = np.linalg.norm(
    total_force,
    axis=1,
)

print(
    f"max |F1 + F2 + F3|: "
    f"{total_force_error.max():.3e}"
)

print(
    f"mean |F1 + F2 + F3|: "
    f"{total_force_error.mean():.3e}"
)


# ============================================================
# РАЗМЕРНОСТИ
# ============================================================

print()
print("=" * 60)
print("РАЗМЕРЫ DATASET")
print("=" * 60)

print(f"X.shape = {X.shape}")
print(f"Y.shape = {Y.shape}")

print()
print("Одна строка X:")
print(X[0])

print()
print("Одна строка Y:")
print(Y[0])


# ============================================================
# СОХРАНЕНИЕ NPZ
# ============================================================

npz_path = OUTPUT_DIR / "dataset.npz"

np.savez_compressed(
    npz_path,

    # Входы
    X=X.astype(np.float32),

    # Силы
    Y=Y.astype(np.float32),

    # Дополнительная информация
    distances=distances.astype(
        np.float32
    ),

    force_norms=force_norms.astype(
        np.float32
    ),

    sigma=np.array(
        SIGMA,
        dtype=np.float32,
    ),

    epsilon=np.array(
        EPSILON,
        dtype=np.float32,
    ),

    r_min=np.array(
        R_MIN,
        dtype=np.float32,
    ),

    r_max=np.array(
        R_MAX,
        dtype=np.float32,
    ),
)

print()
print(f"NPZ сохранён:")
print(npz_path)


# ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================

columns = [

    # Coordinates
    "x2",
    "y2",
    "z2",

    "x3",
    "y3",
    "z3",

    # Forces particle 1
    "Fx1",
    "Fy1",
    "Fz1",

    # Forces particle 2
    "Fx2",
    "Fy2",
    "Fz2",

    # Forces particle 3
    "Fx3",
    "Fy3",
    "Fz3",
]


df = pd.DataFrame(
    np.hstack(
        [
            X,
            Y,
        ]
    ),
    columns=columns,
)

csv_path = OUTPUT_DIR / "dataset.csv"

df.to_csv(
    csv_path,
    index=False,
)

print()
print(f"CSV сохранён:")
print(csv_path)


# ============================================================
# ПЕЧАТЬ ПРИМЕРА
# ============================================================

print()
print("=" * 60)
print("ПРИМЕР")
print("=" * 60)

print()
print("Координаты:")
print()

print("Particle 1:")
print("[0.0, 0.0, 0.0]")

print()
print("Particle 2:")
print(X[0, 0:3])

print()
print("Particle 3:")
print(X[0, 3:6])

print()
print("Силы:")
print()

print(
    Y[0].reshape(
        3,
        3,
    )
)

print()
print("Расстояния:")
print(distances[0])


# ============================================================
# СТАТИСТИКА
# ============================================================

print()
print("=" * 60)
print("СТАТИСТИКА")
print("=" * 60)

print()
print("Расстояния:")

print(
    f"min  = {distances.min():.6f}"
)

print(
    f"max  = {distances.max():.6f}"
)

print(
    f"mean = {distances.mean():.6f}"
)

print(
    f"std  = {distances.std():.6f}"
)


print()
print("Модули сил:")

print(
    f"min  = {force_norms.min():.6e}"
)

print(
    f"max  = {force_norms.max():.6e}"
)

print(
    f"mean = {force_norms.mean():.6e}"
)

print(
    f"std  = {force_norms.std():.6e}"
)


print()
print("=" * 60)
print("ГОТОВО")
print("=" * 60)

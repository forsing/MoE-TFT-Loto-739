#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
MoELite i TFTLite arhitektura

    MoE:
        - dva MLP eksperta;
        - skriveni slojevi širine 96 i 160;
        - naučeni usmerivač zavisan od istorijskog konteksta.

    TFTLite:
        - LSTM, hidden_size=96;
        - četiri glave pažnje;
        - naučena sigmoidna kapija;
        - objedinjavanje vremenske sekvence.

Loto:
    - Loto 7/39;
    - tri nezavisna CSV-a;
    - distribucioni opisi kombinacija;
    - regresija kumulativnih distribucija;
    - analitička inicijalizacija bez generatora slučajnih brojeva;
    - bez dropout-a i mešanja uzoraka;
    - hronološki izbor dužine treniranja;
    - jedan završni NEXT po CSV-u.

Kontekst:
    Poslednjih 30 izvlačenja za svaki trening primer.
    Svi raspoloživi istorijski primeri ulaze u finalno treniranje.

Izlaz:
    Sedam distribucija sortiranih pozicija kombinacije.
    Ne predstavljaju fizički redosled izvlačenja.

Izbor:
    Dinamičko programiranje pronalazi maksimum proizvoda
    izlaznih masa za sedam strogo rastućih brojeva 1..39.

CSV:
    Bez zaglavlja.
    Sedam različitih celih brojeva 1..39 po redu.
    Najstarije izvlačenje prvo, najnovije poslednje.
"""


import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit(
        "Potrebne biblioteke: numpy i torch.\n"
        "Instalacija: python3 -m pip install numpy torch\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

WINDOW = 30
VALIDATION = 48
TEST = 64
MIN_ROWS = 300

MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 128

MODEL_NAMES = ("MoE", "TFTLite")
GRID = np.arange(1, 40)
PAIR_I, PAIR_J = np.triu_indices(7, 1)


def load_csv(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona; dobijeno {len(row)}."
                )

            try:
                values = [int(value.strip()) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "potrebni su celi brojevi."
                ) from exc

            if len(set(values)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            if min(values) < 1 or max(values) > 39:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan opsega 1..39."
                )

            rows.append(sorted(values))

    if len(rows) < MIN_ROWS:
        raise ValueError(
            f"{path.name}: potrebno najmanje {MIN_ROWS} izvlačenja."
        )

    draws = np.asarray(rows, dtype=np.int64)

    canonical = "\n".join(
        ",".join(map(str, row)) for row in draws
    ).encode("ascii")

    fingerprint = hashlib.sha256(canonical).hexdigest()
    return path, draws, fingerprint


def draw_profile(draw):
    values = draw.astype(float)

    distances = np.abs(
        values[:, None] - values[None, :]
    )[PAIR_I, PAIR_J]

    return np.r_[
        values / 39.0,
        [np.mean(distances <= limit) for limit in (2, 5, 10, 20)],
        (values[-1] - values[0]) / 38.0,
    ]


def make_dataset(draws):
    profiles = np.asarray(
        [draw_profile(draw) for draw in draws],
        dtype=np.float32,
    )

    sequences = np.stack([
        profiles[t - WINDOW:t]
        for t in range(WINDOW, len(draws) + 1)
    ])

    targets = draws[WINDOW:]

    cdf = (
        targets[:, :, None] <= GRID[None, None, :]
    ).astype(np.float64)

    return sequences, targets, cdf


def fit_scaler(sequences):
    flat = sequences.reshape(-1, sequences.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64)
    scale = flat.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def analytic_fill(parameter, phase):
    """Deterministička inicijalizacija, bez pozivanja RNG-a."""
    with torch.no_grad():
        if parameter.ndim < 2:
            parameter.zero_()
            return

        rows = parameter.shape[0]
        columns = parameter.numel() // rows

        r = torch.arange(
            1, rows + 1, dtype=torch.float64
        )[:, None]

        c = torch.arange(
            1, columns + 1, dtype=torch.float64
        )[None, :]

        values = torch.sin(
            r * c * math.sqrt(2.0)
            + 0.113 * r.square()
            + 0.071 * c.square()
            + phase
        )

        values *= math.sqrt(2.0 / columns)

        parameter.copy_(
            values.reshape(parameter.shape).to(parameter.dtype)
        )


class Linear(nn.Module):
    def __init__(self, inputs, outputs, phase):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

        analytic_fill(self.weight, phase)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class DeterministicLSTM(nn.LSTM):
    def reset_parameters(self):
        for index, (name, parameter) in enumerate(
            self.named_parameters()
        ):
            analytic_fill(parameter, 0.37 + index * 0.19)

            if "bias" in name:
                with torch.no_grad():
                    hidden = parameter.numel() // 4
                    parameter[hidden:2 * hidden].fill_(0.5)


def distribution_from_logits(logits):
    logits = logits.reshape(-1, 7, 39).to(torch.float64)

    intensity = F.softplus(logits)
    intensity = intensity.clamp_min(
        torch.finfo(torch.float64).tiny
    )

    return intensity / intensity.sum(dim=-1, keepdim=True)


class MLPExpert(nn.Module):
    def __init__(self, inputs, hidden, phase):
        super().__init__()

        self.first = Linear(inputs, hidden, phase)
        self.second = Linear(hidden, hidden, phase + 0.23)
        self.output = Linear(hidden, 7 * 39, phase + 0.51)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = F.relu(self.second(x))
        return distribution_from_logits(self.output(x))


class MoELite(nn.Module):
    def __init__(self, features):
        super().__init__()
        inputs = WINDOW * features

        self.expert_1 = MLPExpert(inputs, 96, 0.17)
        self.expert_2 = MLPExpert(inputs, 160, 0.83)

        self.router_first = Linear(inputs, 32, 1.31)
        self.router_output = Linear(32, 2, 1.67)

    def forward(self, sequence, return_router=False):
        flat = sequence.reshape(sequence.shape[0], -1)

        first = self.expert_1(flat)
        second = self.expert_2(flat)

        router_hidden = F.relu(self.router_first(flat))
        weights = F.softmax(
            self.router_output(router_hidden), dim=1
        ).to(torch.float64)

        output = (
            first * weights[:, 0, None, None]
            + second * weights[:, 1, None, None]
        )

        if return_router:
            return output, weights

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, width=96, heads=4):
        super().__init__()

        self.width = width
        self.heads = heads
        self.head_width = width // heads

        self.query = Linear(width, width, 0.29)
        self.key = Linear(width, width, 0.53)
        self.value = Linear(width, width, 0.79)
        self.output = Linear(width, width, 1.07)

    def forward(self, x):
        batch, steps, _ = x.shape

        def split(value):
            return value.reshape(
                batch, steps, self.heads, self.head_width
            ).transpose(1, 2)

        query = split(self.query(x))
        key = split(self.key(x))
        value = split(self.value(x))

        weights = F.softmax(
            (query @ key.transpose(-2, -1))
            / math.sqrt(self.head_width),
            dim=-1,
        )

        context = (weights @ value).transpose(1, 2).reshape(
            batch, steps, self.width
        )

        return self.output(context)


class TFTLite(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.lstm = DeterministicLSTM(
            input_size=features,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.attention = MultiHeadAttention(96, 4)
        self.gate = Linear(96, 96, 1.43)
        self.output = Linear(96, 7 * 39, 1.89)

    def forward(self, sequence):
        states, _ = self.lstm(sequence)
        attended = self.attention(states)

        gate = torch.sigmoid(self.gate(attended))
        context = (attended * gate).mean(dim=1)

        return distribution_from_logits(self.output(context))


NETWORKS = {
    "MoE": MoELite,
    "TFTLite": TFTLite,
}


class Trainer:
    def __init__(self, name):
        self.name = name

    def transform(self, sequences):
        return np.ascontiguousarray(
            (sequences - self.mean) / self.scale,
            dtype=np.float32,
        )

    def fit(self, sequences, cdf, epochs, validation=None):
        self.mean, self.scale = fit_scaler(sequences)

        x = torch.from_numpy(self.transform(sequences))
        y = torch.from_numpy(
            np.ascontiguousarray(cdf, dtype=np.float64)
        )

        self.model = NETWORKS[self.name](sequences.shape[-1])

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001
        )

        if validation is not None:
            valid_x, valid_y = validation

            vx = torch.from_numpy(self.transform(valid_x))
            vy = torch.from_numpy(
                np.ascontiguousarray(valid_y, dtype=np.float64)
            )

        best_loss = float("inf")
        best_state = None
        best_epoch = epochs
        stale = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            # Bez shuffle-a: redosled ostaje hronološki.
            for start in range(0, len(x), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(x))

                pmf = self.model(x[start:end])
                loss = F.mse_loss(
                    pmf.cumsum(dim=-1), y[start:end]
                )

                value = float(loss.item())

                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"{self.name}: nevažeći trening gubitak."
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )

                optimizer.step()
                epoch_loss += value * (end - start)

            if validation is not None:
                self.model.eval()

                with torch.no_grad():
                    valid_loss = float(F.mse_loss(
                        self.model(vx).cumsum(dim=-1), vy
                    ).item())

                if not math.isfinite(valid_loss):
                    raise FloatingPointError(
                        f"{self.name}: nevažeći validacioni gubitak."
                    )

                if valid_loss < best_loss - 1e-8:
                    best_loss = valid_loss
                    best_epoch = epoch
                    stale = 0

                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stale += 1

                if stale >= PATIENCE:
                    break

            if epoch % 5 == 0:
                print(
                    f"    {self.name}: epoha {epoch}/{epochs}; "
                    f"CDF MSE={epoch_loss / len(x):.7f}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.epochs = best_epoch
        return self

    def predict(self, sequences):
        x = torch.from_numpy(self.transform(sequences))

        outputs = []
        router_outputs = []

        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(x), BATCH_SIZE):
                batch = x[start:start + BATCH_SIZE]

                if self.name == "MoE":
                    pmf, router = self.model(
                        batch, return_router=True
                    )
                    router_outputs.append(router.cpu().numpy())
                else:
                    pmf = self.model(batch)

                outputs.append(pmf.cpu().numpy())

        result = np.concatenate(outputs, axis=0)

        if (
            not np.all(np.isfinite(result))
            or np.any(result <= 0)
        ):
            raise FloatingPointError(
                f"{self.name}: nevažeća izlazna distribucija."
            )

        router = (
            np.concatenate(router_outputs, axis=0)
            if router_outputs else None
        )

        return result, router


def choose_combination(pmf):
    """
    Tačan maksimum modelskog proizvoda sedam pozicionih masa
    među strogo rastućim kombinacijama 1..39.

    Ne dodaje pravila za paritet, polovine, razmake ili nizove.
    """
    pmf = np.asarray(pmf, dtype=float)

    if pmf.shape != (7, 39):
        raise ValueError("Izlazna distribucija mora imati oblik (7, 39).")

    if not np.all(np.isfinite(pmf)) or np.any(pmf <= 0):
        raise FloatingPointError("Nevažeće mase za izbor kombinacije.")

    logarithms = np.log(pmf)

    best = np.full((7, 39), -np.inf)
    previous = np.full((7, 39), -1, dtype=int)

    # Za prvu sortiranu poziciju ostaje prostor za još šest brojeva.
    best[0, :33] = logarithms[0, :33]

    for position in range(1, 7):
        for number_index in range(position, 33 + position):
            before = int(np.argmax(best[position - 1, :number_index]))

            best[position, number_index] = (
                best[position - 1, before]
                + logarithms[position, number_index]
            )
            previous[position, number_index] = before

    last = int(np.argmax(best[6]))
    score = float(best[6, last])

    if not math.isfinite(score):
        raise FloatingPointError("Nema legalne završne kombinacije.")

    selected = [last]

    for position in range(6, 0, -1):
        last = int(previous[position, last])

        if last < 0:
            raise ValueError("Neispravna putanja izbora kombinacije.")

        selected.append(last)

    selected.reverse()
    combination = tuple(index + 1 for index in selected)

    if len(set(combination)) != 7:
        raise ValueError("Završna kombinacija ima ponovljen broj.")

    return combination, score


def validation_weight(moe, tft, target_cdf):
    """
    Analitički izbor pondera koji minimizuje validacioni CDF MSE.
    Završna provera ne učestvuje u izboru.
    """
    first = moe.cumsum(axis=-1)
    second = tft.cumsum(axis=-1)

    difference = first - second
    denominator = float(np.sum(difference ** 2))

    if denominator <= 1e-18:
        return 1.0

    numerator = float(np.sum(
        difference * (target_cdf - second)
    ))

    return float(np.clip(numerator / denominator, 0.0, 1.0))


def longest_run(combination):
    maximum = current = 1

    for before, number in zip(combination[:-1], combination[1:]):
        current = current + 1 if number == before + 1 else 1
        maximum = max(maximum, current)

    return maximum


def fit_models(sequences, cdf, epochs):
    models = {}

    for name in MODEL_NAMES:
        print(f"  Treniranje: {name}", flush=True)

        models[name] = Trainer(name).fit(
            sequences,
            cdf,
            epochs=epochs[name],
        )

    return models


def process_dataset(path, draws, fingerprint):
    sequences, targets, cdf = make_dataset(draws)

    count = len(targets)
    test_start = count - TEST
    validation_start = test_start - VALIDATION

    print("\n" + "=" * 76)
    print(path.name)
    print(f"Izvlačenja: {len(draws)}")
    print(f"Prelaza za finalno treniranje: {count}")
    print(f"SHA-256: {fingerprint}")
    print(
        "Validacija — CSV redovi: "
        f"{WINDOW + validation_start + 1}.."
        f"{WINDOW + test_start}"
    )
    print(
        "Završna provera — CSV redovi: "
        f"{WINDOW + test_start + 1}..{len(draws)}"
    )

    print("\n1. Izbor dužine treniranja", flush=True)

    epochs = {}
    validation_predictions = {}

    for name in MODEL_NAMES:
        print(f"  Validacija: {name}", flush=True)

        trainer = Trainer(name).fit(
            sequences[:validation_start],
            cdf[:validation_start],
            epochs=MAX_EPOCHS,
            validation=(
                sequences[validation_start:test_start],
                cdf[validation_start:test_start],
            ),
        )

        epochs[name] = trainer.epochs

        prediction, _ = trainer.predict(
            sequences[validation_start:test_start]
        )
        validation_predictions[name] = prediction

        print(f"    Izabrano epoha: {trainer.epochs}")
        del trainer

    alpha = validation_weight(
        validation_predictions["MoE"],
        validation_predictions["TFTLite"],
        cdf[validation_start:test_start],
    )

    valid_mixture = (
        alpha * validation_predictions["MoE"]
        + (1.0 - alpha) * validation_predictions["TFTLite"]
    )

    print("\nVALIDACIONI CDF MSE")

    for name in MODEL_NAMES:
        loss = np.mean((
            validation_predictions[name].cumsum(axis=-1)
            - cdf[validation_start:test_start]
        ) ** 2)

        print(f"  {name}: {loss:.7f}")

    mixture_loss = np.mean((
        valid_mixture.cumsum(axis=-1)
        - cdf[validation_start:test_start]
    ) ** 2)

    print(f"  Završna mešavina: {mixture_loss:.7f}")
    print(f"  Ponder MoE: {alpha:.6f}")
    print(f"  Ponder TFTLite: {1.0 - alpha:.6f}")

    del validation_predictions, valid_mixture

    print("\n2. Treniranje pre odvojene završne provere", flush=True)

    models = fit_models(
        sequences[:test_start],
        cdf[:test_start],
        epochs,
    )

    test_moe, test_router = models["MoE"].predict(
        sequences[test_start:count]
    )
    test_tft, _ = models["TFTLite"].predict(
        sequences[test_start:count]
    )

    test_mixture = (
        alpha * test_moe
        + (1.0 - alpha) * test_tft
    )

    combinations = []
    hit_counts = []
    control_hits = []
    copied_previous = 0

    for index, pmf in enumerate(test_mixture):
        transition = test_start + index
        combination, _ = choose_combination(pmf)

        actual = targets[transition]
        before = draws[WINDOW + transition - 1]

        combinations.append(combination)
        hit_counts.append(
            len(set(combination).intersection(actual))
        )
        control_hits.append(
            len(set(before).intersection(actual))
        )

        copied_previous += (
            tuple(combination) == tuple(before)
        )

    hit_counts = np.asarray(hit_counts)
    control_hits = np.asarray(control_hits)

    print("\nODVOJENA PROVERA — POSLEDNJA 64 KOLA")
    print(f"Prosečno pogodaka: {hit_counts.mean():.4f}")
    print(
        "Raspodela pogodaka 0..7: "
        f"{np.bincount(hit_counts, minlength=8).tolist()}"
    )
    print(f"Kola sa 3+: {int(np.sum(hit_counts >= 3))}")
    print(f"Kola sa 4+: {int(np.sum(hit_counts >= 4))}")
    print(
        "Kontrola: prethodno izvuceno kolo — "
        f"{control_hits.mean():.4f}"
    )
    print(
        "Razlika prema kontroli: "
        f"{hit_counts.mean() - control_hits.mean():+.4f}"
    )

    for name, pmf in (
        ("MoE", test_moe),
        ("TFTLite", test_tft),
        ("Mešavina", test_mixture),
    ):
        loss = np.mean((
            pmf.cumsum(axis=-1) - cdf[test_start:]
        ) ** 2)

        print(f"CDF MSE — {name}: {loss:.7f}")

    print(
        f"Različite prognoze: {len(set(combinations))}/{TEST}"
    )
    print(f"Kopije prethodnog kola: {copied_previous}")
    print(
        "Prognoze sa 5+ uzastopnih: "
        f"{sum(longest_run(item) >= 5 for item in combinations)}"
    )
    print(
        "Prognoze potpuno u jednoj polovini: "
        f"{sum(max(item) <= 19 or min(item) >= 20 for item in combinations)}"
    )

    print(
        "Ponder prvog MoE eksperta kroz završnu proveru: "
        f"min={test_router[:, 0].min():.6f}, "
        f"prosek={test_router[:, 0].mean():.6f}, "
        f"max={test_router[:, 0].max():.6f}"
    )

    del models, test_moe, test_tft, test_mixture, test_router

    print("\n3. Finalno treniranje na svim prelazima", flush=True)

    models = fit_models(
        sequences[:-1],
        cdf,
        epochs,
    )

    final_moe, final_router = models["MoE"].predict(
        sequences[-1:]
    )
    final_tft, _ = models["TFTLite"].predict(
        sequences[-1:]
    )

    final_pmf = (
        alpha * final_moe[0]
        + (1.0 - alpha) * final_tft[0]
    )

    combination, log_score = choose_combination(final_pmf)

    print("\nFINALNI NEXT: " + ", ".join(map(str, combination)))
    print(f"Modelski log-skor: {log_score:.8f}")
    print(f"Upotrebljenih trening prelaza: {count}")
    print(
        "MoE usmerivač za NEXT — ekspert 1 / ekspert 2: "
        f"{final_router[0, 0]:.6f} / "
        f"{final_router[0, 1]:.6f}"
    )
    print(
        "Završni ponder MoE / TFTLite: "
        f"{alpha:.6f} / {1.0 - alpha:.6f}"
    )
    print(f"Najduži uzastopni niz: {longest_run(combination)}")

    lower = sum(number <= 19 for number in combination)
    print(f"Brojeva 1..19 / 20..39: {lower} / {7 - lower}")

    return combination


def main():
    parser = argparse.ArgumentParser(
        description="MoE i TFTLite adaptacija za Loto 7/39."
    )
    parser.add_argument("csv", nargs="*")
    args = parser.parse_args()

    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Navedi tačno tri CSV putanje.")

    datasets = [load_csv(filename) for filename in filenames]

    if len({path for path, _, _ in datasets}) != 3:
        raise ValueError("Potrebna su tri različita CSV fajla.")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    print("LOTO PLUS MoE + TFTLite V1")
    print("Tri nezavisna CSV-a; jedan završni NEXT po CSV-u.")
    print("Cela dostupna istorija za finalno treniranje.")
    print("Kontekst svakog primera: prethodnih 30 izvlačenja.")
    print("Bez random inicijalizacije, dropout-a i shuffle-a.")
    print("CDF regresija; bez klasifikacionog gubitka.")
    print("Ulazni redosled mora biti: najstarije → najnovije.")
    print("Napredak treniranja ispisuje se svakih pet epoha.")

    results = []

    for path, draws, fingerprint in datasets:
        combination = process_dataset(path, draws, fingerprint)
        results.append((path.name, combination))

    print("\n" + "=" * 76)
    print("ZAVRŠNI NEXT PREGLED")

    for name, combination in results:
        print(f"{name}: " + ", ".join(map(str, combination)))

    grouped = {}

    for name, combination in results:
        grouped.setdefault(combination, []).append(name)

    for combination, names in grouped.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto.", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        csv.Error,
    ) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        sys.exit(1)



"""
LOTO MoE + TFTLite V1
Tri nezavisna CSV-a; jedan završni NEXT po CSV-u.
Cela dostupna istorija za finalno treniranje.
Kontekst svakog primera: prethodnih 30 izvlačenja.
Bez random inicijalizacije, dropout-a i shuffle-a.
CDF regresija; bez klasifikacionog gubitka.
Ulazni redosled mora biti: najstarije → najnovije.
Napredak treniranja ispisuje se svakih pet epoha.


============================================================================
loto7_4704_k83.csv
Izvlačenja: 4704
Prelaza za finalno treniranje: 4674
SHA-256: 1decd748cc0c398e384d9b16fd7e7eb50a5396f59a2f2cf9e7cdd05bdd07ae09
Validacija — CSV redovi: 4593..4640
Završna provera — CSV redovi: 4641..4704

1. Izbor dužine treniranja
  Validacija: MoE
    MoE: epoha 5/50; CDF MSE=0.0700690
    Izabrano epoha: 2
  Validacija: TFTLite
    TFTLite: epoha 5/50; CDF MSE=0.0732978
    TFTLite: epoha 10/50; CDF MSE=0.0732632
    Izabrano epoha: 7

VALIDACIONI CDF MSE
  MoE: 0.0701015
  TFTLite: 0.0695004
  Završna mešavina: 0.0695004
  Ponder MoE: 0.000000
  Ponder TFTLite: 1.000000

2. Treniranje pre odvojene završne provere
  Treniranje: MoE
  Treniranje: TFTLite
    TFTLite: epoha 5/7; CDF MSE=0.0733476

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.2500
Raspodela pogodaka 0..7: [13, 27, 19, 5, 0, 0, 0, 0]
Kola sa 3+: 5
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.1406
Razlika prema kontroli: +0.1094
CDF MSE — MoE: 0.0744984
CDF MSE — TFTLite: 0.0737264
CDF MSE — Mešavina: 0.0737264
Različite prognoze: 1/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0
Ponder prvog MoE eksperta kroz završnu proveru: min=0.032920, prosek=0.302434, max=0.774914

3. Finalno treniranje na svim prelazima
  Treniranje: MoE
  Treniranje: TFTLite
    TFTLite: epoha 5/7; CDF MSE=0.0732636

FINALNI NEXT: 2, 8, 11, 18, 28, 33, 39
Modelski log-skor: -16.28491085
Upotrebljenih trening prelaza: 4674
MoE usmerivač za NEXT — ekspert 1 / ekspert 2: 0.277610 / 0.722390
Završni ponder MoE / TFTLite: 0.000000 / 1.000000
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3



============================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974
Prelaza za finalno treniranje: 2944
SHA-256: fc5313380978e4826789413029140d99f09bd92c143892bf9b9027ff9f276be7
Validacija — CSV redovi: 2863..2910
Završna provera — CSV redovi: 2911..2974

1. Izbor dužine treniranja
  Validacija: MoE
    MoE: epoha 5/50; CDF MSE=0.0710639
    MoE: epoha 10/50; CDF MSE=0.0510956
    Izabrano epoha: 3
  Validacija: TFTLite
    TFTLite: epoha 5/50; CDF MSE=0.0733327
    Izabrano epoha: 2

VALIDACIONI CDF MSE
  MoE: 0.0705340
  TFTLite: 0.0697871
  Završna mešavina: 0.0697871
  Ponder MoE: 0.000000
  Ponder TFTLite: 1.000000

2. Treniranje pre odvojene završne provere
  Treniranje: MoE
  Treniranje: TFTLite

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.3438
Raspodela pogodaka 0..7: [11, 27, 19, 7, 0, 0, 0, 0]
Kola sa 3+: 7
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.3125
Razlika prema kontroli: +0.0312
CDF MSE — MoE: 0.0730556
CDF MSE — TFTLite: 0.0732049
CDF MSE — Mešavina: 0.0732049
Različite prognoze: 3/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0
Ponder prvog MoE eksperta kroz završnu proveru: min=0.102989, prosek=0.375054, max=0.772699

3. Finalno treniranje na svim prelazima
  Treniranje: MoE
  Treniranje: TFTLite

FINALNI NEXT: 2, 6, 12, 18, 24, 28, 39
Modelski log-skor: -15.43116118
Upotrebljenih trening prelaza: 2944
MoE usmerivač za NEXT — ekspert 1 / ekspert 2: 0.357751 / 0.642249
Završni ponder MoE / TFTLite: 0.000000 / 1.000000
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3



============================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730
Prelaza za finalno treniranje: 1700
SHA-256: 1f4252375372cbbb88cd026892dc19eb5400e6386bfc5dc613ff7d19d9db0504
Validacija — CSV redovi: 1619..1666
Završna provera — CSV redovi: 1667..1730

1. Izbor dužine treniranja
  Validacija: MoE
    MoE: epoha 5/50; CDF MSE=0.0718065
    MoE: epoha 10/50; CDF MSE=0.0554443
    Izabrano epoha: 4
  Validacija: TFTLite
    TFTLite: epoha 5/50; CDF MSE=0.0735811
    TFTLite: epoha 10/50; CDF MSE=0.0733077
    TFTLite: epoha 15/50; CDF MSE=0.0732730
    TFTLite: epoha 20/50; CDF MSE=0.0732479
    TFTLite: epoha 25/50; CDF MSE=0.0732263
    TFTLite: epoha 30/50; CDF MSE=0.0732169
    TFTLite: epoha 35/50; CDF MSE=0.0731990
    TFTLite: epoha 40/50; CDF MSE=0.0731352
    Izabrano epoha: 35

VALIDACIONI CDF MSE
  MoE: 0.0764221
  TFTLite: 0.0757986
  Završna mešavina: 0.0757981
  Ponder MoE: 0.025406
  Ponder TFTLite: 0.974594

2. Treniranje pre odvojene završne provere
  Treniranje: MoE
  Treniranje: TFTLite
    TFTLite: epoha 5/35; CDF MSE=0.0736592
    TFTLite: epoha 10/35; CDF MSE=0.0733815
    TFTLite: epoha 15/35; CDF MSE=0.0733454
    TFTLite: epoha 20/35; CDF MSE=0.0733212
    TFTLite: epoha 25/35; CDF MSE=0.0733070
    TFTLite: epoha 30/35; CDF MSE=0.0732904
    TFTLite: epoha 35/35; CDF MSE=0.0732762

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.2188
Raspodela pogodaka 0..7: [15, 27, 16, 5, 1, 0, 0, 0]
Kola sa 3+: 6
Kola sa 4+: 1
Kontrola: prethodno izvuceno kolo — 1.2812
Razlika prema kontroli: -0.0625
CDF MSE — MoE: 0.0735793
CDF MSE — TFTLite: 0.0726928
CDF MSE — Mešavina: 0.0726994
Različite prognoze: 2/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0
Ponder prvog MoE eksperta kroz završnu proveru: min=0.043888, prosek=0.376774, max=0.893739

3. Finalno treniranje na svim prelazima
  Treniranje: MoE
  Treniranje: TFTLite
    TFTLite: epoha 5/35; CDF MSE=0.0736108
    TFTLite: epoha 10/35; CDF MSE=0.0733670
    TFTLite: epoha 15/35; CDF MSE=0.0733332
    TFTLite: epoha 20/35; CDF MSE=0.0733051
    TFTLite: epoha 25/35; CDF MSE=0.0732940
    TFTLite: epoha 30/35; CDF MSE=0.0732839
    TFTLite: epoha 35/35; CDF MSE=0.0732586

FINALNI NEXT: 1, 8, 11, 23, 27, 31, 39
Modelski log-skor: -16.11428764
Upotrebljenih trening prelaza: 1700
MoE usmerivač za NEXT — ekspert 1 / ekspert 2: 0.499793 / 0.500207
Završni ponder MoE / TFTLite: 0.025406 / 0.974594
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 3 / 4



============================================================================
ZAVRŠNI NEXT PREGLED
loto7_4704_k83.csv: 2, 8, 11, 18, 28, 33, 39
loto7_4704_k83_loto_2974.csv: 2, 6, 12, 18, 24, 28, 39
loto7_4704_k83_loto_plus_1730.csv: 1, 8, 11, 23, 27, 31, 39
"""



"""
Mixture of Experts — MoE 

- MoE: dve neuronske mreže i naučeni usmerivač koji, prema trenutnom istorijskom kontekstu, određuje njihove pondere.
- TFTLite: LSTM, višeglava pažnja i naučena kapija koja reguliše prolazak informacija.

ponderi ovde zavise od ulaznog konteksta, a ne samo od ranijih grešaka eksperata 
MoE sa dve mreže i naučenim usmerivačem, kao i TFTLite: LSTM → pažnja → kapija. 
Validacija određuje njihov završni ponder. 
Izbor sedam brojeva računa se zajednički, bez zaokruživanja regresionih izlaza i bez filtera za raspored kombinacije.
"""

# LeNet-5 baseline

Clean, standard LeNet-5-style baseline for MNIST-sized grayscale images.

```text
[B, 1, 28, 28]
  → Conv5x5 (1 → 6) → Tanh → AvgPool2x2
  → Conv5x5 (6 → 16) → Tanh → AvgPool2x2
  → Conv5x5 (16 → 120) → Tanh
  → Linear (120 → 84) → Tanh
  → Linear (84 → 10)
```

The model uses the conventional dense `6→16` second convolution. The historical 1998 LeNet-5 used a hand-selected partial connectivity pattern in C3; that detail is normally omitted in modern LeNet-5 implementations.

Run a shape smoke test:

```bash
python model.py
```

Train on MNIST (downloads data when needed):

```bash
python train.py --epochs 10
```

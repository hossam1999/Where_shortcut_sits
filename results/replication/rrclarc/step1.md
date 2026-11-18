# P6 (Dreyer, Pahde et al., "From Hope to Safety", RR-ClArC, AAAI 2024) step 1: released Vanilla ISIC checkpoints (machine 2)

Their code (`rrclarc` @ a3e2d15), configs from their `config_generator_correcting_isic_attacked.py` (Vanilla, 0 epochs = the released checkpoint), split `default_rng(0)` 80/10/10 (test n = 2,533), LSB trigger in the lowest 3 bits (VGG-16, ResNet-18) or 2 bits (EfficientNet-B0) of every pixel. Biased test = trigger in every test image (their `attacked`, all classes). Each set scored once, on CPU (fp32). Script `scripts/replication/p6_step1.py`; all rows (test and validation; clean / attacked / as in training) in `step1.csv`.

| model | clean acc. ours | paper | Δ | biased acc. ours | paper | Δ | within 0.03 |
|---|---|---|---|---|---|---|---|
| vgg16 | 0.764 | 0.762 | +0.002 | 0.349 | 0.349 | -0.000 | yes |
| resnet18 | 0.816 | 0.818 | -0.002 | 0.557 | 0.568 | -0.011 | yes |
| efficientnet_b0 | 0.839 | 0.842 | -0.003 | 0.625 | 0.629 | -0.004 | yes |

**Vanilla reproduced for all three architectures** (Table 1, ISIC column). Next (A28): the three RR-ClArC fine-tunes at the paper's λ.

# plato-diffusion

Progressive distillation pipeline for PLATO room intelligence.

## What It Does

Trains progressively smaller models through distillation so PLATO rooms can handle more situations locally. Takes large model knowledge and compresses it down to models that run on edge devices like ESP32s.

## Ecosystem

- **[plato-tiles](https://github.com/SuperInstance/plato-tiles)** ← Depends on (tile types for training data)
- **[plato-nervous](https://github.com/SuperInstance/plato-nervous)** — Distillation targets nervous system models
- **[plato-signal-chain](https://github.com/SuperInstance/plato-signal-chain)** — Distilled models deploy into the pipeline
- **[plato-state](https://github.com/SuperInstance/plato-state)** — State vectors provide training labels

See [DEPENDENCIES.md](./DEPENDENCIES.md) for the full dependency map.

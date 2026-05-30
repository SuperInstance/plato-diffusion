# DEPENDENCIES — plato-diffusion

## Signal Chain Layer

**L1-L2 (Training)** — Progressive distillation pipeline.

Progressive distillation for PLATO room intelligence. Trains and compresses models so rooms can handle more situations locally at lower cost.

## Ecosystem Dependencies

| Repo | Relationship | Description |
|------|-------------|-------------|
| [plato-tiles](https://github.com/SuperInstance/plato-tiles) | **Depends on** | Tile types that define the training data format |
| [plato-nervous](https://github.com/SuperInstance/plato-nervous) | **Related** | Distillation targets the nervous system models |
| [plato-signal-chain](https://github.com/SuperInstance/plato-signal-chain) | **Related** | Distilled models are deployed into the signal chain |
| [plato-state](https://github.com/SuperInstance/plato-state) | **Related** | State vectors provide training labels |

## Data Flow

```
IN:
  - Tile history (from plato-tiles)
  - Model weights to distill
  - Training configuration (layer targets, compression ratios)

OUT:
  - Distilled model weights (progressively smaller)
  - Compression metrics per distillation step
  - Quality benchmarks (accuracy retention)
```

## Dependency Graph Position

```
plato-tiles
  ↓
plato-diffusion ← (this crate, consumes tile history for training)
```

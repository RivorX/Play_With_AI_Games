# Versioned model system

There are two independent versions:

- `architecture_id` identifies executable structure, for example `se_cnn_v9`
  or a future `transformer_v1`.
- `model.version` is the trained series/run label. Many IL/RL checkpoints may
  share one architecture.

Every new checkpoint stores `model_spec`, so loaders do not reconstruct a model
with whichever YAML happens to be active today.

## Add an architecture

1. Add `src/models/architecture/<architecture_id>.py`.
2. Register its class in `MODEL_REGISTRY`.
3. Add `config/models/<architecture_id>.yaml`.
4. Add its encoder, policy codec and IL dataset under
   `src/models/data/<architecture_id>/`.
5. Register the contract in `src/models/data/registry.py`.
6. Select it with `project.active_architecture` in
   `config/default.yaml` when starting a new training run.

Models compared by play/Elo may use different architectures and history depth.
MCTS stays shared as long as both architectures expose the common forward
contract: compact policy logits plus WDL logits. A different action space needs
a corresponding policy codec integration in MCTS, not a second MCTS copy.

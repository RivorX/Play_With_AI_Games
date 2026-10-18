# CNN v10 — nested trunk, FVI and One-BN

`nested_cnn_v10` is the current v10 architecture profile. It keeps the six-block nested trunk and uses Fixed Variance Initialization (FVI), one training-only trunk BatchNorm, independent normalized and deployment heads, and an auxiliary soft-policy target. v9 remains a separate, unchanged architecture.

```text
72 planes → CoordConv 3×3 → 192 trunk channels
→ 6 nested residual blocks (192 / 96; two inner residual units each)
→ policy and value heads
```

- **Input and repetition:** 72-plane `planes_history_v2` input. New IL training uses exact repetition sidecars; each checkpoint records its input and repetition contract.
- **Trunk and context:** six outer blocks remain. Blocks 2, 4 and 6 use global mean/max gated affine modulation of the 96-channel residual map. Each module predicts a channel gate and shift; its projection is zero-initialized, making the initial gate exactly one and shift zero. The residual branch therefore starts with the same behavior as the former neutral additive context, then can learn to amplify or suppress channels as well as shift them. The policy-head context remains additive. Inner depth and head shapes are unchanged.
- **FVI and learned offsets:** every one of the 48 fixed positive FVI scales has a zero-initialized per-channel `beta`. Learned `gamma` is not used. These scales follow the residual graph's unit-variance approximation; they are initialization structure, not runtime statistics. Convolutions use fan-in He initialization. WDL hidden layers use fan-in He because LayerNorm normalizes their input. This profile has 4,172,526 training parameters, including 5,824 FVI beta parameters; 3,251,150 are active in ordinary inference. The gated context adds 55,584 parameters (18,528 per context block).
- **One-BN training:** the raw trunk feeds one BatchNorm and an independently learned policy/value/auxiliary head set. The normalized branch receives 80% of the ordinary policy/value loss and the deployment branch receives 20%. Evaluation and play use the deployment branch. The BN branch and its running statistics are training-only.
- **Soft policy:** each branch has a separate lightweight 1858-logit projection from its policy features. Training softens the MCTS visit distribution with temperature `4.0` and weight `0.25`. These are conservative chess settings; KataGo's Go-specific loss weights are not assumed to transfer.
- **Inference folding:** central eager inference folds adjacent FVI scales and learned offsets into a disposable eval copy. It removes 47 of 48 scale operations; 18 pre-ReLU thresholds remain as activation-dtype shifts and one shared trunk scale remains. Source training weights and checkpoints are unchanged. CUDA latency and playing-strength gains have not been established by this folding check.
- **Eager inference dtype cache:** the disposable copy stores non-persistent FP16 and BF16 versions of the 18 retained shifts. Eager inference reuses these values instead of converting each small offset on every forward. The cache does not change checkpoint keys or training parameters; compiled inference keeps its existing path. A local full-model CUDA probe was directionally faster at batches 1 and 32, nearly unchanged at batch 8, with noisy timings; no end-to-end match-throughput gain is claimed.
- **Checkpoint contract:** only the current v10 parameter layout is supported for `nested_cnn_v10`. Earlier V10 checkpoints are not migrated; start a fresh v10 run. v9 checkpoints continue to use their own model contract.

The gated context is part of the current v10 graph and requires fresh v10 training. It is not an Elo claim: compare against the former additive-context design with matched training and paired-opening evaluation before attributing a playing-strength gain.

## Training and checkpoint use

Select `nested_cnn_v10.yaml` in the IL training prompt and start with **New**. The profile uses exact repetition sidecars and rebuilds aligned PGN sidecars when needed. Checkpoints store the architecture, loss settings, input encoding and repetition contract. The current v10 parameter layout is the only supported layout; earlier v10 checkpoints require a fresh run, while v9 keeps its separate contract.

`return_training_aux=True` is reserved for `model.train()` with `apply_log_softmax=False`. It exposes `main` and `bn` policy/value tuples plus `soft_policy` and `bn_soft_policy` logits. Those auxiliary projections and the BN branch are skipped by ordinary forward calls; gameplay and validation use the deployment branch. Moves-left and search-Q outputs remain optional.

A bounded CPU probe found fan-out initialization gave the LayerNorm-fed value hidden layer pre-activation standard deviations of 4.48–5.32. Fan-in He reduced them to 1.26–1.49 and hidden SiLU RMS from 3.03–3.57 to 0.78–0.91. Neither variant saturated WDL probabilities, so this corrects initialization scale and early gradient size; it does not establish an Elo gain.

The implementation is [`nested_cnn_v10.py`](../../src/models/architecture/nested_cnn_v10.py), and the selected profile is [`nested_cnn_v10.yaml`](../../config/models/nested_cnn_v10.yaml). The FVI beta and auxiliary-target ideas are informed by [KataGo's methods documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) and [its PyTorch training implementation](https://github.com/lightvector/KataGo/blob/master/python/katago/train/model_pytorch.py). They do not prove the same choices improve chess Elo.

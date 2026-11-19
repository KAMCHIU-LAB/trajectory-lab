# Frozen public response experts

These assets preserve inference from three response experts trained in this
repository's public AuctionNet-derived, fixed-opponent simulation. They contain
no commercial data or commercial code. The simulator source and its retained
license are in [`third_party/auctionnet`](../../third_party/auctionnet/README.md).

Each `response-{seed}.npz` contains a numeric state dictionary; its matching
`response-{seed}.json` contains allowlisted architecture, normalization, return
scale and provenance metadata. Loading uses `numpy.load(..., allow_pickle=False)`
and does not deserialize a pickle checkpoint. Readers do not need the original
checkpoint files or their original storage locations.

| Seed | Files | Original checkpoint SHA-256 |
| --- | --- | --- |
| 11 | `response-11.npz`, `response-11.json` | `d7c09784f3748e5e880ec0cfe08af3c018458427f47d0a23474ee1bf6bd5baf0` |
| 23 | `response-23.npz`, `response-23.json` | `407a780646b825a48b3700984a711b6c89649ca9849d124e8851867ef63ae9d0` |
| 37 | `response-37.npz`, `response-37.json` | `ae388409c27c21782bc4aa71d8bcc1db88f9e36b7689b385f1b3a27429c855d9` |

The original checkpoint hash records conversion provenance. The separate
`weights_sha256` in each JSON binds the distributed NPZ bytes. The
[`PublicResponse` loader](../../src/trajectorylab/bidding_public_response.py)
checks those bytes, metadata, state-dictionary keys, tensor shapes, dtypes and
finite values before constructing the model. Its contract is CPU inference,
context 24 and the original 11-action grid.

From a source checkout with the project dependencies installed:

```bash
python - <<'PY'
from trajectorylab.bidding_public_response import PublicResponse

expert = PublicResponse(seed=11)
print({"seed": expert.seed, "context": expert.context, "device": expert.device})
PY
```

The expert-selection study fixes **seed 11 before development**. Seeds 23 and 37
are provided for asset provenance and separate studies; choosing among them from
this study's validation or confirmation outcomes would change the protocol.
The response expert consumes immediate feedback. A delayed-feedback adapter
would be a different expert identity requiring separate evaluation.

These are conversions of frozen checkpoints, not a promise of complete bitwise
reproduction of their original training. For source checkout requirements,
synthetic input generation, freeze records and the gated experiment commands,
see the [expert-selection reproduction guide](../../docs/bidding-expert-reproduction.md).

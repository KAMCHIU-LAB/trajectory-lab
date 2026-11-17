# AuctionNet source subset

Upstream: https://github.com/alimama-tech/AuctionNet
Revision: c20de631a40e24669d1d6a8e2acefd8f492f172d
License: Apache-2.0, reproduced in LICENSE.txt.

- NeurIPSPvGen.py: upstream simul_bidding_env/PvGenerator/NeurIPSPvGen.py,
  unchanged. Import-time global RNG/logging side effects are isolated by our loader.
- BiddingEnv.py: upstream simul_bidding_env/Environment/BiddingEnv.py; removed
  `import gin` and `@gin.configurable`; added a prominent modification/license
  notice. No auction-rule edit.

These files retain their upstream license separately from the independent MIT
code. See docs/bidding-protocol.md for study assumptions and runtime caching.

# Upstream OFA reference

The official research code was downloaded for verification from:

- Repository: https://github.com/watml/one-for-all
- Commit: `f5e9da3f6f35f5fee2de1c89dbd779ed25c505cd`
- Commit date: 2024-12-11
- Paper: *One Sample Fits All: Approximating All Probabilistic Values
  Simultaneously and Efficiently*, NeurIPS 2024

The local snapshot is stored unchanged in `ofa_upstream/` and excluded from
version control.

The upstream repository does not contain a license file as of the pinned
commit.  Consequently, the new implementation in `frame_ofa/` is kept as a
clean-room, paper-derived extension rather than copying and editing upstream
source.  Check redistribution rights with the upstream authors before
publishing a combined derivative repository.

To reproduce the local reference checkout:

```bash
git clone https://github.com/watml/one-for-all ofa_upstream
git -C ofa_upstream checkout f5e9da3f6f35f5fee2de1c89dbd779ed25c505cd
```

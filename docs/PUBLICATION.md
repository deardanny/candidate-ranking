# Publication and evidence notes

The public project contains two distinct kinds of evidence: observational ranking experiments on licensed logs, and behavioral simulations used to exercise an experiment system. Their rates must not be pooled or described as a measured human advertising lift.

## Run the simulations from the public code

Follow the environment setup in the root README. The simulations require no FairJob or EB-NeRD downloads, commercial API keys, LLM service or real-user data. They train their own policies from seeded synthetic exposures. They bind a loopback HTTP port and write local SQLite logs.

Use fresh output locations so a reproduction does not overwrite published results:

```bash
python3 scripts/run.py -m fairjob_ranker.simulation.runner \
  --out runs/synthetic-ab-reproduction --report reports/synthetic-ab-reproduction

python3 scripts/run.py -m fairjob_ranker.simulation.campus_run \
  --out runs/campus-coupons-reproduction --report reports/campus-coupons-reproduction
```

The exact behavioral configuration and original execution receipts are published beside each laboratory's results. Model randomness is seeded, but platform/library differences can affect training. Wall-clock timestamps, timings and log-file hashes will differ between executions.

- [Synthetic click laboratory](SYNTHETIC_AB.md), [results](../reports/synthetic-ab-v1/results.json), [frozen protocol](../reports/synthetic-ab-v1/protocol.json).
- [Campus coupon laboratory](CAMPUS_COUPONS.md), [results](../reports/campus-coupons-v1/results.json), [frozen protocol](../reports/campus-coupons-v1/protocol.json).
- [Original Korean analysis report](../output/pdf/ads-ranking-analysis-report-ko.pdf), dated 21 November 2026. References to not-yet-published simulations describe its preparation date, before this publication update.

The report PDF is preserved byte-for-byte. Its manifest records the original source-report hashes; some sources are research-only records and are not included in the public repository. In the published simulation JSON copies, the local absolute `run_directory` is replaced with a repository-relative path. Measurements, source-code hashes and original execution-evidence hashes are unchanged. Consequently a sanitized JSON copy can have a different file hash from the original listed in the PDF manifest.

## What was verified

The original simulations completed through HTTP and reconciled client traces with durable server logs. Each laboratory recorded 72,000 ranking requests and closed click outcomes. The campus laboratory additionally recorded 72,000 closed coupon outcomes. Missing feedback blocks effect estimation; it is not silently counted as a negative outcome. Each run includes user-stable assignment, an A/A condition, duplicate-event checks and HTTP/direct prediction parity.

`verification.json` in each result directory preserves the test count and checks from the original run, rather than relabeling them as a later rerun. The README reports validation of the current release. Full local logs, simulated persona files and model bundles are excluded; the public code regenerates them.

To regenerate figures from the published aggregates after installing `requirements-figures.txt`:

```bash
python3 scripts/run.py scripts/render_synthetic_ab.py
python3 scripts/run.py scripts/render_campus_coupons.py
```

The simulators contain invented preferences and constraints. The campus result reverses when travel constraints are removed. Bootstrap intervals describe variation inside those invented populations, not uncertainty about real customers. No human A/B test, advertiser revenue gain or deployment of the latest EB-NeRD response model is claimed.

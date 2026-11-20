# Dataset access and publication scope

This repository distributes implementation code, synthetic test fixtures, aggregate evaluation evidence and derived research figures. It does not distribute source datasets, article text, user histories, row-level predictions, trained model bundles, replay requests or local credentials.

## FairJob

FairJob is published by Criteo. Its dataset card specifies **CC BY-NC-SA 4.0** and describes the advertising campaign, anonymized features and positional-bias caveats.

- [Official dataset and access](https://huggingface.co/datasets/criteo/FairJob)
- [FairJob paper](https://arxiv.org/abs/2407.03059)
- [License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Review the publisher's terms before downloading or using the data. `scripts/download_fairjob.py` records the revision and verifies the downloaded file. This project studies ranking and serving; it does not claim to implement the dataset's full fairness benchmark.

## EB-NeRD

EB-NeRD is published by Ekstra Bladet and was used for the RecSys 2024 Challenge.

- [Official dataset website, registration and documentation](https://recsys.eb.dk/)
- [Publisher's research license](https://recsys.eb.dk/assets/pdf/general_license_terms.pdf)
- [Official benchmark repository](https://github.com/ebanalyse/ebnerd-benchmark)

Each user must obtain access and accept the applicable publisher terms independently. The author's completed registration does not grant access rights to anyone cloning this repository. The small-bundle download script requires `--registered` as an explicit acknowledgement that registration is already complete.

The metadata-free experiment uses the small dataset as its reference stream and a disjoint user sample from large validation for evaluation. Obtain the official large bundle separately; use the archive path specified in `scripts/prepare_fresh_users.py`. Locally recorded checksums verify the artifacts used in this experiment; they are not publisher-issued checksums.

## What the published reports contain

Published results are aggregate metrics, model settings, protocols, counts, code/artifact hashes and verification summaries. The research figures are generated from these aggregate JSON files. Input definitions such as `user_id` in source code and schemas are field names, not a release of user records.

Public source code is not a grant of rights to either dataset. Refer to the original publishers for use and redistribution conditions.

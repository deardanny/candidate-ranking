"""Explicit feature allowlist: IDs, labels and original positions cannot enter X."""
VERSION = "fairjob-v1"
CATEGORICAL = [f"cat{i}" for i in range(13)]
NUMERICAL = [f"num{i}" for i in range(16, 51)]
FEATURES = CATEGORICAL + NUMERICAL
CONTEXT = ["cat0", "cat1"] + NUMERICAL
AD_FEATURES = CATEGORICAL[2:]
IDS = ["user_id", "impression_id", "product_id"]
BINARY = ["click", "protected_attribute", "senior", "displayrandom"]
COLUMNS = IDS + CATEGORICAL + NUMERICAL + BINARY + ["rank"]
SPLITS = ["train", "validation", "test"]


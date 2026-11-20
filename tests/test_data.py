import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.data import clean_impressions, prepare, split_for_user, split_users, validate_source
from fairjob_ranker.schema import COLUMNS


def source_frame():
    frame = pd.DataFrame(0, index=range(4), columns=COLUMNS)
    frame["user_id"] = [1, 1, 2, 2]
    frame["impression_id"] = [10, 10, 20, 20]
    frame["product_id"] = [100, 200, 100, 300]
    frame["rank"] = [1, 2, 1, 2]
    frame["click"] = [0, 1, 0, 0]
    return frame


def test_quarantine_whole_impression_and_preserve_zero_clicks():
    frame = source_frame()
    repeated = frame.iloc[[0]].copy()
    repeated["protected_attribute"] = 1
    original = pd.concat([frame, repeated], ignore_index=True)
    before = original.copy(deep=True)
    clean, excluded = clean_impressions(original)
    assert set(clean.impression_id) == {20}
    assert len(clean) == 2 and clean.click.sum() == 0
    assert excluded.impression_id.tolist() == [10]
    assert excluded.repeated_product.all() and excluded.repeated_rank.all()
    pd.testing.assert_frame_equal(original, before)


def test_repeated_position_for_different_ads_is_also_quarantined():
    frame = source_frame()
    frame.loc[1, "rank"] = 1
    clean, excluded = clean_impressions(frame)
    assert set(clean.impression_id) == {20}
    assert not excluded.repeated_product.any() and excluded.repeated_rank.all()


def test_user_split_is_stable_and_group_disjoint():
    frame = pd.DataFrame({
        "user_id": np.repeat(np.arange(100), 4),
        "impression_id": np.repeat(np.arange(200), 2),
        "product_id": np.tile([1, 2], 200),
    })
    parts, assignments = split_users(frame)
    shuffled, assignments_shuffled = split_users(frame.sample(frac=1, random_state=17))
    pd.testing.assert_frame_equal(assignments, assignments_shuffled)
    assert set(assignments["split"]) == {"train", "validation", "test"}
    assert sum(map(len, parts.values())) == len(frame)
    for name in parts:
        pd.testing.assert_frame_equal(parts[name], shuffled[name])
        assert all(split_for_user(uid) == name for uid in parts[name].user_id)
        others = pd.concat([part for other, part in parts.items() if other != name])
        assert set(parts[name].user_id).isdisjoint(others.user_id)
        assert set(parts[name].impression_id).isdisjoint(others.impression_id)


@pytest.mark.parametrize("mutation,error", [
    (lambda f: f.assign(click=2), "Nonbinary"),
    (lambda f: f.assign(num16=np.nan), "nonfinite"),
    (lambda f: f.assign(user_id=[1, 2, 2, 2]), "constant per impression"),
    (lambda f: f.assign(cat2=[0, 0, 1, 0]), "constant per product_id"),
    (lambda f: f.drop(columns="rank"), "56 audited columns"),
])
def test_source_contract_rejects_invalid_data(mutation, error):
    validate_source(source_frame())
    with pytest.raises(ValueError, match=error):
        validate_source(mutation(source_frame()))


def test_checksum_failure_prevents_dataset_publication(tmp_path):
    raw = tmp_path / "bad.csv.gz"
    raw.write_bytes(b"not the official release")
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare(raw, output)
    assert not output.exists()


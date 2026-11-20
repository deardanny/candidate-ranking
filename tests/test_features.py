import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.schema import CATEGORICAL, FEATURES, NUMERICAL


def feature_frame():
    return pd.DataFrame({**{c: [0, 2] for c in CATEGORICAL}, **{c: [2., 4.] for c in NUMERICAL}})


def test_train_only_oov_and_label_allowlist(tmp_path):
    train = feature_frame().assign(click=[0, 1], rank=[1, 2], user_id=[1, 1])
    transformer = FeatureTransformer.fit(train)
    validation = feature_frame().assign(cat0=[2, 99], num16=[3., 1000.], click=[1, 0], rank=[9, 1])
    x = transformer.transform(validation)
    assert x.columns.tolist() == FEATURES
    assert x.cat0.tolist() == [2, 0]
    assert transformer.transform(train).cat0.tolist() == [1, 2]
    assert x.num16.tolist() == [0., 997.]
    assert transformer.means["num16"] == 3.0
    assert 99 not in transformer.vocabularies["cat0"]
    path = tmp_path / "features.json"
    transformer.save(path)
    restored = FeatureTransformer.load(path)
    pd.testing.assert_frame_equal(x, restored.transform(validation))
    pd.testing.assert_frame_equal(x.sort_index(), transformer.transform(validation.iloc[::-1]).sort_index())


def test_constant_numeric_column_and_unknown_category():
    train = feature_frame().assign(num16=5.)
    transformer = FeatureTransformer.fit(train)
    assert transformer.scales["num16"] == 1
    assert transformer.transform(train).num16.eq(0).all()


def test_vectorized_transform_exactly_matches_original_formula():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({**{c: rng.integers(0, 20, 100) for c in CATEGORICAL},
                          **{c: rng.normal(size=100) for c in NUMERICAL}})
    frame = pd.DataFrame({**{c: rng.integers(-1, 25, 70) for c in CATEGORICAL},
                          **{c: rng.normal(size=70) for c in NUMERICAL}}, index=np.arange(70)[::-1])
    transformer = FeatureTransformer.fit(train)
    old = pd.DataFrame(index=frame.index)
    for c in CATEGORICAL:
        mapping = {v: i + 1 for i, v in enumerate(transformer.vocabularies[c])}
        old[c] = frame[c].map(mapping).fillna(0).astype("int32")
    for c in NUMERICAL:
        old[c] = ((frame[c] - transformer.means[c]) / transformer.scales[c]).astype("float32")
    pd.testing.assert_frame_equal(transformer.transform(frame), old[FEATURES], check_exact=True)


@pytest.mark.parametrize("mutation", [
    lambda f: f.drop(columns="cat0"),
    lambda f: f.assign(cat0=.5),
    lambda f: f.assign(num16=np.inf),
])
def test_bad_feature_inputs_fail(mutation):
    transformer = FeatureTransformer.fit(feature_frame())
    with pytest.raises(ValueError):
        transformer.transform(mutation(feature_frame()))


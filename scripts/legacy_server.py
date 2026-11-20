"""Validation-only comparison app reproducing the original Week 2 transform."""
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd

from fairjob_ranker.features import validate_features
from fairjob_ranker.schema import CATEGORICAL, NUMERICAL, FEATURES
from fairjob_ranker.serving import create_app

app = create_app()
original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def legacy_lifespan(application):
    async with original_lifespan(application):
        transformer = application.state.service.bundle.transformer
        def original_transform(frame):
            validate_features(frame)
            result = pd.DataFrame(index=frame.index)
            for column in CATEGORICAL:
                mapping = {value: index + 1 for index,value in enumerate(transformer.vocabularies[column])}
                result[column] = frame[column].map(mapping).fillna(0).astype('int32')
            for column in NUMERICAL:
                result[column] = ((frame[column]-transformer.means[column])/transformer.scales[column]).astype('float32')
            if not np.isfinite(result.to_numpy()).all():
                raise ValueError('Feature transformation overflowed')
            return result[FEATURES]
        transformer.transform = original_transform
        yield


app.router.lifespan_context = legacy_lifespan

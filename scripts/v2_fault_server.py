"""Test-only shadow fault app; production v2_server has no fault controls."""
from functools import partial
import os

from fairjob_ranker.v2.api import create_app
from fairjob_ranker.v2.shadow import Shadow

app=create_app(shadow_factory=partial(Shadow,fault_delay=float(os.environ.get('TEST_SHADOW_DELAY','0')),
                                      fault_error=os.environ.get('TEST_SHADOW_ERROR')=='1'))

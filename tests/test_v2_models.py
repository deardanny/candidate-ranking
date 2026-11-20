import numpy as np
import pandas as pd
import pytest
import torch

from fairjob_ranker.v2.models import CrossLayer, PairSampler
from fairjob_ranker.v2.training import development_split


def test_cross_layer_is_full_matrix_interaction_with_residual():
    layer = CrossLayer(2)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor([[1.,2.],[3.,4.]]))
        layer.linear.bias.copy_(torch.tensor([.5,-.5]))
    original, current = torch.tensor([[2.,3.]]),torch.tensor([[4.,5.]])
    torch.testing.assert_close(layer(original,current),torch.tensor([[33.,99.5]]))


def test_pair_sampling_respects_group_labels_and_is_group_uniform():
    frame = pd.DataFrame({'impression_id':[1,1]+[2]*11+[3]*3,'click':[1,0]+[1]+[0]*10+[0]*3})
    sampler = PairSampler(frame)
    positive, negative = sampler.sample(10000,np.random.default_rng(42))
    assert frame.iloc[positive.numpy()].click.eq(1).all()
    assert frame.iloc[negative.numpy()].click.eq(0).all()
    assert np.array_equal(frame.iloc[positive.numpy()].impression_id,frame.iloc[negative.numpy()].impression_id)
    # Many negatives in group 2 must not give it ten times the group weight.
    assert .47 < frame.iloc[positive.numpy()].impression_id.eq(1).mean() < .53


def test_pairwise_loss_gradient_pushes_clicked_candidate_above_negative():
    positive = torch.tensor([-1.],requires_grad=True)
    negative = torch.tensor([1.],requires_grad=True)
    torch.nn.functional.softplus(-(positive-negative)).mean().backward()
    assert positive.grad.item() < 0 < negative.grad.item()


def test_user_split_keeps_all_user_impressions_together():
    frame = pd.DataFrame({'user_id':np.repeat(np.arange(100),4),'impression_id':np.arange(400)//2})
    train, valid = development_split(frame,'test-salt',80)
    assert len(train)+len(valid)==len(frame)
    assert not set(train.user_id)&set(valid.user_id)
    assert not set(train.impression_id)&set(valid.impression_id)

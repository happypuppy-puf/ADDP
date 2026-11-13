import torch

from addp import (
    ADDPCalibration,
    apply_addp_routing_calibrated,
    calibration_auxiliary_loss,
)


def test_routing_shapes_and_gradient_paths():
    torch.manual_seed(0)
    values = torch.randn(3, 10, 4, 5, requires_grad=True)
    masks = (torch.rand_like(values) > 0.25).float()
    calibration = ADDPCalibration()

    outputs = apply_addp_routing_calibrated(
        values, masks, torch.zeros(values.shape[-1]), calibration
    )
    conservative, aggressive, conservative_topology, aggressive_topology = outputs[:4]
    temperatures, core_ratios, scores = outputs[4:7]
    conservative_membership = outputs[7]

    assert conservative.shape == values.shape
    assert aggressive.shape == values.shape
    assert conservative_topology.shape == values.shape[:2]
    assert aggressive_topology.shape == values.shape[:2]

    # Hard routing does not carry prediction gradients into score/core maps.
    temperatures.mean().backward(retain_graph=True)
    assert calibration.a_temperature.grad is not None
    assert calibration.b_temperature.grad is not None
    assert calibration.a_high.grad is None
    assert calibration.a_low.grad is None
    assert calibration.a_recency.grad is None
    assert calibration.a_core.grad is None
    assert calibration.b_core.grad is None

    calibration.zero_grad(set_to_none=True)
    auxiliary, score_loss, core_loss = calibration_auxiliary_loss(
        scores,
        conservative_membership,
        core_ratios,
        score_weight=0.01,
        margin=10.0,
    )
    auxiliary.backward()
    for parameter in (
        calibration.a_high,
        calibration.a_low,
        calibration.a_recency,
        calibration.a_core,
        calibration.b_core,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert score_loss.item() > 0
    assert core_loss.item() >= 0


def test_full_shared_core_is_never_truncated():
    torch.manual_seed(1)
    values = torch.randn(2, 10, 3, 4)
    masks = torch.ones_like(values)
    calibration = ADDPCalibration()
    with torch.no_grad():
        calibration.a_core.fill_(20.0)

    outputs = apply_addp_routing_calibrated(
        values, masks, torch.zeros(values.shape[-1]), calibration
    )
    aggressive_topology = outputs[3]
    core_ratios = outputs[5]
    retained = (aggressive_topology == 1.0).sum(dim=1)
    expected_core = torch.floor(core_ratios * values.shape[1]).long().clamp_min(1)
    assert torch.all(retained >= expected_core)


def test_empty_patches_receive_zero_score():
    values = torch.zeros(1, 6, 2, 3)
    masks = torch.zeros_like(values)
    calibration = ADDPCalibration()
    outputs = apply_addp_routing_calibrated(
        values, masks, torch.zeros(values.shape[-1]), calibration
    )
    assert torch.equal(outputs[6], torch.zeros_like(outputs[6]))

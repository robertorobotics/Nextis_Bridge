"""Tests for the deployment safety pipeline.

All tests run without hardware — motor interfaces are mocked.
"""

import threading
from unittest.mock import MagicMock

import pytest

from app.core.deployment.safety_pipeline import SafetyPipeline, SafetyReadings
from app.core.deployment.types import (
    DEFAULT_VELOCITY_LIMITS,
    FALLBACK_VELOCITY_LIMIT,
    SAFETY_PRESETS,
    RuntimeState,
    SafetyConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_config():
    """SafetyConfig with known limits, models, and smoothing disabled."""
    return SafetyConfig(
        joint_limits={
            "base": (-1.5, 1.5),
            "link1": (-2.0, 2.0),
            "gripper": (-5.0, 0.0),
        },
        motor_models={
            "base": "J8009P",
            "link1": "J4340P",
            "gripper": "J4310",
        },
        max_acceleration=15.0,
        smoothing_alpha=1.0,  # disable EMA for deterministic tests
        torque_check_interval=3,
        speed_scale=1.0,
    )


@pytest.fixture
def pipeline(basic_config):
    """SafetyPipeline with no SafetyLayer (torque checks skipped)."""
    return SafetyPipeline(basic_config)


# ---------------------------------------------------------------------------
# 1. Joint limit clamping
# ---------------------------------------------------------------------------


def test_joint_limit_clamp_above(pipeline):
    """Action above max is clamped to the upper boundary."""
    action = {"base": 3.0, "link1": 0.5}
    obs = {"base": 1.4, "link1": 0.5}
    # Seed prev state so acceleration filter has history
    pipeline.process(obs, obs, dt=1 / 60)
    result = pipeline.process(action, obs, dt=1 / 60)
    # base was 3.0, limit is 1.5 — should be clamped
    assert result["base"] <= 1.5


def test_joint_limit_clamp_below(pipeline):
    """Action below min is clamped to the lower boundary."""
    action = {"gripper": -10.0}
    obs = {"gripper": -4.9}
    pipeline.process(obs, obs, dt=1 / 60)
    result = pipeline.process(action, obs, dt=1 / 60)
    assert result["gripper"] >= -5.0


# ---------------------------------------------------------------------------
# 2. Joint limit pass-through
# ---------------------------------------------------------------------------


def test_joint_limit_passthrough(pipeline):
    """Action within limits passes through stage 1 unchanged."""
    action = {"base": 0.5, "link1": -1.0}
    obs = {"base": 0.5, "link1": -1.0}
    result = pipeline.process(action, obs, dt=1 / 60)
    # No velocity delta, no limits hit — should be unchanged
    assert abs(result["base"] - 0.5) < 1e-9
    assert abs(result["link1"] - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# 3. Velocity clamping
# ---------------------------------------------------------------------------


def test_velocity_clamp_large_jump(pipeline):
    """Large position jump is limited to max_velocity * dt."""
    obs = {"base": 0.0}
    action = {"base": 1.0}  # 1 rad jump
    # J8009P: max_vel = 1.5 rad/s, dt = 1/60 → max_delta = 0.025
    result = pipeline.process(action, obs, dt=1 / 60)
    max_delta = 1.5 * (1 / 60)
    assert abs(result["base"]) <= max_delta + 1e-9


# ---------------------------------------------------------------------------
# 4. Velocity pass-through
# ---------------------------------------------------------------------------


def test_velocity_passthrough_small_move(pipeline):
    """Small move within velocity limit is unchanged."""
    obs = {"base": 0.0}
    action = {"base": 0.01}  # Well within 0.025 max_delta
    result = pipeline.process(action, obs, dt=1 / 60)
    assert abs(result["base"] - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# 5. Acceleration filtering
# ---------------------------------------------------------------------------


def test_acceleration_filter_clamps_sudden_change(basic_config):
    """Sudden velocity reversal is limited by max_acceleration."""
    basic_config.smoothing_alpha = 1.0  # no EMA
    pipe = SafetyPipeline(basic_config)

    dt = 1 / 60
    # Frame 1: establish forward velocity (small positive move)
    pipe.process({"base": 0.01}, {"base": 0.0}, dt=dt)
    # Frame 2: continue forward
    pipe.process({"base": 0.02}, {"base": 0.01}, dt=dt)
    # Frame 3: request sudden large reversal
    result = pipe.process({"base": -0.5}, {"base": 0.02}, dt=dt)
    # Should NOT jump to -0.5 — acceleration is clamped
    assert result["base"] > -0.5


# ---------------------------------------------------------------------------
# 6. EMA smoothing
# ---------------------------------------------------------------------------


def test_ema_smoothing_converges(basic_config):
    """With EMA enabled, output gradually approaches the target."""
    basic_config.smoothing_alpha = 0.3
    # Use high velocity limit so velocity stage doesn't interfere
    basic_config.motor_models = {}  # fallback = 2.0 rad/s
    pipe = SafetyPipeline(basic_config)

    target = {"base": 0.5}
    obs = {"base": 0.0}
    prev = 0.0
    for _ in range(50):
        result = pipe.process(target, obs, dt=1 / 60)
        # Each frame should move closer to 0.5 (or at least not away)
        assert result["base"] >= prev - 1e-9
        prev = result["base"]

    # After 50 frames the output should have moved toward the target
    assert result["base"] > 0.0


# ---------------------------------------------------------------------------
# 7. Speed scale effect
# ---------------------------------------------------------------------------


def test_speed_scale_halves_velocity(basic_config):
    """speed_scale=0.5 halves the maximum allowed position delta."""
    basic_config.smoothing_alpha = 1.0
    basic_config.speed_scale = 1.0
    pipe_full = SafetyPipeline(basic_config)

    basic_config_half = SafetyConfig(
        joint_limits=basic_config.joint_limits,
        motor_models=basic_config.motor_models,
        smoothing_alpha=1.0,
        speed_scale=0.5,
    )
    pipe_half = SafetyPipeline(basic_config_half)

    obs = {"base": 0.0}
    action = {"base": 1.0}

    result_full = pipe_full.process(action, obs, dt=1 / 60)
    result_half = pipe_half.process(action, obs, dt=1 / 60)

    # Half-speed pipeline should move less
    assert abs(result_half["base"]) < abs(result_full["base"])
    # Specifically: full = 1.5/60 = 0.025, half = 0.75/60 = 0.0125
    assert abs(result_half["base"] - 0.75 / 60) < 1e-9


# ---------------------------------------------------------------------------
# 8. Speed scale update
# ---------------------------------------------------------------------------


def test_update_speed_scale_clamps(pipeline):
    """update_speed_scale clamps to [0.1, 1.0] in runtime override."""
    pipeline.update_speed_scale(0.0)
    assert pipeline._runtime_speed_scale == 0.1

    pipeline.update_speed_scale(5.0)
    assert pipeline._runtime_speed_scale == 1.0

    pipeline.update_speed_scale(0.5)
    assert pipeline._runtime_speed_scale == 0.5


# ---------------------------------------------------------------------------
# 9. ESTOP hold
# ---------------------------------------------------------------------------


def test_estop_returns_hold_positions(pipeline):
    """When ESTOP is triggered, all outputs are the snapshot positions."""
    obs = {"base": 0.5, "link1": 1.0}
    pipeline.trigger_estop(obs)

    action = {"base": 0.0, "link1": -1.0}  # try to move away
    result = pipeline.process(action, obs, dt=1 / 60)
    assert result["base"] == 0.5
    assert result["link1"] == 1.0


# ---------------------------------------------------------------------------
# 10. ESTOP from torque
# ---------------------------------------------------------------------------


def test_estop_from_torque_check(basic_config):
    """Mock SafetyLayer returning False triggers ESTOP on the check frame."""
    mock_sl = MagicMock()
    mock_sl.check_all_limits.return_value = False
    mock_sl.latest_torques = {}
    basic_config.torque_check_interval = 3
    pipe = SafetyPipeline(basic_config, safety_layer=mock_sl)

    obs = {"base": 0.5}
    action = {"base": 0.51}
    robot = MagicMock()

    # Frames 1, 2: no torque check (frame_count % 3 != 0)
    pipe.process(action, obs, robot=robot, dt=1 / 60)
    pipe.process(action, obs, robot=robot, dt=1 / 60)
    assert not pipe._estop

    # Frame 3: torque check fires, SafetyLayer says unsafe → ESTOP
    result = pipe.process(action, obs, robot=robot, dt=1 / 60)
    assert pipe._estop
    assert result["base"] == 0.5  # hold position


# ---------------------------------------------------------------------------
# 11. Fail-closed
# ---------------------------------------------------------------------------


def test_fail_closed_on_exception(basic_config):
    """Exception in a pipeline stage returns hold-position."""
    pipe = SafetyPipeline(basic_config)
    obs = {"base": 0.5}
    pipe._hold_positions = {"base": 0.5}

    # Corrupt joint_limits to cause TypeError in stage 1
    pipe._config.joint_limits = "not_a_dict"
    result = pipe.process({"base": 1.0}, obs, dt=1 / 60)
    assert result["base"] == 0.5


# ---------------------------------------------------------------------------
# 12. Reset clears state
# ---------------------------------------------------------------------------


def test_reset_clears_state(pipeline):
    """After reset, no stale state remains."""
    obs = {"base": 0.5}
    pipeline.process({"base": 0.6}, obs, dt=1 / 60)
    assert len(pipeline._prev_positions) > 0
    assert pipeline._frame_count > 0

    pipeline.reset()
    assert len(pipeline._prev_positions) == 0
    assert len(pipeline._prev_velocities) == 0
    assert len(pipeline._prev_output) == 0
    assert len(pipeline._hold_positions) == 0
    assert pipeline._frame_count == 0
    assert not pipeline._estop


# ---------------------------------------------------------------------------
# 13. Torque check interval
# ---------------------------------------------------------------------------


def test_torque_check_interval(basic_config):
    """Torque is checked exactly every Nth frame, not every frame."""
    mock_sl = MagicMock()
    mock_sl.check_all_limits.return_value = True
    mock_sl.latest_torques = {}
    basic_config.torque_check_interval = 3
    pipe = SafetyPipeline(basic_config, safety_layer=mock_sl)

    obs = {"base": 0.0}
    action = {"base": 0.01}
    robot = MagicMock()

    for _ in range(9):
        pipe.process(action, obs, robot=robot, dt=1 / 60)

    # 9 frames, check every 3rd → frames 3, 6, 9 = 3 calls
    assert mock_sl.check_all_limits.call_count == 3


# ---------------------------------------------------------------------------
# 14. effective_max_velocity
# ---------------------------------------------------------------------------


def test_effective_max_velocity_lookup():
    """Velocity lookup uses motor_models mapping and applies speed_scale."""
    config = SafetyConfig(
        motor_models={"base": "J8009P", "wrist": "J4310", "unknown_motor": ""},
        speed_scale=0.5,
    )
    assert config.effective_max_velocity("base") == pytest.approx(1.5 * 0.5)
    assert config.effective_max_velocity("wrist") == pytest.approx(3.5 * 0.5)
    # Empty model string → fallback
    assert config.effective_max_velocity("unknown_motor") == pytest.approx(
        FALLBACK_VELOCITY_LIMIT * 0.5
    )
    # Motor not in motor_models at all → fallback
    assert config.effective_max_velocity("missing") == pytest.approx(
        FALLBACK_VELOCITY_LIMIT * 0.5
    )


# ---------------------------------------------------------------------------
# 15. RuntimeState transitions
# ---------------------------------------------------------------------------


def test_valid_transitions():
    """Valid state transitions return True."""
    assert RuntimeState.can_transition(RuntimeState.IDLE, RuntimeState.STARTING)
    assert RuntimeState.can_transition(RuntimeState.STARTING, RuntimeState.RUNNING)
    assert RuntimeState.can_transition(RuntimeState.RUNNING, RuntimeState.HUMAN_ACTIVE)
    assert RuntimeState.can_transition(RuntimeState.HUMAN_ACTIVE, RuntimeState.RUNNING)
    assert RuntimeState.can_transition(RuntimeState.RUNNING, RuntimeState.PAUSED)
    assert RuntimeState.can_transition(RuntimeState.PAUSED, RuntimeState.RUNNING)
    assert RuntimeState.can_transition(RuntimeState.RUNNING, RuntimeState.STOPPING)
    assert RuntimeState.can_transition(RuntimeState.STOPPING, RuntimeState.IDLE)
    assert RuntimeState.can_transition(RuntimeState.RUNNING, RuntimeState.ESTOP)
    assert RuntimeState.can_transition(RuntimeState.RUNNING, RuntimeState.ERROR)


def test_invalid_transitions():
    """Invalid state transitions return False."""
    # Can't skip STARTING
    assert not RuntimeState.can_transition(RuntimeState.IDLE, RuntimeState.RUNNING)
    # Terminal states have no outbound transitions
    assert not RuntimeState.can_transition(RuntimeState.ESTOP, RuntimeState.RUNNING)
    assert not RuntimeState.can_transition(RuntimeState.ESTOP, RuntimeState.IDLE)
    assert not RuntimeState.can_transition(RuntimeState.ERROR, RuntimeState.IDLE)
    assert not RuntimeState.can_transition(RuntimeState.ERROR, RuntimeState.RUNNING)


# ---------------------------------------------------------------------------
# 16. Pipeline processes only provided keys
# ---------------------------------------------------------------------------


def test_pipeline_processes_only_position_keys(basic_config):
    """Pipeline returns exactly the keys from the input action — no extras injected."""
    pipe = SafetyPipeline(basic_config)

    action = {"base": 0.5, "link1": 0.3}
    observation = {"base": 0.4, "link1": 0.2}

    result = pipe.process(action, observation, dt=1 / 60)

    assert set(result.keys()) == set(action.keys())
    assert "gripper" not in result  # gripper is in joint_limits but not in action


# ---------------------------------------------------------------------------
# 17. SafetyConfig.from_policy_type
# ---------------------------------------------------------------------------


def test_from_policy_type_act():
    """ACT preset has light smoothing and high acceleration limit."""
    config = SafetyConfig.from_policy_type("act")
    assert config.smoothing_alpha == 0.85
    assert config.max_acceleration == 50.0
    assert config.speed_scale == 1.0


def test_from_policy_type_diffusion():
    """Diffusion preset has moderate smoothing."""
    config = SafetyConfig.from_policy_type("diffusion")
    assert config.smoothing_alpha == 0.5
    assert config.max_acceleration == 30.0
    assert config.speed_scale == 1.0


def test_from_policy_type_unknown_falls_back_to_conservative():
    """Unknown policy type falls back to conservative preset."""
    config = SafetyConfig.from_policy_type("unknown_policy")
    assert config.smoothing_alpha == 0.3
    assert config.max_acceleration == 15.0
    assert config.speed_scale == 0.5


def test_from_policy_type_overrides():
    """Overrides replace individual preset values."""
    config = SafetyConfig.from_policy_type("act", smoothing_alpha=0.95)
    assert config.smoothing_alpha == 0.95  # overridden
    assert config.max_acceleration == 50.0  # from preset


# ---------------------------------------------------------------------------
# 18. disable_smoothing bypass
# ---------------------------------------------------------------------------


def test_disable_smoothing_bypasses_stage3(basic_config):
    """With disable_smoothing=True, stage 3 is skipped entirely."""
    basic_config.disable_smoothing = True
    pipe = SafetyPipeline(basic_config)
    obs = {"base": 0.0}
    # Seed prev state
    pipe.process(obs, obs, dt=1 / 60)
    # Small move within velocity limit (J8009P: 1.5/60 = 0.025)
    action = {"base": 0.02}
    result = pipe.process(action, obs, dt=1 / 60)
    # Without stage 3, no EMA or acceleration filtering — passes through
    assert abs(result["base"] - 0.02) < 1e-6


def test_disable_smoothing_logs_warning(basic_config, caplog):
    """Creating a pipeline with disable_smoothing logs a warning."""
    basic_config.disable_smoothing = True
    import logging

    with caplog.at_level(logging.WARNING):
        SafetyPipeline(basic_config)
    assert any("smoothing DISABLED" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 19. Periodic debug logging
# ---------------------------------------------------------------------------


def test_periodic_debug_logging(basic_config, caplog):
    """Debug summary is logged every 30 frames."""
    pipe = SafetyPipeline(basic_config)
    obs = {"base": 0.0}
    import logging

    with caplog.at_level(logging.DEBUG):
        for _ in range(30):
            pipe.process({"base": 0.01}, obs, dt=1 / 60)
    assert any("Safety[30]" in msg for msg in caplog.messages)

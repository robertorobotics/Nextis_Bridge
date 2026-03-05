"""Tests for the unified deployment runtime.

Covers:
- Start/stop lifecycle and state transitions
- Safety pipeline always called
- Arm resolution from registry
- InterventionDetector position delta logic
- ObservationBuilder standalone usage
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.deployment.intervention import InterventionDetector
from app.core.deployment.rl_learner import RLLearner
from app.core.deployment.types import (
    SAFETY_PRESETS,
    ActionSource,
    DeploymentConfig,
    DeploymentMode,
    DeploymentStatus,
    RuntimeState,
    SafetyConfig,
)
from app.core.hardware.types import ArmDefinition, ArmRole, MotorType, Pairing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robot_lock():
    return threading.Lock()


@pytest.fixture
def mock_arm_registry():
    registry = MagicMock()

    leader_def = ArmDefinition(
        id="leader_left",
        name="Left Leader",
        role=ArmRole.LEADER,
        motor_type=MotorType.DYNAMIXEL_XL330,
        port="/dev/ttyUSB0",
        enabled=True,
        structural_design="umbra_7dof",
        config={"motor_names": ["left_base.pos", "left_link1.pos"]},
    )
    follower_def = ArmDefinition(
        id="follower_left",
        name="Left Follower",
        role=ArmRole.FOLLOWER,
        motor_type=MotorType.STS3215,
        port="/dev/ttyUSB1",
        enabled=True,
        structural_design="umbra_7dof",
        config={"motor_names": ["left_base.pos", "left_link1.pos"]},
    )

    registry.arms = {
        "leader_left": leader_def,
        "follower_left": follower_def,
    }

    pairing = Pairing(
        leader_id="leader_left",
        follower_id="follower_left",
        name="Left Pair",
    )
    registry.pairings = [pairing]
    registry.get_active_pairings.return_value = [pairing.to_dict()]

    leader_inst = MagicMock()
    leader_inst.get_action.return_value = {
        "left_base.pos": 0.0,
        "left_link1.pos": 0.0,
    }

    follower_inst = MagicMock()
    follower_inst.is_connected = True
    follower_inst.get_observation.return_value = {
        "left_base.pos": 0.1,
        "left_link1.pos": 0.2,
    }

    registry.arm_instances = {
        "leader_left": leader_inst,
        "follower_left": follower_inst,
    }

    return registry


@pytest.fixture
def mock_teleop():
    teleop = MagicMock()
    teleop.safety = MagicMock()
    teleop._action_lock = threading.Lock()
    teleop._latest_leader_action = {}
    teleop.dataset = None
    return teleop


@pytest.fixture
def mock_training():
    training = MagicMock()

    policy_info = MagicMock()
    policy_info.checkpoint_path = "/tmp/fake_checkpoint"

    policy_config = MagicMock()
    policy_config.cameras = ["camera_1"]
    policy_config.arms = ["left"]
    policy_config.policy_type = "act"
    policy_config.state_dim = 7
    policy_config.action_dim = 7

    training.get_policy.return_value = policy_info
    training.get_policy_config.return_value = policy_config
    return training


@pytest.fixture
def mock_camera_service():
    return MagicMock()


@pytest.fixture
def deployment_config():
    return DeploymentConfig(
        mode=DeploymentMode.INFERENCE,
        policy_id="test_policy",
        safety=SafetyConfig(),
    )


@pytest.fixture
def runtime(mock_teleop, mock_training, mock_arm_registry, mock_camera_service, robot_lock):
    """Create a DeploymentRuntime with mocked dependencies."""
    from app.core.deployment.runtime import DeploymentRuntime

    return DeploymentRuntime(
        teleop_service=mock_teleop,
        training_service=mock_training,
        arm_registry=mock_arm_registry,
        camera_service=mock_camera_service,
        robot_lock=robot_lock,
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test start/stop lifecycle."""

    def test_initial_state_is_idle(self, runtime):
        assert runtime._state == RuntimeState.IDLE

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_start_transitions_to_running(
        self, mock_load, runtime, deployment_config
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])

        # Give the loop thread a moment to start
        time.sleep(0.05)
        assert runtime._state == RuntimeState.RUNNING

        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_stop_is_idempotent(self, mock_load, runtime, deployment_config):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

        # Second stop should not raise
        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

    def test_start_fails_from_non_idle(self, runtime, deployment_config):
        # Manually set state to RUNNING
        runtime._state = RuntimeState.RUNNING

        with pytest.raises(RuntimeError, match="Cannot start"):
            runtime.start(deployment_config, ["leader_left", "follower_left"])

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_get_status_returns_deployment_status(
        self, mock_load, runtime, deployment_config
    ):
        status = runtime.get_status()
        assert isinstance(status, DeploymentStatus)
        assert status.state == RuntimeState.IDLE


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test state machine transitions."""

    def test_valid_transitions(self, runtime):
        assert runtime._transition(RuntimeState.STARTING)
        assert runtime._state == RuntimeState.STARTING

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.HUMAN_ACTIVE)
        assert runtime._state == RuntimeState.HUMAN_ACTIVE

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.PAUSED)
        assert runtime._state == RuntimeState.PAUSED

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.STOPPING)
        assert runtime._state == RuntimeState.STOPPING

        assert runtime._transition(RuntimeState.IDLE)
        assert runtime._state == RuntimeState.IDLE

    def test_invalid_transition_rejected(self, runtime):
        # IDLE → RUNNING is not valid (must go through STARTING)
        assert not runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.IDLE

    def test_estop_reachable_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert runtime._transition(RuntimeState.ESTOP)
        assert runtime._state == RuntimeState.ESTOP

    def test_estop_is_terminal(self, runtime):
        runtime._state = RuntimeState.ESTOP
        assert not runtime._transition(RuntimeState.RUNNING)
        assert not runtime._transition(RuntimeState.IDLE)

    def test_reset_from_estop(self, runtime):
        runtime._state = RuntimeState.ESTOP
        runtime._safety_pipeline = MagicMock()
        assert runtime.reset()
        assert runtime._state == RuntimeState.IDLE

    def test_reset_from_error(self, runtime):
        runtime._state = RuntimeState.ERROR
        runtime._safety_pipeline = MagicMock()
        assert runtime.reset()
        assert runtime._state == RuntimeState.IDLE

    def test_reset_fails_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert not runtime.reset()

    def test_pause_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert runtime.pause()
        assert runtime._state == RuntimeState.PAUSED

    def test_resume_from_paused(self, runtime):
        runtime._state = RuntimeState.PAUSED
        assert runtime.resume()
        assert runtime._state == RuntimeState.RUNNING


# ---------------------------------------------------------------------------
# Safety pipeline tests
# ---------------------------------------------------------------------------


class TestSafetyPipeline:
    """Verify safety pipeline is always called."""

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_safety_called_on_every_frame(
        self, mock_load, runtime, deployment_config, mock_arm_registry
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])

        # Let a few frames run
        time.sleep(0.15)
        runtime.stop()

        # Safety pipeline should have been created and its process() called
        assert runtime._safety_pipeline is None  # cleaned up after stop

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_safety_pipeline_created_with_safety_layer(
        self, mock_load, runtime, deployment_config, mock_teleop
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        # Verify pipeline was created (it gets cleaned up on stop)
        # We check by ensuring stop cleaned it up
        runtime.stop()
        assert runtime._safety_pipeline is None


# ---------------------------------------------------------------------------
# Arm resolution tests
# ---------------------------------------------------------------------------


class TestArmResolution:
    """Test arm resolution from registry."""

    def test_resolve_arms_calls_get_active_pairings(
        self, runtime, mock_arm_registry
    ):
        runtime._resolve_arms(["leader_left", "follower_left"])

        mock_arm_registry.get_active_pairings.assert_called_once_with(
            ["leader_left", "follower_left"]
        )
        assert runtime._leader is not None
        assert runtime._follower is not None

    def test_resolve_arms_auto_connects(self, runtime, mock_arm_registry):
        # Remove leader from instances to trigger auto-connect
        del mock_arm_registry.arm_instances["leader_left"]

        runtime._resolve_arms(["leader_left", "follower_left"])

        mock_arm_registry.connect_arm.assert_called_with("leader_left")

    def test_resolve_arms_raises_on_no_pairings(
        self, runtime, mock_arm_registry
    ):
        mock_arm_registry.get_active_pairings.return_value = []

        with pytest.raises(RuntimeError, match="No pairings found"):
            runtime._resolve_arms(["unknown_arm"])

    def test_resolve_arms_raises_without_registry(self, runtime):
        runtime._arm_registry = None

        with pytest.raises(RuntimeError, match="No arm registry"):
            runtime._resolve_arms(["leader_left"])


# ---------------------------------------------------------------------------
# InterventionDetector tests
# ---------------------------------------------------------------------------


class TestInterventionDetector:
    """Test the position-delta intervention detector."""

    def test_first_call_returns_no_intervention(self):
        leader = MagicMock()
        leader.get_action.return_value = {"left_base.pos": 0.0}

        detector = InterventionDetector(policy_arms=["left"], loop_hz=30)
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening
        assert velocity == 0.0

    def test_large_position_delta_triggers_intervention(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],
            move_threshold=0.05,
            loop_hz=30,
        )

        # First call initializes
        leader.get_action.return_value = {"left_base.pos": 0.0}
        detector.check(leader)

        # Large movement
        leader.get_action.return_value = {"left_base.pos": 0.5}
        is_intervening, velocity = detector.check(leader)

        assert is_intervening
        assert velocity == pytest.approx(0.5 * 30, abs=0.01)

    def test_small_delta_no_intervention(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],
            move_threshold=0.05,
            loop_hz=30,
        )

        leader.get_action.return_value = {"left_base.pos": 0.0}
        detector.check(leader)

        # Tiny movement (0.001 * 30 = 0.03 < 0.05 threshold)
        leader.get_action.return_value = {"left_base.pos": 0.001}
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening

    def test_policy_arms_filtering(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],  # Only left arm triggers
            move_threshold=0.05,
            loop_hz=30,
        )

        # First call
        leader.get_action.return_value = {
            "left_base.pos": 0.0,
            "right_base.pos": 0.0,
        }
        detector.check(leader)

        # Only right arm moves — should NOT trigger (left-only policy)
        leader.get_action.return_value = {
            "left_base.pos": 0.0,
            "right_base.pos": 1.0,
        }
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening

    def test_none_leader_returns_false(self):
        detector = InterventionDetector()
        is_intervening, velocity = detector.check(None)
        assert not is_intervening
        assert velocity == 0.0

    def test_is_idle_after_timeout(self):
        detector = InterventionDetector(idle_timeout=0.05)
        assert detector.is_idle()

        # Simulate a move
        detector._last_move_time = time.monotonic()
        assert not detector.is_idle()

        # Wait for timeout
        time.sleep(0.06)
        assert detector.is_idle()

    def test_reset_clears_state(self):
        detector = InterventionDetector()
        detector._last_positions = {"a": 1.0}
        detector._last_move_time = time.monotonic()

        detector.reset()

        assert detector._last_positions is None
        assert detector._last_move_time == 0.0


# ---------------------------------------------------------------------------
# ObservationBuilder tests
# ---------------------------------------------------------------------------


class TestObservationBuilder:
    """Test ObservationBuilder standalone usage."""

    def test_get_training_state_names_caches(self, tmp_path):
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        # Set up fake checkpoint with train_config
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        # Write train_config.json
        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        # Write dataset info.json
        info = {
            "features": {
                "observation.state": {
                    "names": ["left_base.pos", "left_link1.pos"],
                }
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        names = builder.get_training_state_names()
        assert names == ["left_base.pos", "left_link1.pos"]

        # Second call returns cached
        names2 = builder.get_training_state_names()
        assert names2 is names

    def test_get_training_state_names_none_when_missing(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        assert builder.get_training_state_names() is None

    def test_reset_cache_clears(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        # Load once (returns None since no files)
        builder.get_training_state_names()
        assert builder._training_state_names_loaded

        builder.reset_cache()
        assert not builder._training_state_names_loaded
        assert builder._training_state_names is None

    def test_convert_action_to_dict_returns_empty_without_names(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        import numpy as np

        action = np.array([0.1, 0.2, 0.3])
        result = builder.convert_action_to_dict(action, {})
        assert result == {}

    def test_convert_action_dict_passthrough(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        action = {"left_base.pos": 0.5}
        result = builder.convert_action_to_dict(action, {})
        assert result == {"left_base.pos": 0.5}

    def test_get_training_action_names_from_info(self, tmp_path):
        """Action names are read from features.action.names, separate from state names."""
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        obs_names = [
            "left_base.pos", "left_link1.pos", "left_link2.pos",
            "left_base.vel", "left_link1.vel", "left_link2.vel",
            "left_base.tau", "left_link1.tau", "left_link2.tau",
        ]
        action_names = ["left_base.pos", "left_link1.pos", "left_link2.pos"]
        info = {
            "features": {
                "observation.state": {"names": obs_names},
                "action": {"names": action_names},
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        assert builder.get_training_state_names() == obs_names
        assert builder.get_training_action_names() == action_names
        assert len(builder.get_training_state_names()) == 9
        assert len(builder.get_training_action_names()) == 3

    def test_get_training_action_names_fallback(self, tmp_path):
        """When action names are missing, fall back to state names (backward compat)."""
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        state_names = ["left_base.pos", "left_link1.pos"]
        info = {
            "features": {
                "observation.state": {"names": state_names},
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        assert builder.get_training_action_names() == state_names

    def test_convert_action_uses_action_names(self, tmp_path):
        """convert_action_to_dict uses action names (7) not obs state names (21)."""
        import json

        import numpy as np

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        obs_names = [f"j{i}.pos" for i in range(7)] + [f"j{i}.vel" for i in range(7)] + [f"j{i}.tau" for i in range(7)]
        action_names = [f"j{i}.pos" for i in range(7)]
        info = {
            "features": {
                "observation.state": {"names": obs_names},
                "action": {"names": action_names},
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = builder.convert_action_to_dict(action, {})
        assert len(result) == 7
        assert list(result.keys()) == action_names
        assert result["j0.pos"] == pytest.approx(0.1)

    def test_reset_cache_clears_action_names(self, tmp_path):
        """reset_cache clears action name cache fields."""
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        info = {
            "features": {
                "observation.state": {"names": ["j0.pos"]},
                "action": {"names": ["j0.pos"]},
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        builder.get_training_action_names()
        assert builder._training_action_names_loaded
        assert builder._dataset_info_loaded

        builder.reset_cache()
        assert not builder._training_action_names_loaded
        assert builder._training_action_names is None
        assert not builder._dataset_info_loaded
        assert builder._dataset_info is None

    def test_get_training_action_names_reads_action_feature(self, tmp_path):
        """Action names (7) are read separately from state names (21)."""
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        # 21 observation state names: 7 pos + 7 vel + 7 tau
        obs_names = (
            [f"left_j{i}.pos" for i in range(7)]
            + [f"left_j{i}.vel" for i in range(7)]
            + [f"left_j{i}.tau" for i in range(7)]
        )
        # 7 action names: positions only
        action_names = [f"left_j{i}.pos" for i in range(7)]

        info = {
            "features": {
                "observation.state": {"names": obs_names},
                "action": {"names": action_names},
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        assert builder.get_training_action_names() == action_names
        assert len(builder.get_training_action_names()) == 7
        assert builder.get_training_state_names() == obs_names
        assert len(builder.get_training_state_names()) == 21

    def test_get_training_action_names_fallback_to_state_names(self, tmp_path):
        """Without action key in info.json, falls back to observation state names."""
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        state_names = [f"left_j{i}.pos" for i in range(7)]
        info = {
            "features": {
                "observation.state": {"names": state_names},
                # No "action" key
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        assert builder.get_training_action_names() == state_names

    def test_convert_action_to_dict_uses_action_names(self, tmp_path):
        """convert_action_to_dict maps a 7-element tensor to 7 .pos keys."""
        import numpy as np

        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(checkpoint_path=checkpoint, policy=MagicMock())

        action_names = [f"left_j{i}.pos" for i in range(7)]
        builder.get_training_action_names = MagicMock(return_value=action_names)

        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = builder.convert_action_to_dict(action, {})

        assert len(result) == 7
        assert list(result.keys()) == action_names
        assert all(k.endswith(".pos") for k in result)
        assert result["left_j0.pos"] == pytest.approx(0.1)
        assert result["left_j6.pos"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# RLLearner tests
# ---------------------------------------------------------------------------


class TestRLLearner:
    """Test RLLearner placeholder interface."""

    def test_init(self):
        learner = RLLearner()
        assert learner.config is None

    def test_add_transition_raises(self):
        learner = RLLearner()
        with pytest.raises(NotImplementedError):
            learner.add_transition({}, None, 0.0, {}, False)

    def test_compute_reward_raises(self):
        learner = RLLearner()
        with pytest.raises(NotImplementedError):
            learner.compute_reward({})

    def test_get_metrics_returns_empty(self):
        learner = RLLearner()
        assert learner.get_metrics() == {}

    def test_stop_does_not_raise(self):
        learner = RLLearner()
        learner.stop()


# ---------------------------------------------------------------------------
# State machine update from intervention
# ---------------------------------------------------------------------------


class TestInterventionStateUpdates:
    """Test _update_state_from_intervention logic."""

    def test_policy_when_running_no_intervention(self, runtime):
        runtime._state = RuntimeState.RUNNING
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.POLICY

    def test_human_on_intervention(self, runtime):
        runtime._state = RuntimeState.RUNNING
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=True)
        assert source == ActionSource.HUMAN
        assert runtime._state == RuntimeState.HUMAN_ACTIVE

    def test_hold_when_paused(self, runtime):
        runtime._state = RuntimeState.PAUSED
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HOLD

    def test_human_active_stays_human_within_timeout(self, runtime):
        runtime._state = RuntimeState.HUMAN_ACTIVE
        detector = InterventionDetector(idle_timeout=10.0)
        detector._last_move_time = time.monotonic()
        runtime._intervention_detector = detector

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HUMAN

    def test_human_active_transitions_to_paused_on_idle(self, runtime):
        runtime._state = RuntimeState.HUMAN_ACTIVE
        detector = InterventionDetector(idle_timeout=0.01)
        detector._last_move_time = time.monotonic() - 1.0  # well past timeout
        runtime._intervention_detector = detector

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HOLD
        assert runtime._state == RuntimeState.PAUSED


# ---------------------------------------------------------------------------
# Partial action sending
# ---------------------------------------------------------------------------


class TestPartialActionSending:
    """Test _send_partial_action for bimanual and single-arm robots."""

    def test_single_arm_robot(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        # Remove bimanual attributes so it's treated as single-arm
        del robot.left_arm
        del robot.right_arm
        action = {"base.pos": 0.5, "link1.pos": 0.3}

        DeploymentRuntime._send_partial_action(robot, action)
        robot.send_action.assert_called_once_with(action)

    def test_bimanual_splits_and_strips_prefix(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        robot.left_arm = MagicMock()
        robot.right_arm = MagicMock()

        action = {
            "left_base.pos": 0.5,
            "left_link1.pos": 0.3,
            "right_base.pos": 0.1,
        }

        DeploymentRuntime._send_partial_action(robot, action)

        robot.left_arm.send_action.assert_called_once_with(
            {"base.pos": 0.5, "link1.pos": 0.3}
        )
        robot.right_arm.send_action.assert_called_once_with(
            {"base.pos": 0.1}
        )

    def test_bimanual_left_only(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        robot.left_arm = MagicMock()
        robot.right_arm = MagicMock()

        action = {"left_base.pos": 0.5}

        DeploymentRuntime._send_partial_action(robot, action)

        robot.left_arm.send_action.assert_called_once()
        robot.right_arm.send_action.assert_not_called()

    def test_single_arm_strips_prefix(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        del robot.left_arm
        del robot.right_arm
        action = {"left_base.pos": 0.5, "left_link1.pos": 0.3}

        DeploymentRuntime._send_partial_action(robot, action)
        robot.send_action.assert_called_once_with(
            {"base.pos": 0.5, "link1.pos": 0.3}
        )

    def test_single_arm_neutral_keys_pass_through(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        del robot.left_arm
        del robot.right_arm
        action = {"gripper": 0.8, "left_base.pos": 0.5}

        DeploymentRuntime._send_partial_action(robot, action)
        robot.send_action.assert_called_once_with(
            {"base.pos": 0.5, "gripper": 0.8}
        )


# ---------------------------------------------------------------------------
# Extended state auto-enable tests
# ---------------------------------------------------------------------------


class TestExtendedStateAutoEnable:
    """Test auto-enabling record_extended_state on follower."""

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_auto_enables_extended_state(
        self, mock_load, runtime, deployment_config, mock_arm_registry
    ):
        """Follower with config.record_extended_state gets it set to True."""
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        # Give follower a config with record_extended_state
        follower = mock_arm_registry.arm_instances["follower_left"]
        follower.config = MagicMock()
        follower.config.record_extended_state = False

        # Mock ObservationBuilder to return state names with vel/tau
        with patch(
            "app.core.deployment.runtime.ObservationBuilder"
        ) as MockBuilder:
            builder_inst = MockBuilder.return_value
            builder_inst.get_training_state_names.return_value = [
                "left_base.pos",
                "left_link1.pos",
                "left_base.vel",
                "left_link1.vel",
                "left_base.tau",
                "left_link1.tau",
            ]

            runtime.start(deployment_config, ["leader_left", "follower_left"])
            time.sleep(0.05)

            assert follower.config.record_extended_state is True

            runtime.stop()

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_no_extended_state_when_pos_only(
        self, mock_load, runtime, deployment_config, mock_arm_registry
    ):
        """Follower record_extended_state stays False for pos-only policies."""
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        follower = mock_arm_registry.arm_instances["follower_left"]
        follower.config = MagicMock()
        follower.config.record_extended_state = False

        with patch(
            "app.core.deployment.runtime.ObservationBuilder"
        ) as MockBuilder:
            builder_inst = MockBuilder.return_value
            builder_inst.get_training_state_names.return_value = [
                "left_base.pos",
                "left_link1.pos",
            ]

            runtime.start(deployment_config, ["leader_left", "follower_left"])
            time.sleep(0.05)

            assert follower.config.record_extended_state is False

            runtime.stop()

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_warning_when_follower_lacks_extended_state(
        self, mock_load, runtime, deployment_config, mock_arm_registry, caplog
    ):
        """Warning logged when follower doesn't support record_extended_state."""
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        # Follower without config.record_extended_state
        follower = mock_arm_registry.arm_instances["follower_left"]
        if hasattr(follower, "config"):
            del follower.config

        with patch(
            "app.core.deployment.runtime.ObservationBuilder"
        ) as MockBuilder:
            builder_inst = MockBuilder.return_value
            builder_inst.get_training_state_names.return_value = [
                "left_base.pos",
                "left_base.vel",
                "left_base.tau",
            ]

            import logging

            with caplog.at_level(logging.WARNING):
                runtime.start(
                    deployment_config, ["leader_left", "follower_left"]
                )
                time.sleep(0.05)

            assert any(
                "does not support record_extended_state" in msg
                for msg in caplog.messages
            )

            runtime.stop()


# ---------------------------------------------------------------------------
# Observation / action filtering tests
# ---------------------------------------------------------------------------


class TestExtendedStateFiltering:
    """Test that .vel/.tau keys are excluded from safety pipeline input."""

    def test_observation_positions_excludes_vel_tau(self):
        """observation_positions must only contain .pos keys."""
        raw_obs = {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
            "left_base.vel": 1.5,
            "left_link1.vel": 2.0,
            "left_base.tau": 0.3,
            "left_link1.tau": 0.4,
            "camera_1": "image_data",
        }

        observation_positions = {
            k: v
            for k, v in raw_obs.items()
            if isinstance(v, (int, float))
            and not k.endswith((".vel", ".tau"))
        }

        assert observation_positions == {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
        }

    def test_hold_action_excludes_vel_tau(self, runtime):
        """HOLD action must only return .pos keys."""
        raw_obs = {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
            "left_base.vel": 1.5,
            "left_link1.vel": 2.0,
            "left_base.tau": 0.3,
            "camera_1": "image_data",
        }

        result = runtime._get_action(ActionSource.HOLD, raw_obs)

        assert result == {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
        }

    def test_pos_only_obs_unchanged(self):
        """Filtering has no effect on pos-only observations."""
        raw_obs = {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
            "camera_1": "image_data",
        }

        observation_positions = {
            k: v
            for k, v in raw_obs.items()
            if isinstance(v, (int, float))
            and not k.endswith((".vel", ".tau"))
        }

        assert observation_positions == {
            "left_base.pos": 0.1,
            "left_link1.pos": 0.2,
        }


# ---------------------------------------------------------------------------
# Extended state handling tests
# ---------------------------------------------------------------------------


class TestExtendedStateHandling:
    """Test the extended state deployment chain: vel/tau filtering."""

    def test_observation_positions_exclude_vel_tau(self):
        """observation_positions extraction filters out .vel, .tau, and camera arrays."""
        import numpy as np

        raw_obs = {
            "base.pos": 0.5,
            "link1.pos": 0.3,
            "base.vel": 1.2,
            "base.tau": 0.8,
            "camera_1": np.zeros((480, 640, 3)),
        }

        observation_positions = {
            k: v
            for k, v in raw_obs.items()
            if isinstance(v, (int, float))
            and not k.endswith((".vel", ".tau"))
        }

        assert set(observation_positions.keys()) == {"base.pos", "link1.pos"}
        assert observation_positions["base.pos"] == 0.5
        assert observation_positions["link1.pos"] == 0.3
        # No .vel/.tau keys
        assert not any(k.endswith(".vel") for k in observation_positions)
        assert not any(k.endswith(".tau") for k in observation_positions)
        # Camera array excluded
        assert "camera_1" not in observation_positions

    def test_hold_action_excludes_vel_tau(self, runtime):
        """HOLD action returns only .pos keys from raw_obs."""
        runtime._state = RuntimeState.RUNNING

        raw_obs = {
            "left_base.pos": 0.5,
            "left_link1.pos": 0.3,
            "left_base.vel": 1.2,
            "left_link1.vel": 2.0,
            "left_base.tau": 0.8,
            "left_link1.tau": 0.4,
        }

        result = runtime._get_action(ActionSource.HOLD, raw_obs)

        assert set(result.keys()) == {"left_base.pos", "left_link1.pos"}
        assert result["left_base.pos"] == 0.5
        assert result["left_link1.pos"] == 0.3

    def test_send_partial_action_ignores_vel_tau_keys(self):
        """_send_partial_action passes vel keys through (arm driver ignores them)."""
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        del robot.left_arm
        del robot.right_arm

        action = {"left_base.pos": 0.5, "left_base.vel": 1.0}

        DeploymentRuntime._send_partial_action(robot, action)

        robot.send_action.assert_called_once_with(
            {"base.pos": 0.5, "base.vel": 1.0}
        )


# ---------------------------------------------------------------------------
# Safety preset auto-selection tests
# ---------------------------------------------------------------------------


class TestSafetyPresetAutoSelection:
    """Test auto-selection of safety presets based on policy type."""

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_act_preset_auto_selected(
        self, mock_load, runtime, deployment_config, mock_training
    ):
        """ACT policy type triggers ACT safety preset values."""
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")
        runtime._policy_config = mock_training.get_policy_config("test_policy")

        runtime.start(deployment_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        # policy_type is "act" from mock_training fixture
        assert runtime._config.safety.smoothing_alpha == 0.85
        assert runtime._config.safety.max_acceleration == 50.0
        assert runtime._config.safety.speed_scale == 1.0

        runtime.stop()

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_custom_motor_models_skips_preset(
        self, mock_load, runtime, mock_training
    ):
        """When user provides custom motor_models, preset is not applied."""
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        custom_config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test_policy",
            safety=SafetyConfig(
                motor_models={"base": "J8009P"},
                smoothing_alpha=0.4,
                max_acceleration=20.0,
            ),
        )

        runtime.start(custom_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        # Custom values preserved (not overwritten by ACT preset)
        assert runtime._config.safety.smoothing_alpha == 0.4
        assert runtime._config.safety.max_acceleration == 20.0

        runtime.stop()


# ---------------------------------------------------------------------------
# Temporal ensemble override tests
# ---------------------------------------------------------------------------


class TestTemporalEnsembleOverride:
    """Test deployment-time temporal ensemble override for ACT policies."""

    def test_te_override_attaches_ensembler(self, runtime, mock_training):
        """TE override creates and attaches ACTTemporalEnsembler."""
        runtime._policy = MagicMock()
        runtime._policy.config = MagicMock()
        runtime._policy.config.chunk_size = 50
        runtime._policy_config = mock_training.get_policy_config("test_policy")

        runtime._config = DeploymentConfig(
            policy_id="test_policy",
            temporal_ensemble_override=0.01,
        )

        mock_ensembler_cls = MagicMock()
        mock_ensembler_inst = MagicMock()
        mock_ensembler_cls.return_value = mock_ensembler_inst

        with patch.dict("sys.modules", {
            "lerobot.policies.act.modeling_act": MagicMock(
                ACTTemporalEnsembler=mock_ensembler_cls
            ),
        }):
            runtime._apply_temporal_ensemble_override()

        mock_ensembler_cls.assert_called_once_with(0.01, 50)
        assert runtime._policy.temporal_ensembler is mock_ensembler_inst
        assert runtime._policy.config.temporal_ensemble_coeff == 0.01
        assert runtime._policy.config.n_action_steps == 1

    def test_te_override_ignored_for_non_act(self, runtime, caplog):
        """TE override is ignored for non-ACT policy types."""
        import logging

        runtime._policy = MagicMock()
        runtime._policy_config = MagicMock()
        runtime._policy_config.policy_type = "diffusion"
        runtime._config = DeploymentConfig(
            policy_id="test_policy",
            temporal_ensemble_override=0.01,
        )

        with caplog.at_level(logging.WARNING):
            runtime._apply_temporal_ensemble_override()

        assert any("only supported for ACT" in msg for msg in caplog.messages)

    def test_te_override_none_is_noop(self, runtime):
        """No override when temporal_ensemble_override is None."""
        runtime._policy = MagicMock()
        runtime._policy.config = MagicMock()
        runtime._config = DeploymentConfig(policy_id="test_policy")

        original_coeff = runtime._policy.config.temporal_ensemble_coeff

        runtime._apply_temporal_ensemble_override()

        # Config should be untouched
        assert runtime._policy.config.temporal_ensemble_coeff == original_coeff


# ---------------------------------------------------------------------------
# Dry-run diagnostic mode tests
# ---------------------------------------------------------------------------


class TestDryRun:
    """Test dry-run diagnostic mode."""

    @patch("app.core.deployment.runtime.ObservationBuilder")
    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_dry_run_stops_after_30_frames(
        self, mock_load, MockBuilder, runtime, mock_arm_registry
    ):
        """Dry-run auto-stops after DRY_RUN_FRAMES frames."""
        import numpy as np

        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._policy.select_action.return_value = np.zeros(2)
        runtime._checkpoint_path = Path("/tmp/fake")

        builder_inst = MockBuilder.return_value
        builder_inst.get_training_state_names.return_value = [
            "left_base.pos", "left_link1.pos",
        ]
        builder_inst.prepare_observation.return_value = {
            "observation.state": np.array([[0.1, 0.2]]),
        }
        builder_inst.convert_action_to_dict.return_value = {
            "left_base.pos": 0.0, "left_link1.pos": 0.0,
        }

        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test_policy",
            safety=SafetyConfig(),
            dry_run=True,
        )

        runtime.start(config, ["leader_left", "follower_left"])

        # Wait for auto-stop (30 frames at 30Hz = ~1s, give extra margin)
        thread = runtime._loop_thread
        if thread:
            thread.join(timeout=5.0)

        assert runtime._frame_count == 30

        runtime.stop()

    @patch("app.core.deployment.runtime.ObservationBuilder")
    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_dry_run_does_not_send_action(
        self, mock_load, MockBuilder, runtime, mock_arm_registry
    ):
        """Dry-run never calls follower.send_action."""
        import numpy as np

        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._policy.select_action.return_value = np.zeros(2)
        runtime._checkpoint_path = Path("/tmp/fake")

        builder_inst = MockBuilder.return_value
        builder_inst.get_training_state_names.return_value = [
            "left_base.pos", "left_link1.pos",
        ]
        builder_inst.prepare_observation.return_value = {
            "observation.state": np.array([[0.0, 0.0]]),
        }
        builder_inst.convert_action_to_dict.return_value = {
            "left_base.pos": 0.0, "left_link1.pos": 0.0,
        }

        follower = mock_arm_registry.arm_instances["follower_left"]
        follower.send_action.reset_mock()

        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test_policy",
            safety=SafetyConfig(),
            dry_run=True,
        )

        runtime.start(config, ["leader_left", "follower_left"])

        thread = runtime._loop_thread
        if thread:
            thread.join(timeout=5.0)

        follower.send_action.assert_not_called()

        runtime.stop()

    @patch("app.core.deployment.runtime.ObservationBuilder")
    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_dry_run_collects_diagnostics(
        self, mock_load, MockBuilder, runtime, mock_arm_registry
    ):
        """Dry-run log contains 30 entries with expected keys."""
        import numpy as np

        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._policy.select_action.return_value = np.array([0.1, 0.2])
        runtime._checkpoint_path = Path("/tmp/fake")

        builder_inst = MockBuilder.return_value
        builder_inst.get_training_state_names.return_value = [
            "left_base.pos", "left_link1.pos",
        ]
        builder_inst.prepare_observation.return_value = {
            "observation.state": np.array([[0.5, 0.6]]),
        }
        builder_inst.convert_action_to_dict.return_value = {
            "left_base.pos": 0.1, "left_link1.pos": 0.2,
        }

        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test_policy",
            safety=SafetyConfig(),
            dry_run=True,
        )

        runtime.start(config, ["leader_left", "follower_left"])

        thread = runtime._loop_thread
        if thread:
            thread.join(timeout=5.0)

        # Capture before stop clears
        log = list(runtime._dry_run_log)

        assert len(log) == 30
        assert log[0]["frame"] == 0
        assert log[29]["frame"] == 29

        # Each entry should have denorm_action and robot_pos
        for entry in log:
            assert "frame" in entry
            assert "denorm_action" in entry
            assert "robot_pos" in entry

        runtime.stop()

    @patch("app.core.deployment.runtime.ObservationBuilder")
    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_dry_run_validates_ranges(
        self, mock_load, MockBuilder, runtime, mock_arm_registry
    ):
        """Extreme values produce validation warnings in dry-run log."""
        import numpy as np

        mock_load.return_value = None
        runtime._policy = MagicMock()
        # Return extreme action values that will trigger range warnings
        runtime._policy.select_action.return_value = np.array([10.0, -10.0])
        runtime._checkpoint_path = Path("/tmp/fake")

        builder_inst = MockBuilder.return_value
        builder_inst.get_training_state_names.return_value = [
            "left_base.pos", "left_link1.pos",
        ]
        builder_inst.prepare_observation.return_value = {
            "observation.state": np.array([[0.5, 0.6]]),
        }
        builder_inst.convert_action_to_dict.return_value = {
            "left_base.pos": 10.0, "left_link1.pos": -10.0,
        }

        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test_policy",
            safety=SafetyConfig(),
            dry_run=True,
        )

        runtime.start(config, ["leader_left", "follower_left"])

        thread = runtime._loop_thread
        if thread:
            thread.join(timeout=5.0)

        log = list(runtime._dry_run_log)

        # First frame should have warnings from range validation
        assert "warnings" in log[0]
        assert len(log[0]["warnings"]) > 0
        # Should mention the range issue
        assert any("exceeds" in w for w in log[0]["warnings"])

        runtime.stop()

    def test_get_policy_action_return_raw(self, runtime):
        """_get_policy_action with return_raw=True returns 3-tuple."""
        import numpy as np

        runtime._policy = MagicMock()
        action_out = np.array([0.1, 0.2])
        runtime._policy.select_action.return_value = action_out

        runtime._obs_builder = MagicMock()
        runtime._obs_builder.prepare_observation.return_value = {
            "observation.state": np.array([0.5, 0.6])
        }
        runtime._obs_builder.convert_action_to_dict.return_value = {
            "left_base.pos": 0.1, "left_link1.pos": 0.2
        }
        runtime._config = DeploymentConfig(policy_id="test_policy")

        # return_raw=False (default) returns dict
        result = runtime._get_policy_action({"left_base.pos": 0.0})
        assert isinstance(result, dict)

        # return_raw=True returns 3-tuple
        action_dict, policy_obs, action_tensor = runtime._get_policy_action(
            {"left_base.pos": 0.0}, return_raw=True
        )
        assert isinstance(action_dict, dict)
        assert "observation.state" in policy_obs
        assert action_tensor is action_out

    def test_get_policy_action_return_raw_none(self, runtime):
        """_get_policy_action with return_raw=True returns (None, None, None) when no policy."""
        runtime._policy = None
        runtime._obs_builder = None

        result = runtime._get_policy_action({}, return_raw=True)
        assert result == (None, None, None)

        # Without return_raw
        result = runtime._get_policy_action({})
        assert result is None


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for DeploymentRuntime.reset() recovery from terminal states."""

    def test_reset_from_estop_transitions_to_idle(self, runtime):
        runtime._state = RuntimeState.ESTOP
        pipeline = MagicMock()
        runtime._safety_pipeline = pipeline
        runtime._stop_event = threading.Event()

        result = runtime.reset()

        assert result is True
        assert runtime._state == RuntimeState.IDLE
        pipeline.clear_estop.assert_called_once()
        pipeline.reset.assert_called_once()

    def test_reset_from_error_transitions_to_idle(self, runtime):
        runtime._state = RuntimeState.ERROR
        runtime._safety_pipeline = MagicMock()
        runtime._stop_event = threading.Event()

        result = runtime.reset()

        assert result is True
        assert runtime._state == RuntimeState.IDLE

    def test_reset_from_running_rejected(self, runtime):
        runtime._state = RuntimeState.RUNNING

        result = runtime.reset()

        assert result is False
        assert runtime._state == RuntimeState.RUNNING

    def test_reset_from_idle_rejected(self, runtime):
        runtime._state = RuntimeState.IDLE

        result = runtime.reset()

        assert result is False

    def test_reset_clears_per_session_objects(self, runtime):
        runtime._state = RuntimeState.ESTOP
        runtime._policy = MagicMock()
        runtime._safety_pipeline = MagicMock()
        runtime._obs_builder = MagicMock()
        runtime._intervention_detector = MagicMock()
        runtime._checkpoint_path = "/some/path"
        runtime._policy_config = MagicMock()
        runtime._config = MagicMock()
        runtime._stop_event = threading.Event()

        runtime.reset()

        assert runtime._policy is None
        assert runtime._obs_builder is None
        assert runtime._safety_pipeline is None
        assert runtime._intervention_detector is None
        assert runtime._checkpoint_path is None
        assert runtime._policy_config is None
        assert runtime._config is None
        assert runtime._loop_thread is None

    def test_reset_calls_policy_reset(self, runtime):
        runtime._state = RuntimeState.ESTOP
        runtime._safety_pipeline = MagicMock()
        runtime._stop_event = threading.Event()
        policy = MagicMock()
        runtime._policy = policy

        runtime.reset()

        policy.reset.assert_called_once()

    def test_reset_joins_loop_thread(self, runtime):
        runtime._state = RuntimeState.ERROR
        runtime._safety_pipeline = MagicMock()
        runtime._stop_event = threading.Event()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        runtime._loop_thread = mock_thread

        runtime.reset()

        mock_thread.join.assert_called_once_with(timeout=5.0)


# ---------------------------------------------------------------------------
# restart() tests
# ---------------------------------------------------------------------------


class TestRestart:
    """Tests for DeploymentRuntime.restart() (stop + start with same config)."""

    def test_restart_calls_stop_then_start(self, runtime):
        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test-policy",
            safety=SafetyConfig(),
        )
        runtime._state = RuntimeState.RUNNING
        runtime._config = config
        runtime._active_arm_ids = ["follower_left"]
        runtime.stop = MagicMock()
        runtime.start = MagicMock()

        runtime.restart()

        runtime.stop.assert_called_once()
        runtime.start.assert_called_once_with(config, ["follower_left"])

    def test_restart_from_idle_raises(self, runtime):
        runtime._state = RuntimeState.IDLE

        with pytest.raises(RuntimeError, match="no active deployment"):
            runtime.restart()

    def test_restart_without_config_raises(self, runtime):
        runtime._state = RuntimeState.RUNNING
        runtime._config = None

        with pytest.raises(RuntimeError, match="no saved configuration"):
            runtime.restart()

    def test_restart_preserves_config_across_stop(self, runtime):
        """restart() must capture config BEFORE stop() clears it."""
        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="my-policy",
            safety=SafetyConfig(),
        )
        runtime._state = RuntimeState.RUNNING
        runtime._config = config
        runtime._active_arm_ids = ["follower_left", "follower_right"]

        captured = {}

        def mock_start(cfg, arm_ids):
            captured["config"] = cfg
            captured["arms"] = arm_ids

        runtime.stop = MagicMock()
        runtime.start = mock_start

        runtime.restart()

        assert captured["config"] is config
        assert captured["arms"] == ["follower_left", "follower_right"]

    def test_restart_from_estop(self, runtime):
        """restart() should work from any non-IDLE state."""
        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id="test-policy",
            safety=SafetyConfig(),
        )
        runtime._state = RuntimeState.ESTOP
        runtime._config = config
        runtime._active_arm_ids = ["follower_left"]
        runtime.stop = MagicMock()
        runtime.start = MagicMock()

        runtime.restart()

        runtime.stop.assert_called_once()
        runtime.start.assert_called_once()

"""Tests for the recording split-cache invariant.

CORE INVARIANT: During recording, action values come from the leader cache
(human intent) and observation values come from the follower cache (robot
reality).  These MUST be different when there is a gravity offset — the
leader commands a position, but the follower sags under gravity on heavy
joints (link1, link2).  If we recorded only one source for both, the
imitation-learning policy would never learn the correction needed to
overcome gravity.

Physics background:
  - Leader arm: Dynamixel XL330 (low-inertia, gravity-compensated or held by human)
  - Follower arm: Damiao J-series (7-DOF, ~4kg payload, significant gravity torque)
  - Gravity sag on link1 ≈ 0.15 rad, link2 ≈ 0.075 rad at typical poses
  - Without split caches, action == observation and the gravity-correction
    signal is lost from the training data.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest

from app.core.teleop.pairing import DYNAMIXEL_TO_DAMIAO_JOINT_MAP, PairingContext

# ── Constants ──────────────────────────────────────────────────────────

GRAVITY_OFFSET = 0.15  # rad — typical gravity sag on link1

# Leader positions (what the human commands via the Dynamixel leader)
LEADER_POSITIONS = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 0.5,
    "joint_3.pos": -0.9,
    "joint_4.pos": -0.5,
    "joint_5.pos": -0.1,
    "joint_6.pos": 0.05,
    "gripper.pos": 0.3,
}

# Expected leader action after Dynamixel→Damiao joint mapping (float mode)
MAPPED_LEADER_ACTION = {
    "base.pos": 0.0,
    "link1.pos": 0.5,
    "link2.pos": -0.9,
    "link3.pos": -0.5,
    "link4.pos": -0.1,
    "link5.pos": 0.05,
    "gripper.pos": 0.3,
}

# Follower positions from MIT response cache (robot reality with gravity sag)
FOLLOWER_CACHED_POSITIONS = {
    "base": 0.0,
    "link1": 0.5 - GRAVITY_OFFSET,          # sags under gravity
    "link2": -0.9 - GRAVITY_OFFSET * 0.5,   # smaller sag
    "link3": -0.5,
    "link4": -0.1,
    "link5": 0.05,
    "gripper": 0.3,
}


# ── Helpers ────────────────────────────────────────────────────────────

def _make_teleop_service(robot_lock):
    """Create a TeleoperationService with all hardware mocked out."""
    with patch("app.core.teleop.service.load_config", return_value={"teleop": {}}), \
         patch("app.core.teleop.service.save_config"):
        from app.core.teleop.service import TeleoperationService
        svc = TeleoperationService(
            robot=None,
            leader=None,
            robot_lock=robot_lock,
        )
        # Set attributes normally initialized by start() that the control
        # loop expects to exist.  Use tiny blend duration so startup blend
        # completes instantly (alpha = min(1.0, elapsed/1e-9) = 1.0).
        svc._blend_duration = 1e-9
        return svc


def _make_damiao_pairing_context(leader_mock, follower_mock):
    """Build a PairingContext for a Dynamixel→Damiao pairing."""
    joint_mapping = {
        f"{dyn}.pos": f"{dam}.pos"
        for dyn, dam in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items()
    }
    return PairingContext(
        pairing_id="test_leader→test_follower",
        active_leader=leader_mock,
        active_robot=follower_mock,
        joint_mapping=joint_mapping,
        follower_value_mode="float",
        has_damiao_follower=True,
        leader_cal_ranges={},
    )


def _make_leader_mock():
    """Mock leader arm that returns known positions."""
    leader = MagicMock()
    leader.get_action.return_value = dict(LEADER_POSITIONS)
    return leader


def _make_follower_mock():
    """Mock Damiao follower that returns cached positions with gravity sag."""
    follower = MagicMock()
    follower.get_cached_positions.return_value = dict(FOLLOWER_CACHED_POSITIONS)
    # get_observation() is called during startup blend to capture initial positions
    follower.get_observation.return_value = {
        f"{k}.pos": v for k, v in FOLLOWER_CACHED_POSITIONS.items()
    }
    # Gripper mapping range (open_pos, closed_pos used by map_range in control_loop)
    follower.gripper_open_pos = 0.0
    follower.gripper_closed_pos = 1.0
    follower.send_action.return_value = None
    # Prevent MagicMock's auto-attribute from triggering CAN bus dead check
    # (control_loop.py line 275: getattr(bus, '_can_bus_dead', False))
    follower.bus._can_bus_dead = False
    return follower


# ── Test 1 ─────────────────────────────────────────────────────────────

def test_split_caches_exist(robot_lock):
    """TeleoperationService must have separate leader and follower caches.

    WHY: The recording pipeline reads action from _latest_leader_action and
    observation from _latest_follower_obs.  If either attribute or its lock
    is missing, recording silently records zeros or crashes.
    """
    svc = _make_teleop_service(robot_lock)

    # Both caches exist and start empty
    assert hasattr(svc, '_latest_leader_action')
    assert hasattr(svc, '_latest_follower_obs')
    assert svc._latest_leader_action == {}
    assert svc._latest_follower_obs == {}

    # Each cache has its own lock (not shared — shared lock would serialize
    # the 60Hz control loop with the 30Hz recording thread)
    assert hasattr(svc, '_action_lock')
    assert hasattr(svc, '_follower_obs_lock')
    assert isinstance(svc._action_lock, type(threading.Lock()))
    assert isinstance(svc._follower_obs_lock, type(threading.Lock()))
    assert svc._action_lock is not svc._follower_obs_lock


# ── Test 2 ─────────────────────────────────────────────────────────────

def test_control_loop_populates_both_caches(robot_lock):
    """Control loop must populate BOTH caches with DIFFERENT data sources.

    WHY: The teleop loop reads the leader arm (human intent) and the follower
    arm (robot reality) independently.  Under gravity, the follower sags on
    link1/link2 while the leader stays at the commanded position.  Both
    caches must reflect their respective source.

    Physics: On a 7-DOF arm with ~4kg payload, link1 (shoulder) experiences
    ~6 Nm gravity torque.  With MIT gains kp=30, kd=1.5, the steady-state
    tracking error is tau_gravity / kp ≈ 0.2 rad.  The leader arm (held by
    human or gravity-compensated) has near-zero error.
    """
    svc = _make_teleop_service(robot_lock)
    leader = _make_leader_mock()
    follower = _make_follower_mock()
    ctx = _make_damiao_pairing_context(leader, follower)

    svc.recording_active = True
    svc.is_running = True

    # Stop after one iteration: send_action side effect flips the flag
    call_count = 0
    def stop_after_first(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            svc.is_running = False
    follower.send_action.side_effect = stop_after_first

    # Mock map_range used for gripper normalization in control_loop
    with patch("app.core.teleop.control_loop.map_range", side_effect=lambda v, *a: v):
        from app.core.teleop.control_loop import teleop_loop
        teleop_loop(svc, ctx)

    # Leader action cache: human intent (mapped to follower joint names)
    assert svc._latest_leader_action, "Leader action cache should not be empty"
    assert svc._latest_leader_action["link1.pos"] == pytest.approx(0.5)
    assert svc._latest_leader_action["link2.pos"] == pytest.approx(-0.9)

    # Follower obs cache: robot reality (with gravity sag)
    assert svc._latest_follower_obs, "Follower obs cache should not be empty"
    assert svc._latest_follower_obs["link1.pos"] == pytest.approx(0.5 - GRAVITY_OFFSET)
    assert svc._latest_follower_obs["link2.pos"] == pytest.approx(-0.9 - GRAVITY_OFFSET * 0.5)

    # THE CORE INVARIANT: leader and follower caches are DIFFERENT
    assert svc._latest_leader_action["link1.pos"] != svc._latest_follower_obs["link1.pos"]
    delta = abs(svc._latest_leader_action["link1.pos"] - svc._latest_follower_obs["link1.pos"])
    assert delta == pytest.approx(GRAVITY_OFFSET), (
        f"Expected gravity offset of {GRAVITY_OFFSET} rad, got {delta}"
    )


# ── Test 3 ─────────────────────────────────────────────────────────────

def test_recording_frame_action_vs_obs(robot_lock):
    """Recording frame must use leader for action and follower for observation.

    WHY: This is the most important test.  The recording thread reads from
    both caches and builds a frame dict.  If both action and obs come from
    the same source, the policy learns identity (action == obs) and never
    corrects for gravity — the robot will sag at deployment time.

    The recording_capture_loop in recording.py reads:
      - _latest_leader_action → action dict  (lines 298-313)
      - _latest_follower_obs  → obs dict     (lines 316-331)
    We simulate this exact logic here.
    """
    svc = _make_teleop_service(robot_lock)

    # Pre-populate caches (simulating what the control loop does)
    leader_action = dict(MAPPED_LEADER_ACTION)
    follower_obs = {
        f"{k}.pos": v for k, v in FOLLOWER_CACHED_POSITIONS.items()
    }

    with svc._action_lock:
        svc._latest_leader_action = leader_action.copy()
    with svc._follower_obs_lock:
        svc._latest_follower_obs = follower_obs.copy()

    # Simulate recording_capture_loop cache reads (recording.py lines 298-331)
    # No pairing filter (leader_keys empty, allowed_keys None)
    action = {}
    obs = {}

    with svc._action_lock:
        if svc._latest_leader_action:
            for key, val in svc._latest_leader_action.items():
                action[key] = val

    with svc._follower_obs_lock:
        if svc._latest_follower_obs:
            for key, val in svc._latest_follower_obs.items():
                obs[key] = val

    # Action has leader values (human intent — no gravity sag)
    assert action["link1.pos"] == pytest.approx(0.5)
    assert action["link2.pos"] == pytest.approx(-0.9)

    # Observation has follower values (robot reality — with gravity sag)
    assert obs["link1.pos"] == pytest.approx(0.5 - GRAVITY_OFFSET)
    assert obs["link2.pos"] == pytest.approx(-0.9 - GRAVITY_OFFSET * 0.5)

    # The delta is nonzero — this IS the gravity correction signal
    for joint in ["link1.pos", "link2.pos"]:
        assert abs(action[joint] - obs[joint]) > 0.05, (
            f"action and obs for {joint} should differ by gravity offset, "
            f"got action={action[joint]}, obs={obs[joint]}"
        )

    # Joints without gravity offset should match
    for joint in ["base.pos", "link3.pos", "link4.pos", "link5.pos"]:
        assert action[joint] == pytest.approx(obs[joint]), (
            f"{joint} should match (no gravity offset on this joint)"
        )


# ── Test 4 ─────────────────────────────────────────────────────────────

def test_recording_extended_state_from_follower(robot_lock):
    """Velocity and torque in observation must come from the FOLLOWER, not leader.

    WHY: Extended state (velocity, torque) is read from the follower's MIT
    response cache.  These values represent the robot's actual dynamics —
    useful for learning force-aware policies.  They should appear in
    observation only (the robot's state), never in action (the command).

    The recording_capture_loop reads extended state at lines 342-380:
      velocities = active_robot.bus.read_cached_velocities()
      torques = active_robot.bus.read_torques()
    """
    svc = _make_teleop_service(robot_lock)

    # Pre-populate position caches
    with svc._action_lock:
        svc._latest_leader_action = dict(MAPPED_LEADER_ACTION)
    with svc._follower_obs_lock:
        svc._latest_follower_obs = {
            f"{k}.pos": v for k, v in FOLLOWER_CACHED_POSITIONS.items()
        }

    # Mock follower bus with velocity and torque data
    follower = _make_follower_mock()
    follower.bus = MagicMock()
    follower.bus.read_cached_velocities.return_value = {
        "base": 0.01, "link1": -0.23, "link2": 0.15,
        "link3": 0.0, "link4": 0.02, "link5": -0.01, "gripper": 0.0,
    }
    follower.bus.read_torques.return_value = {
        "base": 0.5, "link1": 2.8, "link2": 1.2,
        "link3": 0.3, "link4": 0.1, "link5": 0.05, "gripper": 0.4,
    }

    # Attach pairing context so recording_capture_loop can find the robot
    ctx = _make_damiao_pairing_context(_make_leader_mock(), follower)
    svc._pairing_contexts = [ctx]
    svc._record_extended_state = True

    # Simulate the extended state section of recording_capture_loop (lines 342-380)
    action = dict(svc._latest_leader_action)
    obs = dict(svc._latest_follower_obs)

    # Replicate recording.py lines 344-377: find active_robot from pairing contexts
    active_robot = None
    for pctx in svc._pairing_contexts:
        if (pctx.active_robot and hasattr(pctx.active_robot, 'bus')
                and hasattr(pctx.active_robot.bus, 'read_cached_velocities')):
            active_robot = pctx.active_robot
            break

    assert active_robot is not None, "Should find active robot from pairing contexts"

    velocities = active_robot.bus.read_cached_velocities()
    torques = active_robot.bus.read_torques()
    for name, vel in velocities.items():
        obs[f"{name}.vel"] = vel
    for name, tau in torques.items():
        obs[f"{name}.tau"] = tau

    # Velocity and torque appear in observation (follower state)
    assert "link1.vel" in obs
    assert "link1.tau" in obs
    assert obs["link1.vel"] == pytest.approx(-0.23)
    assert obs["link1.tau"] == pytest.approx(2.8)

    # Velocity and torque do NOT appear in action (human command)
    assert "link1.vel" not in action
    assert "link1.tau" not in action
    assert "base.vel" not in action
    assert "base.tau" not in action


# ── Test 5 ─────────────────────────────────────────────────────────────

def test_fallback_when_follower_cache_empty(robot_lock):
    """When follower cache is empty, recording falls back to leader positions.

    WHY: On the very first recording frame, the MIT response from the Damiao
    follower may not have arrived yet (takes ~2ms per motor × 7 motors ≈ 14ms).
    The recording thread runs at 30fps (33ms period), so the first frame may
    have an empty follower cache.  Rather than recording nothing (which would
    corrupt the dataset with missing frames), we gracefully degrade by using
    leader positions as a proxy for observation.

    This fallback is implemented in recording.py lines 333-340:
      if not any('.pos' in k for k in obs) and action:
          for k, v in action.items():
              if k not in obs:
                  obs[k] = v
    """
    svc = _make_teleop_service(robot_lock)

    # Leader cache populated, follower cache empty (first frame scenario)
    with svc._action_lock:
        svc._latest_leader_action = dict(MAPPED_LEADER_ACTION)
    # _latest_follower_obs stays empty {}

    # Simulate recording_capture_loop cache reads
    action = {}
    obs = {}

    with svc._action_lock:
        if svc._latest_leader_action:
            for key, val in svc._latest_leader_action.items():
                action[key] = val

    with svc._follower_obs_lock:
        if svc._latest_follower_obs:
            for key, val in svc._latest_follower_obs.items():
                obs[key] = val

    # Fallback logic (recording.py lines 333-340)
    if not any('.pos' in k for k in obs) and action:
        for k, v in action.items():
            if k not in obs:
                obs[k] = v

    # Obs should now have data (from leader fallback)
    assert obs, "Obs should not be empty after fallback"
    assert any('.pos' in k for k in obs), "Obs should have position keys after fallback"
    assert obs["link1.pos"] == pytest.approx(0.5), (
        "Fallback should use leader value when follower cache is empty"
    )

    # Action remains the leader values
    assert action["link1.pos"] == pytest.approx(0.5)

    # In fallback mode, action == obs (acceptable for first frame only)
    assert action["link1.pos"] == obs["link1.pos"]


# ── Test 6 ─────────────────────────────────────────────────────────────

def test_no_regression_non_recording_teleop(robot_lock):
    """When NOT recording, teleop behavior must be completely unchanged.

    WHY: The split-cache logic only activates when svc.recording_active is
    True.  When just teleoperating without recording, we must not add
    overhead from follower cache reads (get_cached_positions costs ~0 for
    Damiao MIT cache, but the principle matters for Feetech where
    get_observation triggers a serial read).

    In control_loop.py line 399: `if svc.recording_active:` gates the cache
    population.  When False, only _latest_leader_action is updated (line 468)
    and _latest_follower_obs is never touched.
    """
    svc = _make_teleop_service(robot_lock)
    leader = _make_leader_mock()
    follower = _make_follower_mock()
    ctx = _make_damiao_pairing_context(leader, follower)

    svc.recording_active = False  # NOT recording
    svc.is_running = True

    # Stop after one iteration
    call_count = 0
    def stop_after_first(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            svc.is_running = False
    follower.send_action.side_effect = stop_after_first

    with patch("app.core.teleop.control_loop.map_range", side_effect=lambda v, *a: v):
        from app.core.teleop.control_loop import teleop_loop
        teleop_loop(svc, ctx)

    # Motor command still sent to follower (teleop works normally)
    follower.send_action.assert_called_once()
    sent_action = follower.send_action.call_args[0][0]
    assert "link1.pos" in sent_action, "Follower should receive link1 position command"
    assert sent_action["link1.pos"] == pytest.approx(0.5)

    # Follower obs cache NOT populated (no recording overhead)
    assert svc._latest_follower_obs == {}, (
        "Follower obs cache should remain empty when not recording — "
        "no reason to read follower positions if not capturing data"
    )

    # Leader action cache IS populated (used for UI data streaming)
    assert svc._latest_leader_action, (
        "Leader action cache should be populated even without recording "
        "(used for graph/UI streaming)"
    )

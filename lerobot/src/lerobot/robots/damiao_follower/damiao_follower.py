# Copyright 2024 Nextis. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Damiao 7-DOF follower arm for high-torque assembly tasks.

This robot uses Damiao J-series motors (J8009P, J4340P, J4310) connected
via CAN-to-serial bridge. It features a global velocity limiter for safety
when working with high-torque motors.

Safety Features:
- Global velocity_limit (0.0-1.0) applied to ALL motor commands
- Torque monitoring with configurable limits
- Gripper force limiting
- Safe disconnect with torque disable
"""

import logging
import time
import numpy as np
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig
from lerobot.motors.damiao.tables import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS

from .config_damiao_follower import DamiaoFollowerConfig

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another."""
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class DamiaoFollowerRobot(Robot):
    """Damiao 7-DOF follower arm for high-torque assembly tasks.

    This robot is designed for assembly tasks requiring high torque (up to 35Nm).
    It uses a global velocity limiter for safety.

    Motor Configuration:
    - Base/Link1: J8009P (35Nm) - high torque for base movements
    - Link2/Link3: J4340P (8Nm) - medium torque for elbow
    - Link4/Link5/Gripper: J4310 (4Nm) - precision for wrist

    Example:
        from lerobot.robots.damiao_follower import DamiaoFollowerRobot, DamiaoFollowerConfig

        config = DamiaoFollowerConfig(
            port="/dev/ttyUSB0",
            velocity_limit=0.2,  # Start at 20% for safety
        )
        robot = DamiaoFollowerRobot(config)
        robot.connect()

        # Set velocity limit (0.0-1.0)
        robot.velocity_limit = 0.5  # 50%

        # Get observation
        obs = robot.get_observation()

        # Send action
        robot.send_action({"base.pos": 0.5, "link1.pos": 0.3, ...})

        robot.disconnect()
    """

    config_class = DamiaoFollowerConfig
    name = "damiao_follower"
    _supports_cached_observation: bool = True

    def __init__(self, config: DamiaoFollowerConfig):
        super().__init__(config)
        self.config = config

        # Build motor bus config
        bus_config = DamiaoMotorsBusConfig(
            port=config.port,
            baudrate=config.baudrate,
            motors=config.motor_config,
            velocity_limit=config.velocity_limit,
            skip_pid_config=getattr(config, 'skip_pid_config', False),
            # MIT mode settings (recommended over POS_VEL for stability)
            use_mit_mode=getattr(config, 'use_mit_mode', True),
            mit_kp=getattr(config, 'mit_kp', 15.0),
            mit_kd=getattr(config, 'mit_kd', 1.5),
        )

        self.bus = DamiaoMotorsBus(bus_config)
        if not self.calibration:
            self.calibration = {}  # Only init if parent didn't load from HF cache
        self.motor_inversions = {}  # Loaded from calibration_profiles/{arm_id}/inversions.json

        # Gripper positions
        self.gripper_open_pos = config.gripper_open_pos
        self.gripper_closed_pos = config.gripper_closed_pos

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Motor names for feature dictionaries
        self._motor_names = list(config.motor_config.keys())

    @property
    def velocity_limit(self) -> float:
        """Get current global velocity limit (0.0-1.0)."""
        return self.bus.velocity_limit

    @velocity_limit.setter
    def velocity_limit(self, value: float):
        """Set global velocity limit (0.0-1.0). Applied to ALL motor commands."""
        self.bus.velocity_limit = value
        logger.info(f"[DamiaoFollower] Velocity limit set to {self.bus.velocity_limit:.2f}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation/action."""
        features = {f"{motor}.pos": float for motor in self._motor_names}
        if self.config.record_extended_state:
            for motor in self._motor_names:
                features[f"{motor}.vel"] = float
            for motor in self._motor_names:
                features[f"{motor}.tau"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features available in observations."""
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        """Features accepted in actions (position-only)."""
        return {f"{motor}.pos": float for motor in self._motor_names}

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """Damiao motors use absolute encoders, always calibrated."""
        return True

    def calibrate(self) -> None:
        """Calibration (not needed for Damiao - absolute encoders)."""
        pass

    def apply_calibration_limits(self) -> None:
        """Push calibrated joint limits to the motor bus for runtime enforcement.

        Called after loading a calibration profile or completing range discovery.
        Overrides hardcoded defaults in tables.py with user-calibrated ranges.
        """
        limits = {}
        for name, cal in self.calibration.items():
            if hasattr(cal, 'range_min') and hasattr(cal, 'range_max'):
                limits[name] = (cal.range_min, cal.range_max)
        if limits:
            self.bus.update_joint_limits(limits)
            logger.info(f"[DamiaoFollower] Applied calibrated limits for {len(limits)} motors")

    def reload_inversions(self):
        """Load motor inversion settings from calibration_profiles/{arm_id}/inversions.json."""
        import json
        from pathlib import Path

        arm_id = self.id or "aira_zero"
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        profile_path = project_root / "calibration_profiles" / arm_id / "inversions.json"

        self.motor_inversions = {}
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    self.motor_inversions = json.load(f)
                logger.info(f"[DamiaoFollower] Loaded inversions for {arm_id}: {self.motor_inversions}")
            except Exception as e:
                logger.error(f"[DamiaoFollower] Failed to load inversions: {e}")
        else:
            logger.info(f"[DamiaoFollower] No inversions file at {profile_path}")

    def configure(self) -> None:
        """Configure motors (called during connect)."""
        if self.bus.is_connected:
            self.bus.configure()

    def connect(self) -> None:
        """Connect to robot, configure motors, and home gripper."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"[DamiaoFollower] Connecting to {self.config.port}")
        logger.info(f"[DamiaoFollower] Velocity limit: {self.config.velocity_limit:.2f} ({self.config.velocity_limit*100:.0f}%)")

        # Connect motor bus
        self.bus.connect()
        try:
            self.bus.configure()

            # Push calibrated joint limits to bus (overrides narrow defaults from tables.py)
            if self.calibration:
                self.apply_calibration_limits()

            # Home gripper (finds open position)
            if getattr(self.config, 'skip_gripper_homing', False):
                logger.info("[DamiaoFollower] Skipping gripper homing (config)")
            else:
                self.bus.home_gripper()

            # Connect cameras
            for cam in self.cameras.values():
                cam.connect()

        except Exception:
            # SAFETY: On any failure, shut down CAN bus properly to prevent
            # orphaned recv threads and motor state leaks on retry
            logger.error("[DamiaoFollower] Connection failed — disconnecting bus for safe cleanup")
            try:
                self.bus.disconnect(disable_torque=True)
            except Exception as cleanup_err:
                logger.warning(f"[DamiaoFollower] Cleanup disconnect failed: {cleanup_err}")
            raise

        logger.info(f"[DamiaoFollower] Connected successfully")

    def get_observation(self) -> dict[str, Any]:
        """Get current robot state and camera images.

        Returns:
            Dict with:
            - "{motor}.pos": Motor position in radians (gripper normalized 0-1)
            - "{camera}": Camera image as numpy array
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = {}

        # Read motor positions
        positions = self.bus.sync_read("Present_Position")
        for name, pos in positions.items():
            if name == "gripper":
                # Normalize gripper: 0 = open, 1 = closed
                norm_pos = map_range(
                    pos, self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0
                )
                obs_dict[f"{name}.pos"] = norm_pos
            else:
                obs_dict[f"{name}.pos"] = pos

        # Apply inversions for consistent logical positions
        if self.motor_inversions:
            for motor, is_inverted in self.motor_inversions.items():
                key = f"{motor}.pos"
                if is_inverted and key in obs_dict and motor != "gripper":
                    obs_dict[key] = -obs_dict[key]

        # Extended state: velocity and torque from MIT response cache (zero CAN overhead)
        if self.config.record_extended_state:
            velocities = self.bus.read_cached_velocities()
            torques = self.bus.read_torques()
            for name in self._motor_names:
                obs_dict[f"{name}.vel"] = velocities.get(name, 0.0)
                obs_dict[f"{name}.tau"] = torques.get(name, 0.0)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"[DamiaoFollower] Read state: {dt_ms:.1f}ms")

        # Capture camera images
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"[DamiaoFollower] Read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def get_observation_cached(self) -> dict[str, Any]:
        """Get observation from MIT response cache (zero CAN overhead).

        Unlike get_observation() which does sync_read (sends zero-torque probes
        kp=0, kd=0 to each motor), this reads positions/velocities/torques from
        the cache populated by the previous sync_write. This means:
        - No CAN reads → no torque interruption → arm maintains holding torque
        - Observation is from the previous frame (one-frame latency, ~33ms)
        - Suitable for deployment control loops where continuous torque is critical

        Falls back to get_observation() if the cache is empty (first frame
        before any sync_write has been sent).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check if cache is populated (sync_write must have been called at least once)
        cached = self.bus.read_cached_positions()
        if not cached:
            return self.get_observation()

        obs_dict = {}

        # Positions from cache (same key format as get_observation)
        for name, pos in cached.items():
            if name == "gripper":
                norm_pos = map_range(
                    pos, self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0
                )
                obs_dict[f"{name}.pos"] = norm_pos
            else:
                obs_dict[f"{name}.pos"] = pos

        # Apply inversions (same as get_observation)
        if self.motor_inversions:
            for motor, is_inverted in self.motor_inversions.items():
                key = f"{motor}.pos"
                if is_inverted and key in obs_dict and motor != "gripper":
                    obs_dict[key] = -obs_dict[key]

        # Extended state from cache (zero CAN overhead)
        if self.config.record_extended_state:
            velocities = self.bus.read_cached_velocities()
            torques = self.bus.read_torques()
            for name in self._motor_names:
                obs_dict[f"{name}.vel"] = velocities.get(name, 0.0)
                obs_dict[f"{name}.tau"] = torques.get(name, 0.0)

        # Camera images (same as get_observation)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send goal positions to motors.

        IMPORTANT: All motor commands are velocity-limited by self.velocity_limit.

        Args:
            action: Dict with "{motor}.pos" keys and position values.
                   Gripper expects normalized value (0=open, 1=closed).

        Returns:
            Dict with actual goal positions sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                goal_pos[motor_name] = val

        # Convert gripper from normalized to radians FIRST (before joint limits).
        # Joint limits are in radians (-5.32, 0.0), so the gripper must be in radians
        # before clamping. Otherwise np.clip(0.8, -5.32, 0.0) = 0.0 — always open!
        if "gripper" in goal_pos:
            raw = goal_pos["gripper"]
            if raw > 2.0:  # Umbra leader sends 0-100 (RANGE_0_100)
                raw = raw / 100.0
            raw = np.clip(raw, 0.0, 1.0)  # Defensive clamp to prevent out-of-range torque
            goal_pos["gripper"] = map_range(
                raw, 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos
            )

        # Apply motor inversions (before joint limits clipping)
        if self.motor_inversions:
            for motor, is_inverted in self.motor_inversions.items():
                if is_inverted and motor in goal_pos and motor != "gripper":
                    goal_pos[motor] = -goal_pos[motor]

        # Apply joint limits (prefer calibrated ranges, fall back to config defaults)
        for motor_name in list(goal_pos.keys()):
            if motor_name in self.calibration and hasattr(self.calibration[motor_name], 'range_min'):
                cal = self.calibration[motor_name]
                lo, hi = cal.range_min, cal.range_max
            elif motor_name in self.config.joint_limits:
                lo, hi = self.config.joint_limits[motor_name]
            else:
                continue
            goal_pos[motor_name] = np.clip(goal_pos[motor_name], lo, hi)

        # Send joint positions (velocity limited by bus)
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)

        # Return what we sent
        result = {f"{m}.pos": v for m, v in goal_pos.items()}
        return result

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect cameras first
        for cam in self.cameras.values():
            cam.disconnect()

        # Disconnect motor bus (optionally disables torque)
        self.bus.disconnect(disable_torque=self.config.disable_torque_on_disconnect)

        logger.info(f"[DamiaoFollower] Disconnected")

    def get_torques(self) -> dict[str, float]:
        """Read current torques from all motors (for safety monitoring)."""
        return self.bus.read_torques()

    def get_cached_positions(self) -> dict[str, float]:
        """Get cached follower positions in logical space (zero CAN overhead).

        Uses MIT response cache (no CAN reads). Applies inversions for
        consistent coordinate space with leader.
        """
        positions = self.bus.read_cached_positions()
        if self.motor_inversions:
            for motor, is_inverted in self.motor_inversions.items():
                if is_inverted and motor in positions and motor != "gripper":
                    positions[motor] = -positions[motor]
        return positions

    def get_torque_limits(self) -> dict[str, float]:
        """Get torque limits for each motor (85% of max)."""
        return self.bus.get_torque_limits()

    def emergency_stop(self) -> None:
        """Emergency stop: disable all motor torques immediately."""
        logger.warning("[DamiaoFollower] EMERGENCY STOP - Disabling all torques")
        self.bus.disable_torque()

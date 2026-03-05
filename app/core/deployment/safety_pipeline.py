"""4-stage fail-closed safety pipeline for policy deployment.

Every action dict passes through process() before reaching the robot.

Stages:
    1. Joint Limit Clamp  — hard clamp to calibrated [min, max]
    2. Velocity Limiter    — proper delta clamping (not asymptotic scaling)
    3. Acceleration Filter — limit accel + EMA smoothing
    4. Torque Monitor      — periodic, delegates to SafetyLayer
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from .types import SafetyConfig

logger = logging.getLogger(__name__)


@dataclass
class SafetyReadings:
    """Telemetry from the safety pipeline for monitoring / frontend display."""
    per_motor_velocity: Dict[str, float] = field(default_factory=dict)
    per_motor_torque: Dict[str, float] = field(default_factory=dict)
    # stage_name -> number of motors clamped this frame
    active_clamps: Dict[str, int] = field(default_factory=dict)
    estop_active: bool = False
    speed_scale: float = 1.0


class SafetyPipeline:
    """4-stage fail-closed action filter for all deployment modes.

    Usage::

        pipeline = SafetyPipeline(config, safety_layer=existing_safety)
        filtered = pipeline.process(action, observation, robot=robot)
        robot.send_action(filtered)
    """

    def __init__(
        self,
        config: SafetyConfig,
        safety_layer=None,
    ):
        self._config = config
        self._safety_layer = safety_layer

        if config.disable_smoothing:
            logger.warning("Safety pipeline: smoothing DISABLED (stage 3 bypassed)")

        # Internal frame-to-frame state
        self._prev_positions: Dict[str, float] = {}
        self._prev_velocities: Dict[str, float] = {}
        self._prev_output: Dict[str, float] = {}  # for EMA smoothing
        self._hold_positions: Dict[str, float] = {}
        self._frame_count: int = 0
        self._estop: bool = False
        # Separate runtime speed_scale (for warmup ramp etc.) that doesn't
        # mutate the original SafetyConfig.
        self._runtime_speed_scale: Optional[float] = None
        self._readings = SafetyReadings(speed_scale=config.speed_scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        action: Dict[str, float],
        observation: Dict[str, float],
        robot=None,
        dt: float = 1.0 / 60,
    ) -> Dict[str, float]:
        """Filter an action through all safety stages.

        Args:
            action: Target positions keyed by motor name.
            observation: Current positions keyed by motor name.
            robot: Optional robot instance for torque monitoring.
            dt: Timestep in seconds (default 1/60 for 60 Hz loop).

        Returns:
            Filtered action dict.  If ESTOP is active or any stage
            raises an exception, returns hold-position (fail-closed).
        """
        if self._estop:
            self._readings.estop_active = True
            return self._get_hold_output(action, observation)

        try:
            self._frame_count += 1
            self._readings.active_clamps = {}

            # Stage 1 — joint limit clamp
            filtered = self._clamp_joint_limits(action)

            # Stage 2 — velocity limiter
            filtered = self._limit_velocity(filtered, observation, dt)

            # Stage 3 — acceleration filter + EMA smoothing
            if not self._config.disable_smoothing:
                filtered = self._filter_acceleration(filtered, dt)

            # Stage 4 — torque monitor (periodic)
            if robot is not None:
                if not self._check_torque(robot):
                    self.trigger_estop(observation)
                    return self._get_hold_output(action, observation)

            # Commit state for next frame
            self._update_prev_state(filtered, observation, dt)

            # Periodic debug logging (~1/s at 30Hz)
            if self._frame_count % 30 == 0:
                vel_clamps = self._readings.active_clamps.get("velocity", 0)
                accel_clamps = self._readings.active_clamps.get("acceleration", 0)
                logger.debug(
                    "Safety[%d]: vel_clamps=%d accel_clamps=%d smoothing=%.2f scale=%.1f",
                    self._frame_count,
                    vel_clamps,
                    accel_clamps,
                    self._config.smoothing_alpha,
                    self._config.speed_scale,
                )

            return filtered

        except Exception as e:
            logger.error(
                "Safety pipeline exception — returning hold positions "
                "(fail-closed): %s",
                e,
            )
            return self._get_hold_output(action, observation)

    def trigger_estop(self, observation: Dict[str, float]) -> None:
        """Trigger emergency stop.  Snapshot current positions as hold targets."""
        self._estop = True
        self._hold_positions = dict(observation)
        self._readings.estop_active = True
        logger.critical("Safety pipeline ESTOP triggered")

    def clear_estop(self) -> None:
        """Clear ESTOP state.  Caller must verify safety before clearing."""
        self._estop = False
        self._readings.estop_active = False
        logger.info("Safety pipeline ESTOP cleared")

    def update_speed_scale(self, scale: float) -> None:
        """Update runtime speed scale override, clamped to [0.1, 1.0].

        This sets a runtime override that does NOT mutate the original
        SafetyConfig.  Pass ``None`` to revert to the config default.
        """
        self._runtime_speed_scale = max(0.1, min(1.0, scale))
        self._readings.speed_scale = self._runtime_speed_scale

    def get_readings(self) -> SafetyReadings:
        """Return a snapshot of current safety readings."""
        return replace(self._readings)

    def reset(self) -> None:
        """Clear all state between deployments."""
        self._prev_positions.clear()
        self._prev_velocities.clear()
        self._prev_output.clear()
        self._hold_positions.clear()
        self._frame_count = 0
        self._estop = False
        self._runtime_speed_scale = None
        self._readings = SafetyReadings(speed_scale=self._config.speed_scale)

    # ------------------------------------------------------------------
    # Stage 1 — Joint Limit Clamp
    # ------------------------------------------------------------------

    def _clamp_joint_limits(self, action: Dict[str, float]) -> Dict[str, float]:
        """Hard clamp every motor to [min, max] from config.joint_limits."""
        result: Dict[str, float] = {}
        clamp_count = 0

        for motor, target in action.items():
            limits = self._config.joint_limits.get(motor)
            if limits is not None:
                lo, hi = limits
                clamped = max(lo, min(hi, target))
                if clamped != target:
                    clamp_count += 1
                result[motor] = clamped
            else:
                result[motor] = target

        if clamp_count:
            self._readings.active_clamps["joint_limits"] = clamp_count
        return result

    # ------------------------------------------------------------------
    # Stage 2 — Velocity Limiter (proper delta clamping)
    # ------------------------------------------------------------------

    def _limit_velocity(
        self,
        action: Dict[str, float],
        observation: Dict[str, float],
        dt: float,
    ) -> Dict[str, float]:
        """Limit command trajectory velocity to max_velocity * dt.

        Uses the previous filtered output as the trajectory reference
        (not the observation).  This limits how fast the COMMAND TARGET
        moves between frames while allowing the position error
        (target − observation) to grow naturally.

        For MIT impedance motors this is critical: torque = kp × error,
        so capping position error directly caps restoring torque.
        Trajectory-based limiting lets torque build up to counteract
        gravity and friction.

        A per-motor max-position-error cap prevents unbounded torque
        buildup when a motor is physically blocked.
        """
        result: Dict[str, float] = {}
        clamp_count = 0

        for motor, target in action.items():
            # Trajectory reference: previous filtered output, falling
            # back to observation on the very first frame.
            prev = self._prev_output.get(
                motor, observation.get(motor, target)
            )
            delta = target - prev
            max_delta = self._effective_max_velocity(motor) * dt

            if abs(delta) > max_delta:
                clamped_delta = max_delta if delta > 0 else -max_delta
                filtered = prev + clamped_delta
                clamp_count += 1
            else:
                filtered = target

            # Safety cap: limit max position error (filtered - obs)
            # to prevent unbounded torque if motor is blocked.
            obs_pos = observation.get(motor, filtered)
            max_error = self._effective_max_position_error(motor)
            error = filtered - obs_pos
            if abs(error) > max_error:
                filtered = obs_pos + max_error * (
                    1.0 if error > 0 else -1.0
                )

            result[motor] = filtered

            # Record trajectory velocity for readings
            self._readings.per_motor_velocity[motor] = (
                abs(delta) / dt if dt > 0 else 0.0
            )

        if clamp_count:
            self._readings.active_clamps["velocity"] = clamp_count
        return result

    # ------------------------------------------------------------------
    # Stage 3 — Acceleration Filter + EMA Smoothing
    # ------------------------------------------------------------------

    def _filter_acceleration(
        self,
        action: Dict[str, float],
        dt: float,
    ) -> Dict[str, float]:
        """Limit acceleration and apply EMA smoothing."""
        result: Dict[str, float] = {}
        clamp_count = 0

        for motor, target in action.items():
            # Compute implied velocity from position delta
            prev_pos = self._prev_positions.get(motor)
            if prev_pos is not None and dt > 0:
                current_velocity = (target - prev_pos) / dt
            else:
                current_velocity = 0.0

            # Acceleration limiting
            prev_vel = self._prev_velocities.get(motor, 0.0)
            if dt > 0:
                acceleration = (current_velocity - prev_vel) / dt
            else:
                acceleration = 0.0

            if prev_pos is not None and abs(acceleration) > self._config.max_acceleration:
                max_accel = self._config.max_acceleration
                clamped_accel = max_accel if acceleration > 0 else -max_accel
                clamped_velocity = prev_vel + clamped_accel * dt
                target = prev_pos + clamped_velocity * dt
                clamp_count += 1

            # EMA smoothing
            alpha = self._config.smoothing_alpha
            prev_output = self._prev_output.get(motor)
            if prev_output is not None and alpha < 1.0:
                smoothed = alpha * target + (1.0 - alpha) * prev_output
            else:
                smoothed = target

            result[motor] = smoothed

        if clamp_count:
            self._readings.active_clamps["acceleration"] = clamp_count
        return result

    # ------------------------------------------------------------------
    # Stage 4 — Torque Monitor
    # ------------------------------------------------------------------

    def _check_torque(self, robot) -> bool:
        """Periodic torque check via SafetyLayer (every Nth frame).

        Returns True if safe.
        """
        if self._safety_layer is None:
            return True

        if self._frame_count % self._config.torque_check_interval != 0:
            return True

        try:
            safe = self._safety_layer.check_all_limits(robot)
            if hasattr(self._safety_layer, "latest_torques") and self._safety_layer.latest_torques:
                self._readings.per_motor_torque = dict(
                    self._safety_layer.latest_torques
                )
            return safe
        except Exception as e:
            logger.error("Torque check error: %s", e)
            # Single transient failure — don't ESTOP (SafetyLayer has its
            # own consecutive-failure counter for fail-closed behavior).
            return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_max_velocity(self, motor_name: str) -> float:
        """Max velocity for a motor, using runtime override if set."""
        from .types import DEFAULT_VELOCITY_LIMITS, FALLBACK_VELOCITY_LIMIT

        model = self._config.motor_models.get(motor_name, "")
        base_limit = DEFAULT_VELOCITY_LIMITS.get(model, FALLBACK_VELOCITY_LIMIT)
        scale = self._runtime_speed_scale if self._runtime_speed_scale is not None else self._config.speed_scale
        return base_limit * max(0.1, min(1.0, scale))

    def _effective_max_position_error(self, motor_name: str) -> float:
        """Max allowed position error (filtered − observation) for a motor."""
        from .types import DEFAULT_MAX_POSITION_ERROR, FALLBACK_MAX_POSITION_ERROR

        model = self._config.motor_models.get(motor_name, "")
        return DEFAULT_MAX_POSITION_ERROR.get(model, FALLBACK_MAX_POSITION_ERROR)

    def _get_hold_output(
        self,
        action: Dict[str, float],
        observation: Dict[str, float],
    ) -> Dict[str, float]:
        """Return hold-position for every motor in *action*.

        Priority: stored hold snapshot > observation > original action.
        """
        if self._hold_positions:
            return {
                m: self._hold_positions.get(m, observation.get(m, v))
                for m, v in action.items()
            }
        if observation:
            return {m: observation.get(m, v) for m, v in action.items()}
        return dict(action)

    def _update_prev_state(
        self,
        filtered: Dict[str, float],
        observation: Dict[str, float],
        dt: float,
    ) -> None:
        """Commit frame state for next-frame acceleration / EMA calculations."""
        for motor, pos in filtered.items():
            prev = self._prev_positions.get(motor)
            if prev is not None and dt > 0:
                self._prev_velocities[motor] = (pos - prev) / dt
            else:
                self._prev_velocities[motor] = 0.0
            self._prev_positions[motor] = pos
        self._prev_output = dict(filtered)

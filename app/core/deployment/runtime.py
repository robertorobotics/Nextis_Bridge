"""Unified deployment runtime for all policy execution modes.

Replaces the three separate control loops (orchestrator._inference_loop,
hil/loop._hil_loop, rl/service._run_actor_loop) with a single 30Hz loop.
The mode only determines (a) where the action comes from and (b) what
happens with the data.  Safety is always applied.
"""

import logging
import math
import threading
import time
from typing import Dict, List, Optional

from .intervention import InterventionDetector
from .observation_builder import ObservationBuilder
from .safety_pipeline import SafetyPipeline
from .types import (
    ActionSource,
    DeploymentConfig,
    DeploymentMode,
    DeploymentStatus,
    RuntimeState,
    SafetyConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOP_HZ = 30
DRY_RUN_FRAMES = 30


class DeploymentRuntime:
    """Unified runtime for INFERENCE, HIL, and HIL_SERL deployment modes.

    Usage::

        runtime = DeploymentRuntime(teleop, training, arm_registry, cameras, lock)
        runtime.start(config, active_arm_ids=["leader_left", "follower_left"])
        # ... later ...
        runtime.stop()
    """

    def __init__(
        self,
        teleop_service,
        training_service,
        arm_registry,
        camera_service,
        robot_lock: threading.Lock,
    ):
        self._teleop = teleop_service
        self._training = training_service
        self._arm_registry = arm_registry
        self._camera_service = camera_service
        self._robot_lock = robot_lock

        # Runtime state
        self._state = RuntimeState.IDLE
        self._state_lock = threading.Lock()
        self._config: Optional[DeploymentConfig] = None
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Per-session objects (set during start, cleared on stop)
        self._policy = None
        self._checkpoint_path = None
        self._policy_config = None
        self._obs_builder: Optional[ObservationBuilder] = None
        self._safety_pipeline: Optional[SafetyPipeline] = None
        self._intervention_detector: Optional[InterventionDetector] = None
        self._leader = None
        self._follower = None
        self._arm_defs: List = []
        self._active_arm_ids: List[str] = []

        # Dry-run diagnostics log (populated during dry_run=True)
        self._dry_run_log: list = []

        # Post-pre-position snapshot for drift detection
        self._post_preposition_positions: Dict[str, float] = {}
        # Velocity limit to restore in _control_loop (deferred from pre-position)
        self._pre_position_old_vel: Optional[float] = None

        # Counters
        self._frame_count = 0
        self._episode_count = 0
        self._current_episode_frames = 0
        self._autonomous_frames = 0
        self._human_frames = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        config: DeploymentConfig,
        active_arm_ids: List[str],
    ) -> None:
        """Start a deployment session.

        Args:
            config: Deployment configuration (mode, policy_id, safety, etc.).
            active_arm_ids: Follower arm IDs to control.

        Raises:
            RuntimeError: If not in IDLE state or startup fails.
        """
        if not self._transition(RuntimeState.STARTING):
            raise RuntimeError(
                f"Cannot start from state {self._state.value}"
            )

        try:
            self._config = config
            self._active_arm_ids = list(active_arm_ids) if active_arm_ids else []
            self._stop_event.clear()
            self._frame_count = 0
            self._episode_count = 0
            self._current_episode_frames = 0
            self._autonomous_frames = 0
            self._human_frames = 0

            # 1. Resolve arms (sets _follower, _leader, _arm_defs)
            self._resolve_arms(active_arm_ids)

            # ---------------------------------------------------------------
            # Phase A: Non-motor setup (policy, safety, observation builder)
            # All done BEFORE motor operations to eliminate the loading gap
            # that previously let motors drift unmonitored for 200-500ms.
            # ---------------------------------------------------------------

            # 2. Load policy
            self._load_policy(config.policy_id)

            # 2b. Apply temporal ensembling override (must happen BEFORE reset,
            #     because reset() initializes the ensembler's internal state)
            self._apply_temporal_ensemble_override()

            # 2c. Reset policy internal state (clears stale action queue /
            #     temporal ensembler from prior deployments)
            if hasattr(self._policy, "reset"):
                self._policy.reset()
                logger.info("Policy internal state reset")

            # 3. Auto-select safety preset if no custom safety tuning provided
            if not config.safety.motor_models:
                policy_type = getattr(self._policy_config, "policy_type", "")
                preset_config = SafetyConfig.from_policy_type(policy_type)
                config.safety.smoothing_alpha = preset_config.smoothing_alpha
                config.safety.max_acceleration = preset_config.max_acceleration
                config.safety.speed_scale = preset_config.speed_scale
                logger.info(
                    "Auto-selected '%s' safety preset for policy type '%s'",
                    policy_type or "conservative",
                    policy_type,
                )

            # 3b. Populate safety config from arm definitions
            self._populate_safety_config(config.safety)

            # 4. Create safety pipeline
            safety_layer = getattr(self._teleop, "safety", None)
            self._safety_pipeline = SafetyPipeline(
                config.safety, safety_layer=safety_layer
            )

            # 5. Create observation builder
            self._obs_builder = ObservationBuilder(
                checkpoint_path=self._checkpoint_path,
                policy=self._policy,
                policy_type=getattr(self._policy_config, "policy_type", ""),
                task=config.task or "",
            )

            # 6. Auto-enable extended observation if policy expects vel/tau
            state_names = self._obs_builder.get_training_state_names()
            if state_names:
                needs_extended = any(
                    n.endswith(".vel") or n.endswith(".tau")
                    for n in state_names
                )
                if needs_extended and self._follower is not None:
                    if (
                        hasattr(self._follower, "config")
                        and hasattr(self._follower.config, "record_extended_state")
                    ):
                        self._follower.config.record_extended_state = True
                        logger.warning(
                            "DEPLOY: Auto-enabled extended observation on follower "
                            "(policy trained with %d states including vel/tau)",
                            len(state_names),
                        )
                    else:
                        logger.warning(
                            "DEPLOY CRITICAL: Policy expects extended state (vel/tau) "
                            "but follower %s does not support record_extended_state. "
                            "%d states will be zero at deployment!",
                            type(self._follower).__name__,
                            sum(
                                1
                                for n in state_names
                                if n.endswith(".vel") or n.endswith(".tau")
                            ),
                        )
                elif needs_extended and self._follower is None:
                    logger.warning(
                        "DEPLOY CRITICAL: Policy expects extended state (vel/tau) "
                        "but no follower is available!"
                    )
            else:
                logger.warning(
                    "DEPLOY: Could not load training state names from dataset — "
                    "cannot verify extended observation requirements"
                )

            # 6b. Check camera availability if policy expects cameras
            if hasattr(self._policy, "config") and hasattr(self._policy.config, "image_features"):
                expected_cams = [
                    k.split(".")[-1]
                    for k in self._policy.config.image_features
                ]
                if expected_cams:
                    available_cams = (
                        list(self._camera_service.cameras.keys())
                        if self._camera_service
                        else []
                    )
                    missing = [c for c in expected_cams if c not in available_cams]
                    if missing:
                        logger.warning(
                            "DEPLOY CRITICAL: Policy expects cameras %s but only %s "
                            "are available. Missing cameras will produce zero input — "
                            "policy output will be UNRELIABLE!",
                            expected_cams,
                            available_cams or "none",
                        )
                    else:
                        logger.warning(
                            "DEPLOY: All %d expected cameras available: %s",
                            len(expected_cams),
                            expected_cams,
                        )

            # 7. Create intervention detector (for HIL/SERL modes)
            policy_arms = (
                getattr(self._policy_config, "arms", None) or ["left", "right"]
            )
            self._intervention_detector = InterventionDetector(
                policy_arms=policy_arms,
                loop_hz=config.loop_hz,
            )

            # ---------------------------------------------------------------
            # Phase B: Motor operations (enable, pre-position, control loop)
            # Kept together to minimize unmonitored time.
            # ---------------------------------------------------------------

            # 8. Re-enable motors (may be disabled from teleop homing)
            self._enable_follower_motors()
            self._log_follower_positions("after enable")

            # 9. Pre-position follower to leader's current position
            self._pre_position_to_leader()
            self._log_follower_positions("after pre-position")

            # 10. Start recording for HIL/SERL modes
            if config.mode in (DeploymentMode.HIL, DeploymentMode.HIL_SERL):
                self._start_recording(config)

            # 11. Start control loop
            self._transition(RuntimeState.RUNNING)
            self._loop_thread = threading.Thread(
                target=self._control_loop,
                name="deployment-runtime",
                daemon=True,
            )
            self._loop_thread.start()
            logger.info(
                "Deployment started: mode=%s, policy=%s",
                config.mode.value,
                config.policy_id,
            )

        except Exception as e:
            logger.error("Deployment start failed: %s", e)
            self._transition(RuntimeState.ERROR)
            raise RuntimeError(f"Deployment start failed: {e}") from e

    def stop(self) -> None:
        """Stop the deployment session."""
        if self._state == RuntimeState.IDLE:
            return

        self._transition(RuntimeState.STOPPING)
        self._stop_event.set()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        # Cleanup
        if self._safety_pipeline:
            self._safety_pipeline.reset()
        if self._intervention_detector:
            self._intervention_detector.reset()
        if self._policy and hasattr(self._policy, "reset"):
            self._policy.reset()

        self._policy = None
        self._checkpoint_path = None
        self._policy_config = None
        self._obs_builder = None
        self._safety_pipeline = None
        self._intervention_detector = None
        self._leader = None
        self._follower = None
        self._arm_defs = []
        self._config = None
        self._loop_thread = None
        self._dry_run_log = []

        self._transition(RuntimeState.IDLE)
        logger.info("Deployment stopped")

    def pause(self) -> bool:
        """Pause the deployment (hold position)."""
        return self._transition(RuntimeState.PAUSED)

    def resume(self) -> bool:
        """Resume autonomous execution from PAUSED state."""
        return self._transition(RuntimeState.RUNNING)

    def reset(self) -> bool:
        """Reset from ESTOP or ERROR back to IDLE.

        Clears safety pipeline state.  Caller should verify physical safety
        before calling.
        """
        if self._state not in (RuntimeState.ESTOP, RuntimeState.ERROR):
            return False

        self._stop_event.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        if self._safety_pipeline:
            self._safety_pipeline.clear_estop()
            self._safety_pipeline.reset()

        # Force transition (ESTOP/ERROR have no valid transitions in the table,
        # but reset is the explicit escape hatch)
        with self._state_lock:
            self._state = RuntimeState.IDLE
        logger.info("Deployment reset from %s to IDLE", self._state.value)
        return True

    def estop(self) -> bool:
        """Emergency stop — hold position immediately."""
        if self._safety_pipeline:
            obs = self._get_observation() or {}
            self._safety_pipeline.trigger_estop(obs)
        with self._state_lock:
            self._state = RuntimeState.ESTOP
        logger.critical("E-STOP triggered via API")
        return True

    def update_speed_scale(self, scale: float) -> None:
        """Update the safety pipeline speed scale."""
        if self._safety_pipeline:
            self._safety_pipeline.update_speed_scale(scale)

    def get_status(self) -> DeploymentStatus:
        """Return a snapshot of current deployment status."""
        safety_readings = {}
        if self._safety_pipeline:
            readings = self._safety_pipeline.get_readings()
            safety_readings = {
                "per_motor_velocity": readings.per_motor_velocity,
                "per_motor_torque": readings.per_motor_torque,
                "active_clamps": readings.active_clamps,
                "estop_active": readings.estop_active,
                "speed_scale": readings.speed_scale,
            }

        policy_config_dict = None
        if self._policy_config:
            policy_config_dict = {
                "cameras": getattr(self._policy_config, "cameras", []),
                "arms": getattr(self._policy_config, "arms", []),
                "policy_type": getattr(self._policy_config, "policy_type", ""),
            }

        return DeploymentStatus(
            state=self._state,
            mode=self._config.mode if self._config else DeploymentMode.INFERENCE,
            frame_count=self._frame_count,
            episode_count=self._episode_count,
            current_episode_frames=self._current_episode_frames,
            safety=safety_readings,
            policy_config=policy_config_dict,
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Unified 30Hz control loop for all deployment modes."""
        loop_hz = self._config.loop_hz if self._config else DEFAULT_LOOP_HZ
        loop_period = 1.0 / loop_hz
        dt = loop_period
        is_hil = self._config and self._config.mode in (
            DeploymentMode.HIL,
            DeploymentMode.HIL_SERL,
        )
        is_dry_run = self._config and self._config.dry_run
        if is_dry_run:
            self._dry_run_log = []
            logger.info(
                "DRY RUN mode: will run %d frames without sending actions",
                DRY_RUN_FRAMES,
            )
        logger.info("Control loop running at %dHz", loop_hz)

        # Reset policy state at loop entry (belt-and-suspenders with start())
        if self._policy and hasattr(self._policy, "reset"):
            self._policy.reset()

        # Log positions at policy start and check for drift
        self._log_follower_positions("policy start")
        if self._post_preposition_positions:
            try:
                obs = self._follower.get_observation() if self._follower else {}
                for key, prev in self._post_preposition_positions.items():
                    current = obs.get(key)
                    if current is not None and abs(current - prev) > 0.05:
                        logger.warning(
                            "DEPLOY DRIFT: %s moved %.4f rad during loading gap "
                            "(%.4f → %.4f)",
                            key,
                            current - prev,
                            prev,
                            current,
                        )
            except Exception:
                pass

        # Restore velocity_limit that was deferred from pre-position.
        # Kept at PRE_POSITION_VEL during the gap to prevent rate-limiter
        # bypass; now safe to restore because the control loop will send
        # motor commands every frame.
        if self._pre_position_old_vel is not None:
            try:
                from lerobot.motors.damiao.damiao import DamiaoMotorsBus

                bus = getattr(self._follower, "bus", None)
                if bus and isinstance(bus, DamiaoMotorsBus):
                    bus.velocity_limit = self._pre_position_old_vel
                    logger.info(
                        "Control loop: velocity_limit restored to %.2f",
                        self._pre_position_old_vel,
                    )
            except Exception:
                pass
            self._pre_position_old_vel = None

        # Capture the target speed_scale before warmup ramp modifies it
        target_speed_scale = (
            self._config.safety.speed_scale if self._config else 1.0
        )

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            try:
                # 1. Get observation from follower
                raw_obs = self._get_observation()
                if raw_obs is None:
                    time.sleep(0.01)
                    continue

                # 2. Intervention detection (HIL/SERL only)
                action_source = ActionSource.POLICY
                if is_hil and self._intervention_detector and self._leader:
                    is_intervening, velocity = self._intervention_detector.check(
                        self._leader
                    )
                    action_source = self._update_state_from_intervention(
                        is_intervening
                    )

                # 3. Determine action based on source
                if is_dry_run:
                    action_source = ActionSource.POLICY
                elif self._state == RuntimeState.PAUSED:
                    action_source = ActionSource.HOLD

                # In dry-run mode, get intermediates for diagnostics
                policy_obs = None
                action_tensor = None
                if is_dry_run:
                    result = self._get_policy_action(raw_obs, return_raw=True)
                    if result[0] is None:
                        time.sleep(0.001)
                        continue
                    action, policy_obs, action_tensor = result
                else:
                    action = self._get_action(action_source, raw_obs)
                    if action is None:
                        time.sleep(0.001)
                        continue

                # 4. ALWAYS apply safety pipeline
                observation_positions = {
                    k: v
                    for k, v in raw_obs.items()
                    if isinstance(v, (int, float))
                    and not k.endswith((".vel", ".tau"))
                }

                # 4a. Velocity ramp during warmup: start at 30% speed,
                #     linearly increase to full speed over warmup period
                warmup = self._config.warmup_frames if self._config else 0
                if warmup > 0 and self._frame_count < warmup:
                    ramp = 0.3 + 0.7 * (self._frame_count / warmup)
                    self._safety_pipeline.update_speed_scale(
                        target_speed_scale * ramp
                    )
                elif self._frame_count == warmup and warmup > 0:
                    self._safety_pipeline.update_speed_scale(target_speed_scale)

                filtered_action = self._safety_pipeline.process(
                    action, observation_positions, robot=self._follower, dt=dt
                )

                # 4b. Propagate safety-pipeline ESTOP to runtime state
                if self._safety_pipeline._estop and self._state != RuntimeState.ESTOP:
                    with self._state_lock:
                        self._state = RuntimeState.ESTOP
                    logger.critical(
                        "Runtime ESTOP: safety pipeline triggered torque emergency stop"
                    )

                # 4c. Warm-up blending: ramp from current position to
                #     policy target over the first N frames to avoid jerks
                if warmup > 0 and self._frame_count < warmup:
                    blend_alpha = (self._frame_count + 1) / warmup
                    for key in filtered_action:
                        if key in observation_positions:
                            current = observation_positions[key]
                            target = filtered_action[key]
                            filtered_action[key] = current + (target - current) * blend_alpha

                # 5. Dry-run diagnostics or send to robot
                if is_dry_run:
                    self._log_dry_run_frame(
                        self._frame_count, policy_obs, action_tensor,
                        action, filtered_action, raw_obs,
                    )
                else:
                    self._send_action(filtered_action)

                    # 6. Cache for recording
                    self._cache_for_recording(filtered_action, raw_obs)

                # 7. Update counters
                self._frame_count += 1
                self._current_episode_frames += 1
                if action_source == ActionSource.HUMAN:
                    self._human_frames += 1
                elif action_source == ActionSource.POLICY:
                    self._autonomous_frames += 1

                # 7b. Auto-stop after DRY_RUN_FRAMES in dry-run mode
                if is_dry_run and self._frame_count >= DRY_RUN_FRAMES:
                    logger.info(
                        "Dry run complete (%d frames)", self._frame_count
                    )
                    self._stop_event.set()

            except Exception as e:
                logger.error("Control loop error: %s", e)

            # Maintain loop rate
            elapsed = time.monotonic() - t0
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Control loop stopped")

    # ------------------------------------------------------------------
    # Action sources
    # ------------------------------------------------------------------

    def _get_action(
        self, source: ActionSource, raw_obs: dict
    ) -> Optional[Dict[str, float]]:
        """Get action dict based on the current action source."""
        if source == ActionSource.HOLD:
            return {
                k: v
                for k, v in raw_obs.items()
                if isinstance(v, (int, float))
                and not k.endswith((".vel", ".tau"))
            }

        if source == ActionSource.HUMAN:
            return self._get_human_action()

        # POLICY
        return self._get_policy_action(raw_obs)

    def _get_policy_action(self, raw_obs: dict, return_raw: bool = False):
        """Run policy inference and return denormalized action dict.

        When *return_raw* is True, returns a 3-tuple
        ``(action_dict, policy_obs, action_tensor)`` so that callers
        (e.g. dry-run diagnostics) can inspect intermediates.
        """
        if self._policy is None or self._obs_builder is None:
            return (None, None, None) if return_raw else None

        policy_obs = self._obs_builder.prepare_observation(raw_obs)
        action_tensor = self._policy.select_action(policy_obs)

        movement_scale = 1.0
        if self._config:
            movement_scale = self._config.movement_scale

        action_dict = self._obs_builder.convert_action_to_dict(
            action_tensor, raw_obs, movement_scale=movement_scale
        )

        if return_raw:
            return action_dict, policy_obs, action_tensor
        return action_dict

    def _get_human_action(self) -> Optional[Dict[str, float]]:
        """Read leader arm positions for human teleop."""
        if self._leader is None:
            return None
        try:
            action = self._leader.get_action()
            return dict(action) if action else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> Optional[dict]:
        """Get observation from follower robot with lock.

        Merges motor positions from the follower with camera frames
        from CameraService (async_read, ZOH pattern).
        """
        if self._follower is None:
            return None
        if hasattr(self._follower, "is_connected") and not self._follower.is_connected:
            return None

        try:
            if self._robot_lock:
                with self._robot_lock:
                    obs = self._follower.get_observation()
            else:
                obs = self._follower.get_observation()
        except Exception as e:
            logger.debug("Observation error: %s", e)
            return None

        # Inject camera frames from CameraService
        if self._camera_service:
            for cam_key, cam in self._camera_service.cameras.items():
                try:
                    frame = cam.async_read(blocking=False)
                    if frame is not None:
                        obs[cam_key] = frame
                except Exception:
                    pass

        return obs

    # ------------------------------------------------------------------
    # Action sending
    # ------------------------------------------------------------------

    def _send_action(self, action_dict: Dict[str, float]) -> None:
        """Send action to robot, handling bimanual/single-arm cases."""
        if self._follower is None or not action_dict:
            return

        try:
            if self._robot_lock:
                with self._robot_lock:
                    self._send_partial_action(self._follower, action_dict)
            else:
                self._send_partial_action(self._follower, action_dict)
        except Exception as e:
            logger.debug("Send action error: %s", e)

    @staticmethod
    def _send_partial_action(robot, action_dict: dict) -> None:
        """Send action only to arms that have entries in the dict.

        For bimanual robots, splits by left_/right_ prefix and strips
        the prefix before sending to individual arms.
        """
        is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
        if not is_bimanual:
            # Strip any left_/right_ prefix so single-arm drivers recognise the keys.
            stripped = {}
            for k, v in action_dict.items():
                if k.startswith("left_"):
                    stripped[k.removeprefix("left_")] = v
                elif k.startswith("right_"):
                    stripped[k.removeprefix("right_")] = v
                else:
                    stripped[k] = v
            robot.send_action(stripped)
            return

        left_action = {
            k.removeprefix("left_"): v
            for k, v in action_dict.items()
            if k.startswith("left_")
        }
        right_action = {
            k.removeprefix("right_"): v
            for k, v in action_dict.items()
            if k.startswith("right_")
        }

        if left_action:
            try:
                robot.left_arm.send_action(left_action)
            except Exception as e:
                logger.debug("Left arm send error: %s", e)

        if right_action:
            try:
                robot.right_arm.send_action(right_action)
            except Exception as e:
                logger.debug("Right arm send error: %s", e)

    # ------------------------------------------------------------------
    # Recording cache
    # ------------------------------------------------------------------

    def _cache_for_recording(
        self, action_dict: Dict[str, float], raw_obs: dict
    ) -> None:
        """Cache action in teleop for the recording capture thread."""
        if not hasattr(self._teleop, "_action_lock") or not hasattr(
            self._teleop, "_latest_leader_action"
        ):
            return

        padded = self._pad_action_for_recording(action_dict, raw_obs)
        with self._teleop._action_lock:
            self._teleop._latest_leader_action = padded

    def _pad_action_for_recording(
        self, action_dict: dict, raw_obs: dict
    ) -> dict:
        """Pad action dict with missing motor positions for recording.

        If the policy only outputs one arm but the dataset expects both,
        fill in the other arm's current positions from raw_obs.
        """
        if not hasattr(self._teleop, "dataset") or self._teleop.dataset is None:
            return dict(action_dict)

        try:
            features = self._teleop.dataset.features
            if "action" not in features or "names" not in features["action"]:
                return dict(action_dict)

            expected_names = features["action"]["names"]
            if len(action_dict) >= len(expected_names):
                return dict(action_dict)

            padded = dict(action_dict)
            for name in expected_names:
                if name not in padded and name in raw_obs:
                    padded[name] = float(raw_obs[name])

            return padded

        except Exception:
            return dict(action_dict)

    # ------------------------------------------------------------------
    # Dry-run diagnostics
    # ------------------------------------------------------------------

    def _log_dry_run_frame(
        self,
        frame: int,
        policy_obs: dict,
        action_tensor,
        denorm_action: dict,
        filtered_action: dict,
        raw_obs: dict,
    ) -> dict:
        """Log comprehensive diagnostics for one dry-run frame."""
        entry: Dict = {"frame": frame}
        logger.info("=== DRY RUN FRAME %d/%d ===", frame + 1, DRY_RUN_FRAMES)

        # Observation state diagnostics
        if policy_obs and "observation.state" in policy_obs:
            s = policy_obs["observation.state"]
            if hasattr(s, "shape"):
                info = {
                    "shape": list(s.shape),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
                logger.info(
                    "  OBS STATE: shape=%s min=%.4f max=%.4f mean=%.4f",
                    info["shape"], info["min"], info["max"], info["mean"],
                )
                entry["obs_state"] = info

        # Image observation diagnostics
        if policy_obs:
            for k, v in policy_obs.items():
                if "image" in k and hasattr(v, "shape"):
                    img_info = {
                        "shape": list(v.shape),
                        "min": float(v.min()),
                        "max": float(v.max()),
                    }
                    logger.info(
                        "  OBS IMAGE %s: shape=%s min=%.3f max=%.3f",
                        k, img_info["shape"], img_info["min"], img_info["max"],
                    )
                    entry.setdefault("obs_images", {})[k] = img_info

        # Raw action tensor diagnostics
        if action_tensor is not None and hasattr(action_tensor, "shape"):
            act_info = {
                "shape": list(action_tensor.shape),
                "min": float(action_tensor.min()),
                "max": float(action_tensor.max()),
                "mean": float(action_tensor.mean()),
            }
            logger.info(
                "  RAW ACTION: shape=%s min=%.4f max=%.4f mean=%.4f",
                act_info["shape"], act_info["min"], act_info["max"],
                act_info["mean"],
            )
            entry["raw_action"] = act_info

        # Denormalized action
        if denorm_action:
            denorm_fmt = {k: f"{v:+.4f}" for k, v in denorm_action.items()}
            logger.info("  DENORM ACTION: %s", denorm_fmt)
            entry["denorm_action"] = {
                k: float(v) for k, v in denorm_action.items()
            }

        # Safety filter delta
        if filtered_action and denorm_action:
            delta = {
                k: float(filtered_action[k] - denorm_action[k])
                for k in filtered_action
                if k in denorm_action
            }
            delta_fmt = {k: f"{v:+.4f}" for k, v in delta.items()}
            logger.info("  SAFETY DELTA: %s", delta_fmt)
            entry["safety_delta"] = delta

        # Current robot positions
        obs_positions = {
            k: float(v)
            for k, v in raw_obs.items()
            if isinstance(v, (int, float)) and k.endswith(".pos")
        }
        if obs_positions:
            logger.info(
                "  ROBOT POS: %s",
                {k: f"{v:+.4f}" for k, v in obs_positions.items()},
            )
            entry["robot_pos"] = obs_positions

        # Range validation on first frame
        if frame == 0:
            warnings = self._validate_dry_run_ranges(
                policy_obs, action_tensor, denorm_action
            )
            if warnings:
                entry["warnings"] = warnings

        self._dry_run_log.append(entry)
        return entry

    def _validate_dry_run_ranges(
        self,
        policy_obs: Optional[dict],
        action_tensor,
        denorm_action: Optional[dict],
    ) -> List[str]:
        """Check if values are in sane ranges. Returns list of warning strings."""
        warnings: List[str] = []

        # Check normalized observation state
        if policy_obs and "observation.state" in policy_obs:
            s = policy_obs["observation.state"]
            if hasattr(s, "min") and hasattr(s, "max"):
                s_min, s_max = float(s.min()), float(s.max())
                if s_min < -3.0 or s_max > 3.0:
                    msg = (
                        f"Observation state range [min={s_min:.2f}, "
                        f"max={s_max:.2f}] exceeds [-3, 3]. "
                        "Normalization may not be applied. "
                        "Check norm stats keys in checkpoint."
                    )
                    logger.error("RANGE CHECK: %s", msg)
                    warnings.append(msg)

        # Check raw action tensor
        if action_tensor is not None and hasattr(action_tensor, "min"):
            a_min, a_max = float(action_tensor.min()), float(action_tensor.max())
            if a_min < -3.0 or a_max > 3.0:
                msg = (
                    f"Raw action tensor range [min={a_min:.2f}, "
                    f"max={a_max:.2f}] exceeds [-3, 3]. "
                    "Output may be raw radians instead of normalized. "
                    "Check policy output normalization."
                )
                logger.error("RANGE CHECK: %s", msg)
                warnings.append(msg)

        # Check denormalized actions (motor positions, should be in [-pi, pi])
        if denorm_action:
            for k, v in denorm_action.items():
                if abs(v) > math.pi:
                    msg = (
                        f"Denormalized action '{k}'={v:+.4f} "
                        f"exceeds [-pi, pi]. "
                        "Denormalization may be incorrect."
                    )
                    logger.error("RANGE CHECK: %s", msg)
                    warnings.append(msg)
                    break  # One warning per check is enough

        return warnings

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: RuntimeState) -> bool:
        """Attempt a state transition.  Returns True if successful."""
        with self._state_lock:
            if RuntimeState.can_transition(self._state, new_state):
                old = self._state
                self._state = new_state
                logger.debug("State: %s → %s", old.value, new_state.value)
                return True

            logger.warning(
                "Invalid transition: %s → %s",
                self._state.value,
                new_state.value,
            )
            return False

    def _update_state_from_intervention(
        self, is_intervening: bool
    ) -> ActionSource:
        """Update runtime state based on intervention detection.

        Returns the ActionSource to use for this frame.
        """
        if is_intervening:
            if self._state == RuntimeState.RUNNING:
                self._transition(RuntimeState.HUMAN_ACTIVE)
            return ActionSource.HUMAN

        # Not intervening — check if human went idle
        if self._state == RuntimeState.HUMAN_ACTIVE:
            if self._intervention_detector and self._intervention_detector.is_idle():
                self._transition(RuntimeState.PAUSED)
                return ActionSource.HOLD

            # Still in human mode (within idle timeout)
            return ActionSource.HUMAN

        if self._state == RuntimeState.PAUSED:
            return ActionSource.HOLD

        return ActionSource.POLICY

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _resolve_arms(self, active_arm_ids: List[str]) -> None:
        """Resolve leader/follower from arm registry pairings.

        Follows the same pattern as teleop/service.py start().
        """
        if not self._arm_registry:
            raise RuntimeError("No arm registry available")

        pairings = self._arm_registry.get_active_pairings(active_arm_ids)
        if not pairings:
            raise RuntimeError(
                f"No pairings found for arms: {active_arm_ids}"
            )

        # Use first pairing (deployment controls one policy at a time)
        pairing = pairings[0]
        leader_id = pairing["leader_id"]
        follower_id = pairing["follower_id"]

        # Auto-connect if needed
        for arm_id in (leader_id, follower_id):
            if arm_id not in self._arm_registry.arm_instances:
                logger.info("Auto-connecting arm: %s", arm_id)
                self._arm_registry.connect_arm(arm_id)

        self._leader = self._arm_registry.arm_instances.get(leader_id)
        self._follower = self._arm_registry.arm_instances.get(follower_id)

        if self._follower is None:
            raise RuntimeError(f"Follower arm {follower_id} not available")

        # Collect arm definitions for safety config
        self._arm_defs = [
            self._arm_registry.arms[aid]
            for aid in (leader_id, follower_id)
            if aid in self._arm_registry.arms
        ]

        logger.info(
            "Arms resolved: leader=%s, follower=%s",
            leader_id,
            follower_id,
        )

    def _enable_follower_motors(self) -> None:
        """Re-enable follower motors (they may be disabled from teleop homing)."""
        if self._follower is None:
            return
        try:
            from lerobot.robots.damiao_follower.damiao_follower import (
                DamiaoFollowerRobot,
            )

            if isinstance(self._follower, DamiaoFollowerRobot):
                self._follower.bus.configure()
                logger.info("Follower motors re-enabled (Damiao MIT mode)")
            elif hasattr(self._follower, "bus"):
                self._follower.bus.enable_torque()
                logger.info("Follower motors re-enabled")
        except Exception as e:
            logger.warning("Failed to re-enable follower motors: %s", e)

    def _pre_position_to_leader(self) -> None:
        """Smoothly move follower to leader's current position before inference.

        Training episodes always started from the leader's resting position.
        Uses the homing_loop pattern: low velocity_limit + repeated sync_write.
        """
        if self._leader is None or self._follower is None:
            return

        from app.core.teleop.pairing import DYNAMIXEL_TO_DAMIAO_JOINT_MAP

        # 1. Read leader positions
        try:
            leader_obs = self._leader.get_action()
        except Exception as e:
            logger.warning("Pre-position: cannot read leader: %s", e)
            return

        # 2. Map leader joint names → follower joint names
        target = {}
        for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
            leader_key = f"{dyn_name}.pos"
            if leader_key in leader_obs:
                target[dam_name] = float(leader_obs[leader_key])

        # Fallback: direct name matching (handles Damiao-to-Damiao where
        # leader uses same naming convention as follower, e.g. "base.pos")
        if not target:
            follower_motors = set(
                getattr(self._follower, "_motor_names", [])
            )
            for key, value in leader_obs.items():
                if isinstance(value, (int, float)) and key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    if motor_name in follower_motors:
                        target[motor_name] = float(value)
            if target:
                logger.info(
                    "Pre-position: used direct name matching (%d motors)",
                    len(target),
                )

        if not target:
            logger.warning("Pre-position: no leader positions mapped")
            return

        # 3. Ramp via send_action (applies inversions + joint limits)
        #    with conservative velocity limit on the bus.
        target_action = {f"{k}.pos": v for k, v in target.items()}

        try:
            from lerobot.motors.damiao.damiao import DamiaoMotorsBus

            bus = getattr(self._follower, "bus", None)
            has_damiao_bus = bus and isinstance(bus, DamiaoMotorsBus)

            PRE_POSITION_VEL = 0.05   # Same as homing_vel — conservative
            PRE_POSITION_DURATION = 3.0
            SETTLE_DURATION = 1.0

            old_vel = None
            if has_damiao_bus:
                old_vel = bus.velocity_limit
                bus.velocity_limit = PRE_POSITION_VEL

            logger.info(
                "Pre-positioning follower to leader position "
                "(vel=%.2f, duration=%.1fs, %d joints): %s",
                PRE_POSITION_VEL,
                PRE_POSITION_DURATION,
                len(target),
                {k: f"{v:+.3f}" for k, v in target_action.items()},
            )

            # Ramp phase — with periodic torque monitoring for collision
            t0 = time.monotonic()
            ramp_frame = 0
            collision_detected = False
            while time.monotonic() - t0 < PRE_POSITION_DURATION:
                if self._stop_event.is_set():
                    break
                self._follower.send_action(target_action)
                ramp_frame += 1

                # Torque check every 10 frames (~3Hz) to detect collisions
                if (
                    ramp_frame % 10 == 0
                    and hasattr(self._follower, "get_torques")
                    and hasattr(self._follower, "get_torque_limits")
                ):
                    try:
                        torques = self._follower.get_torques()
                        limits = self._follower.get_torque_limits()
                        for name, tau in torques.items():
                            limit = limits.get(name, 10.0)
                            if abs(tau) > limit * 0.9:
                                logger.warning(
                                    "DEPLOY: Pre-position aborted — %s "
                                    "torque %.1f Nm near limit %.1f Nm "
                                    "(possible collision)",
                                    name,
                                    tau,
                                    limit,
                                )
                                collision_detected = True
                                break
                    except Exception:
                        pass
                if collision_detected:
                    break

                time.sleep(1.0 / 30)

            # Settle phase — hold position to let velocity decay
            logger.info("Pre-position settling (%.1fs)...", SETTLE_DURATION)
            t0 = time.monotonic()
            while time.monotonic() - t0 < SETTLE_DURATION:
                if self._stop_event.is_set():
                    break
                self._follower.send_action(target_action)
                time.sleep(1.0 / 30)

            # NOTE: Do NOT "freeze" motors by sending actual positions as
            # goals.  The MIT position error (p_des − p_actual ≈ gravity/kp)
            # IS the gravity compensation.  Zeroing it removes the torque
            # that holds the arm up, causing it to drop under gravity.
            # Instead, keep the pre-position target as p_des — the MIT
            # controller will hold the arm stable against gravity.

            # Delay velocity_limit restoration until _control_loop() so the
            # rate limiter stays active during the brief gap before the
            # first policy command.
            if has_damiao_bus and old_vel is not None:
                self._pre_position_old_vel = old_vel

            logger.info("Pre-positioning complete")

        except Exception as e:
            logger.warning("Pre-position failed: %s", e)

    def _load_policy(self, policy_id: str) -> None:
        """Load policy from training service.

        Follows the same pattern as orchestrator.py deploy_policy().
        """
        if not self._training:
            raise RuntimeError("No training service available")

        policy_info = self._training.get_policy(policy_id)
        if policy_info is None:
            raise RuntimeError(f"Policy not found: {policy_id}")

        self._policy_config = self._training.get_policy_config(policy_id)

        checkpoint_path = getattr(policy_info, "checkpoint_path", None)
        if not checkpoint_path:
            raise RuntimeError(
                f"Policy {policy_id} has no checkpoint path"
            )

        self._checkpoint_path = checkpoint_path

        # Load the policy model
        try:
            import json
            from pathlib import Path

            cp = Path(checkpoint_path)
            config_path = cp / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.json not found in {cp}"
                )

            with open(config_path) as f:
                _policy_cfg = json.load(f)

            policy_type = policy_info.policy_type
            from lerobot.policies.factory import get_policy_class

            policy_cls = get_policy_class(policy_type)
            self._policy = policy_cls.from_pretrained(str(cp))
            logger.info(
                "Policy loaded: %s (%s)", policy_id, policy_type
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load policy {policy_id}: {e}"
            ) from e

    def _apply_temporal_ensemble_override(self) -> None:
        """Attach or replace ACTTemporalEnsembler at deployment time.

        When config.temporal_ensemble_override is set and the loaded policy
        is ACT, we manually create the ensembler and wire it in.  This lets
        users experiment with TE at deploy time without retraining.

        Must be called BEFORE policy.reset() — reset() will call
        ensembler.reset() when temporal_ensemble_coeff is set.
        """
        if self._config is None or self._config.temporal_ensemble_override is None:
            return

        policy_type = ""
        if self._policy_config:
            policy_type = getattr(self._policy_config, "policy_type", "")

        if policy_type != "act":
            logger.warning(
                "temporal_ensemble_override is only supported for ACT policies "
                "(got policy_type=%r), ignoring",
                policy_type,
            )
            return

        if self._policy is None:
            return

        coeff = self._config.temporal_ensemble_override

        try:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        except ImportError:
            logger.error(
                "Cannot apply temporal_ensemble_override: "
                "lerobot.policies.act.modeling_act not importable"
            )
            return

        chunk_size = getattr(self._policy.config, "chunk_size", 100)
        self._policy.temporal_ensembler = ACTTemporalEnsembler(coeff, chunk_size)
        self._policy.config.temporal_ensemble_coeff = coeff
        self._policy.config.n_action_steps = 1

        logger.info(
            "Temporal ensemble override applied: coeff=%.4f, chunk_size=%d",
            coeff,
            chunk_size,
        )

    def _populate_safety_config(self, safety: SafetyConfig) -> None:
        """Fill joint_limits and motor_models from arm definitions.

        Tries YAML config first, then falls back to reading directly from
        the DamiaoFollowerRobot bus which has per-motor type and joint limits.
        """
        for arm_def in self._arm_defs:
            motor_model = arm_def.motor_type.value.upper()

            # Map motor type names to deployment velocity limit keys
            model_map = {
                "DAMIAO": "J8009P",    # Default to largest Damiao
                "STS3215": "STS3215",
                "DYNAMIXEL_XL330": "STS3215",  # Similar class
                "DYNAMIXEL_XL430": "STS3215",
            }
            model_key = model_map.get(motor_model, motor_model)

            # Get motor names from arm config if available
            motor_names = arm_def.config.get("motor_names", [])
            motor_models_cfg = arm_def.config.get("motor_models", {})

            for name in motor_names:
                # Per-motor model if available (e.g. Damiao arms have
                # different motor types per joint)
                per_motor_model = motor_models_cfg.get(name, model_key)
                safety.motor_models[f"{name}.pos"] = per_motor_model

            # Joint limits from calibration ranges in config
            joint_limits = arm_def.config.get("joint_limits", {})
            for name, limits in joint_limits.items():
                if isinstance(limits, (list, tuple)) and len(limits) == 2:
                    safety.joint_limits[f"{name}.pos"] = tuple(limits)

        # Fallback: read per-motor type and joint limits directly from the
        # DamiaoFollowerRobot bus.  The bus knows each motor's exact type
        # (J8009P, J4340P, J4310) and has calibrated joint limits.
        if not safety.motor_models and self._follower is not None:
            try:
                from lerobot.robots.damiao_follower.damiao_follower import (
                    DamiaoFollowerRobot,
                )

                if isinstance(self._follower, DamiaoFollowerRobot):
                    bus = self._follower.bus
                    for name, mcfg in bus._motor_configs.items():
                        key = f"{name}.pos"
                        safety.motor_models[key] = mcfg.motor_type
                    for name, (lo, hi) in bus._active_joint_limits.items():
                        key = f"{name}.pos"
                        safety.joint_limits[key] = (lo, hi)
                    logger.warning(
                        "DEPLOY: Populated safety config from Damiao bus: "
                        "%d motor models, %d joint limits — %s",
                        len(safety.motor_models),
                        len(safety.joint_limits),
                        {k: v for k, v in safety.motor_models.items()},
                    )
            except Exception as e:
                logger.warning(
                    "Failed to read safety config from robot bus: %s", e
                )

    def _start_recording(self, config: DeploymentConfig) -> None:
        """Start recording session for HIL/SERL modes."""
        if not self._teleop:
            return

        try:
            if hasattr(self._teleop, "start_recording_session"):
                repo_id = config.intervention_dataset or f"deployment_{config.policy_id}"
                task = config.task or "deployment"
                self._teleop.start_recording_session(
                    repo_id=repo_id,
                    task=task,
                    fps=config.loop_hz,
                )
                logger.info("Recording session started: %s", repo_id)
        except Exception as e:
            logger.warning("Failed to start recording session: %s", e)

    def _log_follower_positions(self, label: str) -> None:
        """Log follower joint positions at WARNING level for diagnostics.

        When label is "after pre-position", also stores a snapshot for
        drift detection at control loop start.
        """
        if self._follower is None:
            return
        try:
            obs = self._follower.get_observation()
            raw_positions = {
                k: v
                for k, v in obs.items()
                if isinstance(v, (int, float)) and k.endswith(".pos")
            }
            display = {k: f"{v:+.4f}" for k, v in raw_positions.items()}
            logger.warning("DEPLOY [%s] follower positions: %s", label, display)

            if label == "after pre-position":
                self._post_preposition_positions = dict(raw_positions)
        except Exception:
            pass

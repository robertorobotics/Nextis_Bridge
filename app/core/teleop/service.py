import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from lerobot.motors.feetech.feetech import OperatingMode

from app.core.config import load_config, save_config
from app.core.leader_assist import LeaderAssistService
from app.core.safety_layer import SafetyLayer
from app.core.teleop.pairing import PairingContext

logger = logging.getLogger(__name__)


class TeleoperationService:
    def __init__(
        self,
        robot: Any,
        leader: Any,
        robot_lock: threading.Lock,
        leader_assists: dict | None = None,
        arm_registry: Any | None = None,
        camera_service: Any | None = None,
        trigger_listener: Any | None = None,
    ) -> None:
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        self.arm_registry = arm_registry  # For pairing-based mapping
        self.camera_service = camera_service  # Standalone camera manager
        self.trigger_listener = trigger_listener  # Auto-start/stop with teleop

        self.safety = SafetyLayer(robot_lock) # Initialize Safety Layer

        # Initialize Leader Assist (Leader Only)
        # Passed from SystemState to share calibration state with API
        self.leader_assists = leader_assists if leader_assists else {}

        # Fallback local init if not passed (Legacy/Standalone)
        if self.leader and not self.leader_assists:
            # Detect BiUmbra
            if hasattr(self.leader, "left_arm") and hasattr(self.leader, "right_arm"):
                 logger.info("Initializing Leader Assist for Bi-Manual Leader (Local)")
                 self.leader_assists["left"] = LeaderAssistService(arm_id="left_leader")
                 self.leader_assists["right"] = LeaderAssistService(arm_id="right_leader")
            else:
                 # Mono or Generic
                 logger.info("Initializing Leader Assist for Single Leader (Local)")
                 self.leader_assists["default"] = LeaderAssistService(arm_id="leader")

        # Initialize Calibration Models for Followers (For Haptics & Transparency)
        self.follower_gravity_models = {}
        if self.robot:
            logger.info("Initializing Follower Gravity Models...")
            if hasattr(self.robot, "left_arm") and hasattr(self.robot, "right_arm"):
                 self.follower_gravity_models["left"] = LeaderAssistService(arm_id="left_follower")
                 self.follower_gravity_models["right"] = LeaderAssistService(arm_id="right_follower")
            else:
                 self.follower_gravity_models["default"] = LeaderAssistService(arm_id="follower")

        self.joint_names_template = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
        self.last_leader_pos = {} # Stores {full_joint_name: deg}
        self.last_loop_time = None

        # Velocity Smoothing (EMA)
        self.leader_vel_kf = {} # Stores {full_joint_name: last_filtered_vel}
        self.alpha_vel = 0.2     # Smoothing factor (0.2 = heavy smoothing, 0.8 = light)

        if self.leader and self.leader_assists:
             # Default to disabled as per user request
             self.assist_enabled = False
        else:
             self.assist_enabled = False

        self.is_running = False

        # Data storage for Graph
        self.max_history = 100
        self.history_lock = threading.Lock()
        self.action_history = deque(maxlen=self.max_history)

        # Optimization: Pre-computed mappings (legacy single-pair; kept for backward compat)
        self.joint_mapping = {} # {leader_key: follower_key}
        self.assist_groups = {} # {arm_key: [joint_name, ...]}
        self._leader_cal_ranges = {}  # {follower_key: (range_min, range_max)} from leader calibration
        self._active_leader = None  # Resolved in start() from arm registry or legacy
        self._active_robot = None   # Resolved in start() from arm registry or legacy

        # Multi-pair support: per-pairing contexts and threads
        self._pairing_contexts: list[PairingContext] = []
        self._teleop_threads: list[threading.Thread] = []

        # Load persisted force feedback settings (default True if not yet saved)
        _teleop_cfg = load_config().get("teleop", {})

        # Gripper Force Feedback (follower torque → leader current ceiling)
        self._force_feedback_enabled = _teleop_cfg.get("gripper_force_feedback", True)
        self._filtered_gripper_torque = 0.0   # EMA-filtered absolute torque (Nm)
        self._ff_alpha = 0.3                  # EMA smoothing (τ ≈ 55ms at 60Hz)
        self._ff_baseline_current = 60        # mA — light spring (perceptible, not fatiguing)
        self._ff_max_current = 1750           # mA — full XL330 range (must be <= Current_Limit)
        self._ff_torque_threshold = 0.2       # Nm — dead zone (friction/gravity noise)
        self._ff_torque_saturation = 2.0      # Nm — torque at which max current is reached

        # Joint Force Feedback: CURRENT_POSITION mode (same mechanism as gripper)
        # Goal_Position = follower position, Goal_Current = error magnitude
        self._joint_ff_enabled = _teleop_cfg.get("joint_force_feedback", False)
        self._joint_ff_k_spring = 4000.0    # mA/rad — gentle nudge (follower lag makes aggressive springs feel draggy)
        self._joint_ff_deadzone = 0.30      # rad (~17°) — must exceed follower tracking dead zone (0.15-0.25 rad)
        self._joint_ff_max_current = 800    # mA — ~46% of XL330 range (safety feel, not rigid constraint)
        self._joint_ff_min_force = 40       # mA — barely perceptible entry threshold

        # Teleop Configuration
        # Lowered to 60Hz to match lerobot default and reduce USB congestion
        self.frequency = 60
        self.dt = 1.0 / self.frequency

        # Recording State (Preserved from original)
        self.dataset = None
        self.dataset_config = None
        self.recording_active = False # Episode Level
        self.session_active = False   # Dataset Level
        self.episode_count = 0
        self.video_manager = None
        self.data_queue = deque(maxlen=1) # For UI data streaming if needed, though get_data uses history

        # Recording selections (which cameras/arms to record)
        self._selected_cameras = None   # None = all cameras
        self._selected_pairing_ids = None  # None = all followers (follower arm IDs)
        self._selected_arms = None      # Legacy prefix filter ("left", "right")

        # Recording frame rate control
        # Record at 30fps to match dataset fps, not teleop rate (60Hz)
        self.recording_fps = 30
        self._recording_frame_counter = 0
        self._recording_skip_frames = max(1, self.frequency // self.recording_fps)  # Skip every 2nd frame

        # Async frame writing queue
        self._frame_queue = deque()  # explicit backpressure in recording_capture_loop
        self._frame_writer_thread = None
        self._frame_writer_stop = threading.Event()

        # Shared action state for recording thread
        self._latest_leader_action = {}
        self._action_lock = threading.Lock()
        self._latest_follower_obs: dict = {}
        self._follower_obs_lock = threading.Lock()

        # Recording capture thread (separate from teleop loop)
        self._recording_capture_thread = None
        self._recording_stop_event = threading.Event()

        # Lock to prevent race between stop_episode and stop_session
        # Ensures save_episode() completes before finalize() is called
        self._episode_save_lock = threading.Lock()
        self._episode_saving = False  # Flag to track if save is in progress

        # Observation capture thread (for recording without teleop)
        self._obs_thread = None
        self._obs_stop_event = threading.Event()
        self._latest_obs = None
        self._latest_obs_lock = threading.Lock()
        self._obs_ready_event = threading.Event()

    # ── Assist / Force Feedback ──────────────────────────────────

    def set_assist_enabled(self, enabled: bool):
        self.assist_enabled = enabled
        logger.info(f"Leader Assist Enabled: {self.assist_enabled}")

        # Apply Hardware Change Immediately
        if self.is_running and self.leader and self.leader_assists:
            try:
                if self.assist_enabled:
                    logger.info("Enabling Leader Torque (PWM Mode)...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.left_arm.bus.enable_torque()
                         self.leader.right_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.right_arm.bus.enable_torque()
                    else:
                         self.leader.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.bus.enable_torque()
                else:
                    logger.info("Disabling Leader Torque...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.disable_torque()
                         self.leader.right_arm.bus.disable_torque()
                    else:
                         self.leader.bus.disable_torque()
            except Exception as e:
                logger.error(f"Failed to toggle Leader Assist State: {e}")

    def get_force_feedback_state(self) -> dict:
        """Return current force feedback toggle states."""
        return {
            "gripper": self._force_feedback_enabled,
            "joint": self._joint_ff_enabled,
        }

    def set_force_feedback(self, gripper: bool | None = None, joint: bool | None = None):
        """Toggle force feedback at runtime. When disabling, zero the current immediately."""
        if gripper is not None:
            self._force_feedback_enabled = gripper
            logger.info(f"Gripper force feedback: {'enabled' if gripper else 'disabled'}")
            leader = getattr(self, '_active_leader', None)
            if not gripper and self._has_damiao_follower and leader:
                try:
                    leader.bus.write(
                        "Goal_Current", "gripper", self._ff_baseline_current, normalize=False
                    )
                except Exception:
                    pass

        if joint is not None:
            self._joint_ff_enabled = joint
            logger.info(f"Joint force feedback: {'enabled' if joint else 'disabled'}")
            leader = getattr(self, '_active_leader', None)
            if not joint and leader:
                try:
                    leader.bus.write(
                        "Goal_Current", "joint_4", 0, normalize=False
                    )
                except Exception:
                    pass

        # Persist to settings.yaml
        self._save_force_feedback_config()

    def _save_force_feedback_config(self):
        """Persist force feedback toggles to settings.yaml."""
        try:
            config = load_config()
            config.setdefault("teleop", {})
            config["teleop"]["gripper_force_feedback"] = self._force_feedback_enabled
            config["teleop"]["joint_force_feedback"] = self._joint_ff_enabled
            save_config(config)
            logger.info("Force feedback config saved to settings.yaml")
        except Exception as e:
            logger.warning(f"Failed to save force feedback config: {e}")

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self, force: bool = False, active_arms: list[str] | None = None) -> None:
        from app.core.teleop import control_loop as _control
        from app.core.teleop import pairing as _pairing

        # Cancel any ongoing homing before starting new teleop
        if getattr(self, '_homing_thread', None) and self._homing_thread.is_alive():
            self._homing_cancel = True
            self._homing_thread.join(timeout=2.0)

        if self.is_running:
            logger.info("Teleop already running — skipping redundant start")
            return

        if not self.robot and not (self.arm_registry and active_arms):
             raise Exception("Robot not connected and no arm registry arms selected")

        # Store active arms (if provided, else None means All)
        self.active_arms = active_arms
        logger.info(f"Teleoperation Request: Active Arms = {self.active_arms}")

        # Validate selection if provided
        if self.active_arms is not None:
             leaders = []
             followers = []
             for a in self.active_arms:
                 if "leader" in a:
                     leaders.append(a)
                 elif "follower" in a:
                     followers.append(a)
                 elif self.arm_registry:
                     arm_info = self.arm_registry.get_arm(a)
                     if arm_info and arm_info.get("role") == "leader":
                         leaders.append(a)
                     elif arm_info and arm_info.get("role") == "follower":
                         followers.append(a)
             if not force and (not leaders or not followers):
                  logger.error(f"Selection Validation Failed: leaders={leaders}, followers={followers}, active_arms={self.active_arms}")
                  raise Exception("Invalid Selection: Must select at least one Leader and one Follower.")

        if not self.check_calibration():
             msg = "System not fully calibrated."
             if not force:
                 logger.warning(f"IGNORING CALIBRATION CHECK: {msg}")
             else:
                 logger.warning(f"FORCE START: {msg}")

        # Resolve active robot/leader from arm registry pairings
        # Build per-pairing contexts for independent teleop loops
        self._active_robot = self.robot  # default to legacy (used by recording)
        self._active_leader = self.leader  # default to legacy
        self._pairing_contexts = []
        self._teleop_threads = []
        print(f"[TELEOP] arm_registry={self.arm_registry is not None}, active_arms={self.active_arms}", flush=True)

        if self.arm_registry and self.active_arms:
            pairings = self.arm_registry.get_active_pairings(self.active_arms)
            print(f"[TELEOP] Found {len(pairings)} pairings for active_arms={self.active_arms}", flush=True)

            for pairing in pairings:
                leader_id = pairing['leader_id']
                follower_id = pairing['follower_id']
                print(f"[TELEOP] Resolving arm instances for pairing: {leader_id} → {follower_id}", flush=True)

                # Auto-connect if not already connected
                if leader_id not in self.arm_registry.arm_instances:
                    print(f"[TELEOP] Auto-connecting leader: {leader_id}", flush=True)
                    try:
                        result = self.arm_registry.connect_arm(leader_id)
                        print(f"[TELEOP] Leader connect result: {result}", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[TELEOP] Leader connect EXCEPTION: {e}", flush=True)
                        traceback.print_exc()
                        continue  # Skip this pairing if leader can't connect
                else:
                    print(f"[TELEOP] Leader {leader_id} already connected", flush=True)
                if follower_id not in self.arm_registry.arm_instances:
                    print(f"[TELEOP] Auto-connecting follower: {follower_id}", flush=True)
                    try:
                        result = self.arm_registry.connect_arm(follower_id)
                        print(f"[TELEOP] Follower connect result: {result}", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[TELEOP] Follower connect EXCEPTION: {e}", flush=True)
                        traceback.print_exc()
                        continue  # Skip this pairing if follower can't connect
                else:
                    print(f"[TELEOP] Follower {follower_id} already connected", flush=True)

                leader_inst = self.arm_registry.arm_instances.get(leader_id)
                follower_inst = self.arm_registry.arm_instances.get(follower_id)

                if leader_inst:
                    print(f"[TELEOP] Using arm-registry leader: {leader_id} ({type(leader_inst).__name__})", flush=True)
                else:
                    print(f"[TELEOP] WARNING: No instance for leader {leader_id}, skipping pairing", flush=True)
                    continue
                if follower_inst:
                    print(f"[TELEOP] Using arm-registry follower: {follower_id} ({type(follower_inst).__name__})", flush=True)
                else:
                    print(f"[TELEOP] WARNING: No instance for follower {follower_id}, skipping pairing", flush=True)
                    continue

                # Build isolated per-pairing context (prevents cross-contamination)
                ctx = _pairing.build_pairing_context(self, pairing, leader_inst, follower_inst)
                self._pairing_contexts.append(ctx)

                # Reload inversions per follower
                if hasattr(follower_inst, "reload_inversions"):
                    try:
                        follower_inst.reload_inversions()
                    except Exception as e:
                        logger.warning(f"Failed to reload inversions for {follower_id}: {e}")

                # Enable torque per follower
                self._enable_torque_for_robot(follower_inst)

            # Set _active_robot/_active_leader to first pair for recording/legacy compatibility
            if self._pairing_contexts:
                self._active_robot = self._pairing_contexts[0].active_robot
                self._active_leader = self._pairing_contexts[0].active_leader
                # Also set legacy joint_mapping/value_mode from first pair (for any code
                # that still reads self.joint_mapping directly)
                self.joint_mapping = self._pairing_contexts[0].joint_mapping
                self._follower_value_mode = self._pairing_contexts[0].follower_value_mode
                self._has_damiao_follower = self._pairing_contexts[0].has_damiao_follower
                self._leader_cal_ranges = self._pairing_contexts[0].leader_cal_ranges
        else:
            print("[TELEOP] No arm_registry or no active_arms — using legacy robot/leader", flush=True)
            # Legacy single-pair: build context from self.robot/self.leader
            _pairing.precompute_mappings(self)

        # Switch Leader to PWM Mode for Active Assist
        if self.leader and self.leader_assists:
            try:
                if self.assist_enabled:
                    logger.info("Switching Leader(s) to PWM Mode for Assist...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.left_arm.bus.enable_torque()
                         self.leader.right_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.right_arm.bus.enable_torque()
                    else:
                         self.leader.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.bus.enable_torque()
                else:
                    logger.info("Assist Disabled: Ensuring Leader Torque is OFF.")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.disable_torque()
                         self.leader.right_arm.bus.disable_torque()
                    else:
                         self.leader.bus.disable_torque()

            except Exception as e:
                logger.error(f"Failed to switch Leader Mode: {e}")

        # Startup blend config
        self._blend_duration = 0.5  # seconds — rate limiter provides additional smooth ramping
        self._filtered_gripper_torque = 0.0  # Reset force feedback filter

        self.is_running = True

        # Start per-pairing teleop loop threads
        if self._pairing_contexts:
            for ctx in self._pairing_contexts:
                t = threading.Thread(
                    target=_control.teleop_loop,
                    args=(self, ctx),
                    daemon=True,
                    name=f"teleop-{ctx.pairing_id}",
                )
                self._teleop_threads.append(t)
                t.start()
                print(f"[TELEOP] Started loop thread for {ctx.pairing_id}", flush=True)
        else:
            # Legacy single-pair fallback: build a context from self state
            ctx = PairingContext(
                pairing_id="legacy",
                active_leader=self._active_leader,
                active_robot=self._active_robot,
                joint_mapping=self.joint_mapping,
                follower_value_mode=getattr(self, '_follower_value_mode', 'int'),
                has_damiao_follower=getattr(self, '_has_damiao_follower', False),
                leader_cal_ranges=self._leader_cal_ranges,
            )
            self._pairing_contexts = [ctx]
            t = threading.Thread(target=_control.teleop_loop, args=(self, ctx), daemon=True, name="teleop-legacy")
            self._teleop_threads = [t]
            t.start()

        # Auto-start trigger listener if tools are configured
        if self.trigger_listener and not self.trigger_listener.is_running:
            tl_result = self.trigger_listener.start()
            if tl_result.get("success"):
                logger.info("Trigger listener auto-started with teleop")

    def _enable_torque_for_active_arms(self):
        """Legacy helper — delegates to _enable_torque_for_robot with self._active_robot."""
        active_robot = getattr(self, '_active_robot', None) or self.robot
        self._enable_torque_for_robot(active_robot)

    def _enable_torque_for_robot(self, active_robot):
        """Enable torque on a specific follower robot instance."""
        if not active_robot:
            print("[TELEOP] WARNING: No active robot for torque enable", flush=True)
            return

        # Skip MagicMock (fallback mock robot)
        from unittest.mock import MagicMock
        if isinstance(active_robot, MagicMock):
            print("[TELEOP] Skipping torque enable on MagicMock robot", flush=True)
            return

        try:
            print(f"[TELEOP] Enabling torque on {type(active_robot).__name__}...", flush=True)

            # For Damiao arms: re-configure ensures correct control mode is set.
            # Motors ARE enabled at end of configure() and ready for position commands.
            # MIT mode is used by default (stable), POS_VEL available via config.
            from lerobot.robots.damiao_follower.damiao_follower import DamiaoFollowerRobot
            if isinstance(active_robot, DamiaoFollowerRobot):
                mode_name = "MIT" if active_robot.bus.config.use_mit_mode else "POS_VEL"
                print(f"[TELEOP] Damiao detected — running configure() ({mode_name} mode)...", flush=True)
                active_robot.bus.configure()
                print(f"[TELEOP] Damiao configure() complete (motors enabled in {mode_name} mode)", flush=True)
            else:
                # Single-arm robot
                if hasattr(active_robot, "bus"):
                     active_robot.bus.enable_torque()
                # Dual-arm robot (e.g. BiUmbraFollower)
                if hasattr(active_robot, "left_arm"):
                     active_robot.left_arm.bus.enable_torque()
                if hasattr(active_robot, "right_arm"):
                     active_robot.right_arm.bus.enable_torque()

            print("[TELEOP] Torque enabled successfully", flush=True)

        except Exception as e:
            import traceback
            print(f"[TELEOP] Failed to enable torque: {e}", flush=True)
            traceback.print_exc()
            logger.error(f"Failed to enable torque: {e}")

    def stop(self) -> None:
        from app.core.teleop import homing as _homing
        from app.core.teleop import observation as _obs

        if not self.is_running:
            return
        logger.info("Stopping teleoperation...")
        self.is_running = False

        # Auto-stop trigger listener
        if self.trigger_listener and self.trigger_listener.is_running:
            try:
                self.trigger_listener.stop()
                logger.info("Trigger listener auto-stopped with teleop")
            except Exception as e:
                logger.warning(f"Failed to stop trigger listener: {e}")

        # Join all per-pairing teleop threads
        current = threading.current_thread()
        for t in self._teleop_threads:
            if t and t != current and t.is_alive():
                t.join(timeout=2.0)
        # Legacy fallback: join self.thread if it exists
        if hasattr(self, 'thread') and self.thread and self.thread != current:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass

        # Stop observation capture thread if running
        _obs.stop_obs_thread(self)

        # Stop Recording if active
        if self.session_active:
            try:
                from app.core.teleop import recording as _rec
                _rec.stop_recording_session(self)
            except Exception as e:
                logger.error(f"Failed to auto-stop recording session: {e}")

        # Switch Leader back to Position Mode (Safety)
        if self.leader:
            try:
                logger.info("Restoring Leader to Position Mode...")
                if "left" in self.leader_assists:
                     self.leader.left_arm.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.left_arm.bus.disable_torque()
                     self.leader.right_arm.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.right_arm.bus.disable_torque()
                else:
                     self.leader.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.bus.disable_torque()
            except Exception as e:
                logger.error(f"Failed to restore Leader Mode: {e}")

        # Per-pairing cleanup: reset force feedback and home each follower
        for ctx in self._pairing_contexts:
            # Reset force feedback on leader gripper
            if ctx.has_damiao_follower and ctx.active_leader:
                try:
                    ctx.active_leader.bus.write(
                        "Goal_Current", "gripper", self._ff_baseline_current, normalize=False
                    )
                except Exception as e:
                    logger.warning(f"[{ctx.pairing_id}] Failed to reset gripper Goal_Current: {e}")

                # Zero joint force feedback current (joint_4 → limp)
                if self._joint_ff_enabled:
                    try:
                        ctx.active_leader.bus.write(
                            "Goal_Current", "joint_4", 0, normalize=False
                        )
                    except Exception:
                        pass

            # Home follower arm (or disable immediately if no home position)
            if ctx.active_robot:
                home_pos = _homing.get_home_position(self, ctx.active_robot)
                if home_pos:
                    logger.info(f"[{ctx.pairing_id}] Homing follower to saved position ({len(home_pos)} joints)...")
                    self._homing_cancel = False
                    self._homing_thread = threading.Thread(
                        target=_homing.homing_loop,
                        args=(self, ctx.active_robot, home_pos),
                        daemon=True
                    )
                    self._homing_thread.start()
                else:
                    _homing.disable_follower_motors(ctx.active_robot)

        # Legacy fallback if no pairing contexts (shouldn't happen, but just in case)
        if not self._pairing_contexts:
            active_robot = getattr(self, '_active_robot', self.robot)
            if active_robot:
                home_pos = _homing.get_home_position(self, active_robot)
                if home_pos:
                    self._homing_cancel = False
                    self._homing_thread = threading.Thread(
                        target=_homing.homing_loop,
                        args=(self, active_robot, home_pos),
                        daemon=True
                    )
                    self._homing_thread.start()
                else:
                    _homing.disable_follower_motors(active_robot)

        # Reset active instances
        self._active_robot = self.robot
        self._active_leader = self.leader
        self._pairing_contexts = []
        self._teleop_threads = []

        logger.info("Teleoperation stopped.")

        if hasattr(self, '_debug_handler'):
             logger.removeHandler(self._debug_handler)
             self._debug_handler.close()

    # ── Delegating methods (public API preserved) ────────────────

    def check_calibration(self) -> bool:
        from app.core.teleop.homing import check_calibration
        return check_calibration(self)

    def get_data(self) -> dict:
        from app.core.teleop.observation import get_data
        return get_data(self)

    def _update_history(self, action_dict: dict) -> None:
        from app.core.teleop.observation import update_history
        return update_history(self, action_dict)

    def start_recording_session(
        self,
        repo_id: str,
        task: str,
        fps: int = 30,
        root: str | None = None,
        selected_cameras: list[str] | None = None,
        selected_pairing_ids: list[str] | None = None,
        selected_arms: list[str] | None = None,
        record_extended_state: bool = False,
        **kwargs,
    ) -> dict:
        from app.core.teleop.recording import start_recording_session
        return start_recording_session(
            self, repo_id, task, fps, root,
            selected_cameras, selected_pairing_ids, selected_arms,
            record_extended_state=record_extended_state,
            **kwargs,
        )

    def stop_recording_session(self) -> dict:
        from app.core.teleop.recording import stop_recording_session
        return stop_recording_session(self)

    def sync_to_disk(self) -> None:
        from app.core.teleop.recording import sync_to_disk
        return sync_to_disk(self)

    def refresh_metadata_from_disk(self) -> None:
        from app.core.teleop.recording import refresh_metadata_from_disk
        return refresh_metadata_from_disk(self)

    def start_episode(self) -> dict:
        from app.core.teleop.recording import start_episode
        return start_episode(self)

    def stop_episode(self) -> dict:
        from app.core.teleop.recording import stop_episode
        return stop_episode(self)

    def delete_last_episode(self) -> dict:
        from app.core.teleop.recording import delete_last_episode
        return delete_last_episode(self)

    def _filter_observation_features(
        self, obs_features,
        selected_cameras=None, selected_pairing_ids=None, selected_arms=None,
    ):
        from app.core.teleop.recording import filter_observation_features
        return filter_observation_features(
            obs_features, selected_cameras, selected_pairing_ids, selected_arms,
        )

    def _filter_action_features(
        self, action_features,
        selected_pairing_ids=None, selected_arms=None,
    ):
        from app.core.teleop.recording import filter_action_features
        return filter_action_features(action_features, selected_pairing_ids, selected_arms)

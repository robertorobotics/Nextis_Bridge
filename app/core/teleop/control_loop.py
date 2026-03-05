import logging
import time

import numpy as np
from lerobot.robots.damiao_follower.damiao_follower import map_range

from app.core.teleop.observation import update_history
from app.core.teleop.pairing import PairingContext

logger = logging.getLogger(__name__)

try:
    from lerobot.utils.robot_utils import precise_sleep
except ImportError:
    def precise_sleep(dt):
        time.sleep(max(0, dt))


def teleop_loop(svc, ctx: PairingContext):
    """Main teleop control loop for one leader→follower pairing.

    Each pairing runs in its own thread with fully isolated state via ctx,
    preventing cross-contamination between different arm types.
    """
    pid = ctx.pairing_id
    logger.info(f"[{pid}] Teleoperation Control Loop Running at {svc.frequency}Hz")
    print(f"[TELEOP] [{pid}] Control loop started at {svc.frequency}Hz", flush=True)

    # Start blend timer NOW — when the loop actually begins executing
    ctx.blend_start_time = time.time()

    loop_count = 0
    svc.last_loop_time = time.perf_counter()

    # Performance Monitoring
    perf_interval = svc.frequency # Log every 1s
    perf_start = time.time()

    try:
        while svc.is_running:
            loop_start = time.perf_counter()

            # 1. Read Leader State
            leader_action = {}
            # Capture Follower Obs (Needed for Recording)
            follower_obs = {}

            if ctx.active_leader:
                obs = None

                # Retry loop for transient serial communication errors on leader
                for attempt in range(3):
                    try:
                        obs = ctx.active_leader.get_action()
                        break  # Success
                    except (OSError, ConnectionError) as e:
                        error_str = str(e)
                        if "Incorrect status packet" in error_str or "Port is in use" in error_str:
                            if attempt < 2:
                                time.sleep(0.005)  # 5ms backoff
                                continue
                            else:
                                if loop_count % 60 == 0:
                                    logger.warning(f"Leader read failed after 3 attempts: {e}")
                        else:
                            logger.error(f"Leader read error: {e}")
                            break

                if not obs:
                    # Skip this loop iteration if leader read completely failed
                    continue

                # 1a. Leader Assist (Gravity/Transparency)
                if svc.leader_assists and svc.assist_enabled:
                     # Iterate pre-computed groups
                     for arm_key, arm_joint_names in svc.assist_groups.items():
                         service = svc.leader_assists[arm_key]

                         positions = []
                         velocities = []
                         valid = True

                         # Extract positions/velocities for this arm
                         for fullname in arm_joint_names:
                             pos_key = f"{fullname}.pos"
                             if pos_key in obs:
                                 deg = obs[pos_key]
                                 positions.append(deg)

                                 # Smooth Velocity Estimate (EMA)
                                 raw_vel = 0.0
                                 if fullname in svc.last_leader_pos:
                                     delta = deg - svc.last_leader_pos[fullname]
                                     # Handle wrapping? Usually robot driver handles this or returns absolute.
                                     # Assuming absolute degrees.
                                     raw_vel = delta / svc.dt

                                 # Apply EMA
                                 prev_vel = svc.leader_vel_kf.get(fullname, 0.0)
                                 filtered_vel = svc.alpha_vel * raw_vel + (1 - svc.alpha_vel) * prev_vel
                                 svc.leader_vel_kf[fullname] = filtered_vel

                                 velocities.append(filtered_vel)

                                 # Update Cache
                                 svc.last_leader_pos[fullname] = deg
                             else:
                                 valid = False
                                 break

                         if valid:
                             try:
                                 follower_loads = svc.safety.latest_loads
                                 haptic_forces = {}

                                 # Haptics: Compute External Force on Follower
                                 if arm_key in svc.follower_gravity_models:
                                      follower_model = svc.follower_gravity_models[arm_key]
                                      predicted_gravity = follower_model.predict_gravity(positions)

                                      for i, name in enumerate(arm_joint_names):
                                          # Heuristic mapping for load lookup
                                          # In pre-compute we could optimize this too, but it's okay for now
                                          follower_name = name.replace("leader", "follower")
                                          measured_load = float(follower_loads.get(follower_name, 0))
                                          expected_load = predicted_gravity[i]
                                          haptic_forces[name] = measured_load - expected_load

                                 # Compute PWM
                                 pwm_dict = service.compute_assist_torque(
                                     arm_joint_names, positions, velocities, follower_torques=haptic_forces
                                 )

                                 # Write PWM
                                 if pwm_dict:
                                     if arm_key == "left":
                                          local_pwm = {k.replace("left_", ""): v for k, v in pwm_dict.items()}
                                          svc.leader.left_arm.bus.write_pwm(local_pwm)
                                     elif arm_key == "right":
                                          local_pwm = {k.replace("right_", ""): v for k, v in pwm_dict.items()}
                                          svc.leader.right_arm.bus.write_pwm(local_pwm)
                                     else:
                                          svc.leader.bus.write_pwm(pwm_dict)
                             except Exception as e:
                                 logger.error(f"Assist Error: {e}") # Enable logging for debug

                # Debug: log on first frame to diagnose mapping issues
                if loop_count == 0:
                    print(f"[TELEOP DEBUG] [{pid}] _active_leader type: {type(ctx.active_leader).__name__}", flush=True)
                    print(f"[TELEOP DEBUG] [{pid}] _active_robot type: {type(ctx.active_robot).__name__}", flush=True)
                    print(f"[TELEOP DEBUG] [{pid}] joint_mapping ({len(ctx.joint_mapping)} entries): {ctx.joint_mapping}", flush=True)
                    print(f"[TELEOP DEBUG] obs keys from leader: {list(obs.keys()) if obs else 'None'}", flush=True)

                # 1b. Map to Follower Action (Optimized)
                # Use per-pairing mapping from ctx (prevents cross-contamination)
                for l_key, f_key in ctx.joint_mapping.items():
                    if l_key in obs:
                        val = obs[l_key]
                        if ctx.follower_value_mode == "float":
                            leader_action[f_key] = val  # Damiao: radians
                        elif ctx.follower_value_mode == "rad_to_percent":
                            if 'gripper' in f_key:
                                # Gripper: leader 0-1 → follower 0-100 (already absolute)
                                leader_action[f_key] = val * 100.0
                            elif f_key in ctx.leader_cal_ranges:
                                # Absolute mapping: leader radians → leader's ±100% → follower's ±100%
                                rmin, rmax = ctx.leader_cal_ranges[f_key]
                                leader_ticks = (val + np.pi) * 4096.0 / (2 * np.pi)
                                # Unwrap: if homed ticks crossed the 0/4096 encoder boundary,
                                # bring them back into the calibration range
                                center = (rmin + rmax) * 0.5
                                while leader_ticks < center - 2048:
                                    leader_ticks += 4096
                                while leader_ticks > center + 2048:
                                    leader_ticks -= 4096
                                leader_action[f_key] = ((leader_ticks - rmin) / (rmax - rmin)) * 200 - 100
                            else:
                                # Fallback: delta-based (no leader calibration available)
                                if l_key in ctx.leader_start_rad and f_key in ctx.follower_start_pos:
                                    delta = val - ctx.leader_start_rad[l_key]
                                    scale = ctx.rad_to_percent_scale.get(f_key, 100.0 / np.pi)
                                    leader_action[f_key] = ctx.follower_start_pos[f_key] + delta * scale
                                else:
                                    scale = ctx.rad_to_percent_scale.get(f_key, 100.0 / np.pi)
                                    leader_action[f_key] = val * scale
                        else:
                            leader_action[f_key] = int(val)  # Legacy Feetech→Feetech

                if loop_count == 0:
                    print(f"[TELEOP DEBUG] [{pid}] leader_action after mapping ({len(leader_action)} entries): {leader_action}", flush=True)

                # 1c. Startup blend: ramp from follower's current position
                if ctx.blend_start_time and leader_action:
                    # First frame: capture leader start for delta-based tracking (Dynamixel→Feetech)
                    if not ctx.leader_start_rad and ctx.follower_value_mode == "rad_to_percent" and obs:
                        ctx.leader_start_rad = {l_key: obs[l_key] for l_key in ctx.joint_mapping if l_key in obs}
                        print(f"[Teleop] [{pid}] Delta tracking: captured {len(ctx.leader_start_rad)} leader start positions", flush=True)

                    # First frame: capture follower's actual position
                    if not ctx.follower_start_pos and ctx.active_robot:
                        try:
                            fobs = ctx.active_robot.get_observation()
                            ctx.follower_start_pos = {
                                k: v for k, v in fobs.items() if k.endswith('.pos')
                            }
                            print(f"[Teleop] [{pid}] Startup blend: captured {len(ctx.follower_start_pos)} follower positions", flush=True)
                        except Exception as e:
                            print(f"[Teleop] [{pid}] Startup blend: get_observation() FAILED: {e}", flush=True)
                            # Fallback: use last known positions from bus (seeded during configure)
                            if hasattr(ctx.active_robot, 'bus') and hasattr(ctx.active_robot.bus, '_last_positions'):
                                lp = ctx.active_robot.bus._last_positions
                                ctx.follower_start_pos = {f"{k}.pos": v for k, v in lp.items()}
                                print(f"[Teleop] [{pid}] Startup blend: using bus fallback ({len(ctx.follower_start_pos)} joints)", flush=True)

                    # Compute per-joint rad→percent scale factors from follower calibration
                    if not ctx.rad_to_percent_scale and ctx.follower_value_mode == "rad_to_percent":
                        if hasattr(ctx.active_robot, 'calibration') and ctx.active_robot.calibration:
                            for l_key, f_key in ctx.joint_mapping.items():
                                motor_name = f_key.replace('.pos', '')
                                if 'gripper' in motor_name:
                                    continue  # Gripper uses its own scaling
                                cal = ctx.active_robot.calibration.get(motor_name)
                                if cal:
                                    tick_range = cal.range_max - cal.range_min
                                    if tick_range > 0:
                                        ctx.rad_to_percent_scale[f_key] = 4096.0 * 100.0 / (np.pi * tick_range)
                            print(f"[Teleop] [{pid}] Per-joint scales: { {k: f'{v:.1f}' for k, v in ctx.rad_to_percent_scale.items()} }", flush=True)

                    elapsed = time.time() - ctx.blend_start_time
                    alpha = min(1.0, elapsed / svc._blend_duration)

                    if alpha < 1.0 and ctx.follower_start_pos:
                        for key in list(leader_action.keys()):
                            if key in ctx.follower_start_pos:
                                start = ctx.follower_start_pos[key]
                                target = leader_action[key]
                                leader_action[key] = start + alpha * (target - start)

                        # First-frame diagnostic: log large position deltas
                        if loop_count == 0:
                            for key in list(leader_action.keys()):
                                if key in ctx.follower_start_pos:
                                    orig_target = leader_action.get(key, 0)  # Already blended
                                    follower_pos = ctx.follower_start_pos[key]
                                    raw_delta = orig_target - follower_pos
                                    if abs(raw_delta) > 0.01:
                                        print(f"[Teleop] [{pid}] Blend frame 0: {key} "
                                              f"follower={follower_pos:.3f}, blended={orig_target:.3f}, "
                                              f"delta={raw_delta:+.4f} rad", flush=True)

                # Debug: trace link3 position through teleop pipeline (first 5 frames)
                if loop_count < 5 and leader_action and "link3.pos" in leader_action:
                    _alpha = alpha if ctx.blend_start_time else -1.0
                    blend_active = (ctx.blend_start_time is not None and _alpha < 1.0)
                    fstart = ctx.follower_start_pos.get("link3.pos", "N/A") if ctx.follower_start_pos else "N/A"
                    print(f"[Teleop] Frame {loop_count}: link3.pos={leader_action['link3.pos']:.4f} "
                          f"(alpha={_alpha:.4f}, blend_active={blend_active}, "
                          f"follower_start={fstart})", flush=True)

                # 2. Send Action IMMEDIATELY (Low Latency Control)
                # Key insight from LeRobot: send_action and get_observation are independent.
                # Motor writes use sync_write (no response wait), cameras have background threads.
                # NO LOCK NEEDED - LeRobot's architecture handles thread safety at bus level.
                if leader_action and ctx.active_robot:
                    try:
                        ctx.active_robot.send_action(leader_action)
                        if loop_count == 0:
                            print(f"[TELEOP DEBUG] [{pid}] send_action SUCCESS, sent {len(leader_action)} values to {type(ctx.active_robot).__name__}", flush=True)
                    except Exception as e:
                        if loop_count % 60 == 0:
                            print(f"[TELEOP] Send Action Failed: {e}", flush=True)
                            logger.error(f"Send Action Failed: {e}")

                    # SAFETY: Check if CAN bus died (emergency shutdown triggered in driver)
                    if hasattr(ctx.active_robot, 'bus') and getattr(ctx.active_robot.bus, '_can_bus_dead', False):
                        print(f"[TELEOP] [{pid}] CAN BUS DEAD — stopping teleop for safety", flush=True)
                        logger.error(f"[TELEOP] [{pid}] CAN bus failure detected — emergency stop")
                        svc.is_running = False
                        break

                    # SAFETY: Check Damiao torque limits (every 6th frame = ~10Hz)
                    # read_torques() costs ~14ms (7 motors x 2ms), so throttle to avoid
                    # consuming too much of the 16.7ms frame budget at 60Hz.
                    # 3-violation debounce in SafetyLayer triggers e-stop within ~300ms.
                    if ctx.has_damiao_follower and loop_count % 6 == 3:
                        try:
                            if not svc.safety.check_damiao_limits(ctx.active_robot):
                                print(f"[TELEOP] [{pid}] SAFETY: Torque limit exceeded — stopping for homing", flush=True)
                                logger.error(f"[TELEOP] [{pid}] Damiao torque limit exceeded — graceful stop")
                                break  # stop() in finally sets is_running=False, does homing, disables
                        except Exception as e:
                            if loop_count % 60 == 0:
                                logger.warning(f"[TELEOP] Safety check error (non-fatal): {e}")

                    # 2a. Gripper Force Feedback (follower torque → leader current ceiling)
                    if (svc._force_feedback_enabled
                            and ctx.has_damiao_follower
                            and ctx.active_leader
                            and loop_count > 0):
                        try:
                            torques = ctx.active_robot.get_torques()
                            raw_torque = torques.get("gripper", 0.0)

                            # Use absolute value — direction doesn't matter for grip feel
                            torque_mag = abs(raw_torque)

                            # EMA filter
                            ctx.filtered_gripper_torque = (
                                svc._ff_alpha * torque_mag
                                + (1 - svc._ff_alpha) * ctx.filtered_gripper_torque
                            )

                            # Map to Goal_Current: dead zone → linear ramp → saturation
                            if ctx.filtered_gripper_torque <= svc._ff_torque_threshold:
                                goal_current = svc._ff_baseline_current
                            elif ctx.filtered_gripper_torque >= svc._ff_torque_saturation:
                                goal_current = svc._ff_max_current
                            else:
                                t = (ctx.filtered_gripper_torque - svc._ff_torque_threshold) / (
                                    svc._ff_torque_saturation - svc._ff_torque_threshold
                                )
                                goal_current = int(
                                    svc._ff_baseline_current
                                    + t * (svc._ff_max_current - svc._ff_baseline_current)
                                )

                            goal_current = max(svc._ff_baseline_current, min(svc._ff_max_current, goal_current))

                            ctx.active_leader.bus.write(
                                "Goal_Current", "gripper", goal_current, normalize=False
                            )

                            # Debug log once per second
                            if loop_count % 60 == 0:
                                print(
                                    f"[FORCE_FB] [{pid}] torque={raw_torque:.2f}Nm "
                                    f"filtered={ctx.filtered_gripper_torque:.2f}Nm "
                                    f"goal_current={goal_current}mA",
                                    flush=True,
                                )
                        except Exception as e:
                            if loop_count % 60 == 0:
                                logger.warning(f"[FORCE_FB] Error: {e}")

                        # 2a-ii. Joint force feedback: CURRENT_POSITION mode (virtual spring)
                        # Goal_Position = follower's actual position (spring target)
                        # Goal_Current = position error magnitude (how firmly to hold)
                        if svc._joint_ff_enabled:
                            try:
                                cached = ctx.active_robot.get_cached_positions()
                                follower_pos = cached.get("link3", None)
                                leader_pos = obs.get("joint_4.pos", None)

                                if follower_pos is not None and leader_pos is not None:
                                    pos_error = abs(leader_pos - follower_pos)

                                    # Goal_Current: how firmly the motor holds at Goal_Position
                                    if pos_error > svc._joint_ff_deadzone:
                                        excess = pos_error - svc._joint_ff_deadzone
                                        goal_current = min(
                                            int(max(svc._joint_ff_k_spring * excess, svc._joint_ff_min_force)),
                                            svc._joint_ff_max_current,
                                        )
                                    else:
                                        goal_current = 0  # Completely limp during normal tracking

                                    # Goal_Position: follower's actual position (spring target)
                                    # Convert radians → Dynamixel raw ticks
                                    homed_ticks = int((follower_pos + np.pi) / (2 * np.pi) * 4096)
                                    j4_id = ctx.active_leader.bus.motors["joint_4"].id
                                    raw_ticks = homed_ticks - ctx.active_leader.bus._software_homing_offsets.get(j4_id, 0)

                                    ctx.active_leader.bus.write(
                                        "Goal_Position", "joint_4", int(raw_ticks), normalize=False
                                    )
                                    ctx.active_leader.bus.write(
                                        "Goal_Current", "joint_4", goal_current, normalize=False
                                    )

                                    if loop_count % 60 == 0:
                                        print(
                                            f"[JOINT_FB] leader={leader_pos:.3f} follower={follower_pos:.3f} "
                                            f"error={pos_error:.3f}rad current={goal_current}mA",
                                            flush=True,
                                        )
                            except Exception as e:
                                if loop_count % 60 == 0:
                                    logger.warning(f"[JOINT_FB] Error: {e}")

                elif ctx.active_robot and loop_count % 60 == 0:
                    logger.warning(f"[{pid}] No Leader Action generated (Mapping issue or Empty Obs)")

                # 2b. Share motor positions with recording thread
                # IMPORTANT: Use FOLLOWER robot positions, not leader positions!
                # This ensures recorded data matches what HIL reads at inference time.
                # Leader and follower may have different calibrations, so same physical
                # position can give different encoder values.
                if ctx.active_robot:
                    if svc.recording_active:
                        if ctx.has_damiao_follower:
                            # Damiao: use MIT response cache (zero CAN overhead,
                            # no torque disruption from zero-torque probes).
                            cached = ctx.active_robot.get_cached_positions()
                            if cached:
                                follower_motors = {}
                                for k, v in cached.items():
                                    if k == "gripper":
                                        v = map_range(
                                            v,
                                            ctx.active_robot.gripper_open_pos,
                                            ctx.active_robot.gripper_closed_pos,
                                            0.0, 1.0,
                                        )
                                    follower_motors[f"{k}.pos"] = v
                                with svc._follower_obs_lock:
                                    svc._latest_follower_obs = follower_motors.copy()

                            # LEADER ACTION: the human's commanded intent (already in follower
                            # coordinate space after joint mapping in section 2a above)
                            if leader_action:
                                with svc._action_lock:
                                    svc._latest_leader_action = leader_action.copy()
                            elif cached:
                                # Fallback: if leader read failed this frame, use follower as
                                # approximate action (better than stale data)
                                with svc._action_lock:
                                    svc._latest_leader_action = follower_motors.copy()
                        else:
                            # Non-Damiao (Feetech/Dynamixel): sync_read is passive,
                            # safe to call get_observation() with retry.
                            follower_obs = None

                            for attempt in range(3):  # Up to 3 attempts
                                try:
                                    follower_obs = ctx.active_robot.get_observation()
                                    break  # Success, exit retry loop
                                except (OSError, ConnectionError) as e:
                                    error_str = str(e)
                                    if "Incorrect status packet" in error_str or "Port is in use" in error_str:
                                        if attempt < 2:  # Not the last attempt
                                            time.sleep(0.005)  # 5ms backoff
                                            continue
                                        else:
                                            if loop_count % 60 == 0:
                                                logger.warning(f"Motor read failed after 3 attempts: {e}")
                                    else:
                                        logger.error(f"Motor read error: {e}")
                                        break

                            if follower_obs:
                                follower_motors = {k: v for k, v in follower_obs.items() if '.pos' in k}
                                if follower_motors:
                                    with svc._follower_obs_lock:
                                        svc._latest_follower_obs = follower_motors.copy()

                            if leader_action:
                                with svc._action_lock:
                                    svc._latest_leader_action = leader_action.copy()
                            elif follower_obs:
                                follower_motors = {k: v for k, v in follower_obs.items() if '.pos' in k}
                                if follower_motors:
                                    with svc._action_lock:
                                        svc._latest_leader_action = follower_motors.copy()
                    elif leader_action:
                        # Not recording: use leader action directly as cached action
                        # (no need to read follower positions over CAN)
                        with svc._action_lock:
                            svc._latest_leader_action = leader_action.copy()

                # Debug heartbeat (disabled - too spammy)
                # if loop_count % 60 == 0:
                #     print(f"Teleop Heartbeat: {loop_count} (Active: {svc.is_running}, Recording: {svc.recording_active})")

                # 4. Store Data (for UI)
                if loop_count % 5 == 0:
                    update_history(svc, leader_action)

                # 5. Performance Logging
                loop_count += 1
                if loop_count % perf_interval == 0:
                     now = time.time()
                     real_hz = perf_interval / (now - perf_start)
                     logger.info(f"Teleop Loop Rate: {real_hz:.1f} Hz")
                     perf_start = now

                # 6. Sleep
                dt_s = time.perf_counter() - loop_start
                precise_sleep(svc.dt - dt_s)

        # Normal exit - log why we stopped
        logger.info(f"[{pid}] Teleop loop exited normally (is_running=False)")
        print(f"[TELEOP] [{pid}] Loop exited normally")

    except OSError as e:
        if e.errno == 5:
            logger.error(f"[{pid}] TELEOP STOPPED: Hardware Disconnected: {e}")
            print(f"[TELEOP ERROR] [{pid}] Hardware Disconnected: {e}")
        else:
            logger.error(f"[{pid}] TELEOP STOPPED: OSError {e.errno}: {e}")
            print(f"[TELEOP ERROR] [{pid}] OSError: {e}")
    except Exception as e:
        logger.error(f"[{pid}] TELEOP STOPPED: {e}")
        print(f"[TELEOP ERROR] [{pid}] Loop Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal all loops to stop (idempotent), then trigger cleanup
        logger.info(f"[{pid}] TELEOP CLEANUP: Calling stop()")
        print(f"[TELEOP] [{pid}] Cleanup - calling stop()")
        svc.stop()

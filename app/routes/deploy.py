from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.dependencies import get_state

router = APIRouter(tags=["deployment"])


@router.post("/deploy/start")
async def start_deployment(request: Request):
    """Start a policy deployment session."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        data = await request.json()

        policy_id = data.get("policy_id")
        active_arms = data.get("active_arms")

        if not policy_id:
            return JSONResponse(status_code=400, content={"error": "policy_id required"})
        if not active_arms or not isinstance(active_arms, list):
            return JSONResponse(status_code=400, content={"error": "active_arms required (list of arm IDs)"})

        # Build DeploymentConfig from request body
        from app.core.deployment import DeploymentConfig, DeploymentMode, SafetyConfig

        # Parse mode
        mode_str = data.get("mode", "inference")
        try:
            mode = DeploymentMode(mode_str)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": f"Invalid mode: {mode_str}"})

        # Parse safety config
        safety_data = data.get("safety", {})
        safety = SafetyConfig(
            speed_scale=float(safety_data.get("speed_scale", 1.0)),
            max_acceleration=float(safety_data.get("max_acceleration", 15.0)),
            smoothing_alpha=float(safety_data.get("smoothing_alpha", 0.3)),
            torque_check_interval=int(safety_data.get("torque_check_interval", 3)),
        )

        config = DeploymentConfig(
            mode=mode,
            policy_id=policy_id,
            safety=safety,
            movement_scale=float(data.get("movement_scale", 1.0)),
            intervention_dataset=data.get("intervention_dataset"),
            task=data.get("task"),
            reward_source=data.get("reward_source"),
            reward_model=data.get("reward_model"),
            max_episodes=data.get("max_episodes"),
            temporal_ensemble_override=(
                float(data["temporal_ensemble_override"])
                if data.get("temporal_ensemble_override") is not None
                else None
            ),
        )

        system.deployment_runtime.start(config, active_arms)
        return {"status": "started", "mode": mode.value, "policy_id": policy_id, "active_arms": active_arms}

    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/dry-run")
async def dry_run_deployment(request: Request):
    """Run a diagnostic dry-run: full pipeline for 30 frames without sending actions.

    Returns per-frame diagnostics and range-validation warnings as JSON.
    """
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        data = await request.json()

        policy_id = data.get("policy_id")
        active_arms = data.get("active_arms")

        if not policy_id:
            return JSONResponse(status_code=400, content={"error": "policy_id required"})
        if not active_arms or not isinstance(active_arms, list):
            return JSONResponse(status_code=400, content={"error": "active_arms required (list of arm IDs)"})

        from app.core.deployment import DeploymentConfig, DeploymentMode, SafetyConfig

        config = DeploymentConfig(
            mode=DeploymentMode.INFERENCE,
            policy_id=policy_id,
            safety=SafetyConfig(),
            dry_run=True,
        )

        system.deployment_runtime.start(config, active_arms)

        # Wait for the dry-run loop to finish (auto-stops after 30 frames)
        thread = system.deployment_runtime._loop_thread
        if thread and thread.is_alive():
            thread.join(timeout=10.0)

        # Capture logs before stop() clears them
        dry_run_log = list(system.deployment_runtime._dry_run_log)
        frame_count = system.deployment_runtime._frame_count

        system.deployment_runtime.stop()

        return {
            "status": "completed",
            "frames": frame_count,
            "diagnostics": dry_run_log,
        }

    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/stop")
def stop_deployment():
    """Stop the active deployment session."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        # Capture summary before stopping
        status = system.deployment_runtime.get_status()
        summary = {
            "frame_count": status.frame_count,
            "episode_count": status.episode_count,
        }

        system.deployment_runtime.stop()
        return {"status": "stopped", "summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/deploy/status")
def get_deployment_status():
    """Get current deployment status for frontend polling."""
    system = get_state()
    if not system.deployment_runtime:
        return {"state": "idle"}

    status = system.deployment_runtime.get_status()
    result = {
        "state": status.state.value,
        "mode": status.mode.value,
        "frame_count": status.frame_count,
        "episode_count": status.episode_count,
        "current_episode_frames": status.current_episode_frames,
        "safety": status.safety,
        "policy_config": status.policy_config,
        "rl_metrics": status.rl_metrics,
    }

    # Include active arms and their connection states
    result["active_arms"] = []
    arm_ids = getattr(system.deployment_runtime, "_active_arm_ids", [])
    if arm_ids and system.arm_registry:
        for arm_id in arm_ids:
            arm = system.arm_registry.get_arm(arm_id)
            if arm:
                result["active_arms"].append({
                    "id": arm_id,
                    "connected": system.arm_registry.arm_status.get(arm_id, "disconnected") == "connected",
                })

    return result


@router.patch("/deploy/settings")
async def update_deployment_settings(request: Request):
    """Live-adjust speed_scale and other settings during deployment."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        data = await request.json()

        if "speed_scale" in data:
            try:
                scale = float(data["speed_scale"])
                scale = max(0.1, min(1.0, scale))
                system.deployment_runtime.update_speed_scale(scale)
            except (ValueError, TypeError):
                pass

        # Return current settings
        status = system.deployment_runtime.get_status()
        return {
            "status": "updated",
            "speed_scale": status.safety.get("speed_scale", 1.0),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/episode/start")
def start_deployment_episode():
    """Start a new recording episode during deployment."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        if system.teleop_service and hasattr(system.teleop_service, "start_episode"):
            result = system.teleop_service.start_episode()
            system.deployment_runtime._episode_count += 1
            system.deployment_runtime._current_episode_frames = 0
            return result
        return JSONResponse(status_code=400, content={"error": "Recording not available"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/episode/stop")
def stop_deployment_episode():
    """Stop the current recording episode."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        if system.teleop_service and hasattr(system.teleop_service, "stop_episode"):
            return system.teleop_service.stop_episode()
        return JSONResponse(status_code=400, content={"error": "Recording not available"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/episode/next")
def next_deployment_episode():
    """Stop current episode and immediately start the next one."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        if not system.teleop_service or not hasattr(system.teleop_service, "stop_episode"):
            return JSONResponse(status_code=400, content={"error": "Recording not available"})

        stop_result = system.teleop_service.stop_episode()
        start_result = system.teleop_service.start_episode()
        system.deployment_runtime._episode_count += 1
        system.deployment_runtime._current_episode_frames = 0

        return {
            "status": "next_episode_started",
            "previous_episode": stop_result,
            "new_episode": start_result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/resume")
def resume_deployment():
    """Resume autonomous execution after intervention pause."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        success = system.deployment_runtime.resume()
        if success:
            return {"status": "resumed"}
        return JSONResponse(status_code=400, content={"error": "Cannot resume from current state"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/reset")
def reset_deployment():
    """Reset from ESTOP or ERROR back to IDLE.

    After reset, the user can start a new deployment from the setup screen.
    """
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        success = system.deployment_runtime.reset()
        if success:
            return {"status": "reset_to_idle"}
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Cannot reset from state: {system.deployment_runtime._state.value}. "
                "Reset is only available from ESTOP or ERROR states."
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/restart")
def restart_deployment():
    """Stop current deployment and restart with the same configuration.

    The robot will re-home to the leader position before policy
    execution resumes.
    """
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        system.deployment_runtime.restart()
        status = system.deployment_runtime.get_status()
        return {
            "status": "restarted",
            "state": status.state.value,
            "mode": status.mode.value,
        }
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/deploy/estop")
def estop_deployment():
    """Trigger emergency stop on the deployment runtime."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    try:
        system.deployment_runtime.estop()
        return {"status": "estop_triggered"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/deploy/safety")
def get_deployment_safety():
    """Get detailed per-motor safety readings."""
    system = get_state()
    if not system.deployment_runtime:
        return {"estop_active": False, "per_motor_velocity": {}, "per_motor_torque": {}, "active_clamps": {}, "speed_scale": 1.0}

    pipeline = system.deployment_runtime._safety_pipeline
    if not pipeline:
        return {"estop_active": False, "per_motor_velocity": {}, "per_motor_torque": {}, "active_clamps": {}, "speed_scale": 1.0}

    readings = pipeline.get_readings()
    return {
        "per_motor_velocity": readings.per_motor_velocity,
        "per_motor_torque": readings.per_motor_torque,
        "active_clamps": readings.active_clamps,
        "estop_active": readings.estop_active,
        "speed_scale": readings.speed_scale,
    }


@router.post("/deploy/retrain")
async def trigger_deployment_retrain(request: Request):
    """Trigger DAgger fine-tune on intervention data."""
    system = get_state()
    if not system.deployment_runtime:
        return JSONResponse(status_code=503, content={"error": "Deployment runtime not initialized"})

    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized (needed for retraining)"})

    try:
        data = await request.json()
    except Exception:
        data = {}

    try:
        return system.hil_service.trigger_retrain(config=data.get("config"))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

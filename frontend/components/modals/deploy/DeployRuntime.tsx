import React, { useState, useEffect } from "react";
import {
  Play,
  StopCircle,
  User,
  Bot,
  RefreshCw,
  Loader2,
  SkipForward,
  Pause,
  ShieldAlert,
  Zap,
  Lock,
  Activity,
  OctagonX,
} from "lucide-react";
import CameraFeed from "../../ui/CameraFeed";
import { usePolling } from "../../../hooks/usePolling";
import { deployApi } from "../../../lib/api";
import type { DeploymentStatus, SafetyReadings, CameraConfig } from "../../../lib/api/types";

interface DeployRuntimeProps {
  status: DeploymentStatus | null;
  activeCameras: CameraConfig[];
  speak: (text: string) => void;
  onStopped: () => void;
}

const STATE_OVERLAY: Record<string, { bg: string; label: string }> = {
  running: { bg: "bg-emerald-500/90", label: "Autonomous" },
  human_active: { bg: "bg-blue-500/90", label: "Human Control" },
  paused: { bg: "bg-yellow-500/90", label: "Paused" },
  estop: { bg: "bg-red-500/90", label: "E-STOP" },
  starting: { bg: "bg-blue-400/90", label: "Starting..." },
  stopping: { bg: "bg-neutral-500/90", label: "Stopping..." },
};

export default function DeployRuntime({ status, activeCameras, speak, onStopped }: DeployRuntimeProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [safety, setSafety] = useState<SafetyReadings | null>(null);
  const [liveSpeedScale, setLiveSpeedScale] = useState<number>(
    (status?.safety as Record<string, number>)?.speed_scale ?? 1.0
  );

  const isHilMode = status?.mode === "hil" || status?.mode === "hil_serl";
  const isRlMode = status?.mode === "hil_serl";

  // Poll safety readings
  usePolling(async () => {
    try {
      const data = await deployApi.safety();
      setSafety(data);
    } catch (e) {
      console.error(e);
    }
  }, 300, !!status && status.state !== "idle");

  // Sync speed_scale from safety readings
  useEffect(() => {
    if (safety) setLiveSpeedScale(safety.speed_scale);
  }, [safety?.speed_scale]);

  const updateSpeedScale = async (value: number) => {
    setLiveSpeedScale(value);
    try {
      await deployApi.settings({ speed_scale: value });
    } catch (e) {
      console.error(e);
    }
  };

  // Episode controls
  const toggleEpisode = async () => {
    setIsProcessing(true);
    try {
      if (status?.current_episode_frames && status.current_episode_frames > 0) {
        await deployApi.stopEpisode();
      } else {
        await deployApi.startEpisode();
        speak(`Starting episode ${(status?.episode_count || 0) + 1}`);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const nextEpisode = async () => {
    setIsProcessing(true);
    try {
      await deployApi.nextEpisode();
      speak("Next episode");
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  const resumeAutonomous = async () => {
    setIsProcessing(true);
    try {
      await deployApi.resume();
      speak("Resuming autonomous mode");
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  const stopEpisode = async () => {
    setIsProcessing(true);
    try {
      await deployApi.stopEpisode();
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  const stopDeployment = async () => {
    if (!confirm("Stop deployment? This will save all recorded data.")) return;
    try {
      await deployApi.stop();
      onStopped();
    } catch (e) {
      console.error(e);
    }
  };

  const triggerEstop = async () => {
    try {
      await deployApi.estop();
    } catch (e) {
      console.error(e);
    }
  };

  const resetFromEstop = async () => {
    setIsProcessing(true);
    try {
      await deployApi.reset();
      speak("Reset complete. Ready for new deployment.");
      onStopped();
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  const restartDeployment = async () => {
    if (!confirm("Restart deployment? The robot will re-home to the start position.")) return;
    setIsProcessing(true);
    try {
      await deployApi.restart();
      speak("Restarting deployment");
    } catch (e: unknown) {
      console.error(e);
      alert(`Restart failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const triggerRetrain = async () => {
    if (!confirm("Start retraining on intervention data?")) return;
    try {
      await deployApi.retrain({});
    } catch (e: unknown) {
      alert(`Retrain failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    }
  };

  // Clamp counts from safety
  const clampCounts = safety?.active_clamps || {};
  const totalClamps = Object.values(clampCounts).reduce((a, b) => a + b, 0);

  // State overlay info
  const overlay = STATE_OVERLAY[status?.state || ""] || { bg: "bg-neutral-500/90", label: "Idle" };
  const episodeActive = status?.current_episode_frames != null && status.current_episode_frames > 0;

  const speedColor =
    liveSpeedScale >= 0.8 ? "text-red-500" : liveSpeedScale >= 0.5 ? "text-amber-500" : "text-green-500";

  return (
    <div className="flex h-full gap-5">
      {/* Left: Camera Feeds */}
      <div className="flex-1 flex flex-col gap-4">
        <div className="flex-1 bg-black rounded-2xl overflow-hidden relative min-h-0">
          <div
            className="grid h-full gap-0.5"
            style={{
              gridTemplateColumns: `repeat(auto-fill, minmax(${activeCameras.length <= 1 ? "100%" : "300px"}, 1fr))`,
            }}
          >
            {activeCameras.map((cam) => (
              <div key={cam.id} className="relative w-full h-full overflow-hidden">
                <CameraFeed cameraId={cam.id} mode="contain" className="rounded-none border-0" />
              </div>
            ))}
            {activeCameras.length === 0 && (
              <div className="flex items-center justify-center text-white/50">
                No cameras configured for this policy
              </div>
            )}
          </div>

          {/* Mode overlay */}
          <div className={`absolute top-4 right-4 flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold shadow-lg backdrop-blur-md text-white transition-all ${overlay.bg}`}>
            {status?.state === "human_active" ? <User className="w-4 h-4" /> :
             status?.state === "paused" ? <Pause className="w-4 h-4" /> :
             status?.state === "estop" ? <ShieldAlert className="w-4 h-4" /> :
             <Bot className="w-4 h-4" />}
            {overlay.label}
          </div>

          {/* Recording indicator */}
          {episodeActive && (
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/90 text-white px-3 py-1.5 rounded-full text-xs font-bold">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              REC
            </div>
          )}

          {/* Paused action panel */}
          {status?.state === "paused" && (
            <>
              <div className="absolute inset-0 border-4 border-yellow-400 rounded-2xl animate-pulse pointer-events-none" />
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3 p-4 bg-yellow-50/95 dark:bg-yellow-950/95 border border-yellow-300 dark:border-yellow-700 rounded-xl shadow-xl backdrop-blur-sm">
                <button
                  onClick={resumeAutonomous}
                  disabled={isProcessing}
                  className="px-5 py-2.5 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                >
                  <Play className="w-4 h-4" /> Resume Autonomous
                </button>
                <button
                  onClick={stopEpisode}
                  disabled={isProcessing}
                  className="px-5 py-2.5 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                >
                  <StopCircle className="w-4 h-4" /> Stop Episode
                </button>
              </div>
            </>
          )}

          {/* E-STOP / ERROR recovery panel */}
          {(status?.state === "estop" || status?.state === "error") && (
            <>
              <div className="absolute inset-0 border-4 border-red-500 rounded-2xl pointer-events-none" />
              <div className="absolute inset-0 bg-red-950/30 rounded-2xl pointer-events-none" />
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 p-5 bg-red-50/95 dark:bg-red-950/95 border border-red-300 dark:border-red-700 rounded-xl shadow-xl backdrop-blur-sm">
                <p className="text-sm font-semibold text-red-700 dark:text-red-300">
                  {status?.state === "estop"
                    ? "Emergency stop activated. Verify the robot is safe before resetting."
                    : "An error occurred. Check logs, then reset to try again."}
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={resetFromEstop}
                    disabled={isProcessing}
                    className="px-5 py-2.5 bg-amber-500 text-white rounded-lg font-semibold hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                  >
                    <RefreshCw className="w-4 h-4" /> Reset to Setup
                  </button>
                  <button
                    onClick={stopDeployment}
                    disabled={isProcessing}
                    className="px-5 py-2.5 bg-neutral-500 text-white rounded-lg font-semibold hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                  >
                    <StopCircle className="w-4 h-4" /> Stop & Close
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Episode Control Bar (HIL/SERL only) */}
        {isHilMode && (
          <div className="h-20 bg-white dark:bg-zinc-900 border border-neutral-100 dark:border-zinc-800 rounded-2xl shadow-sm flex items-center justify-between px-6 flex-shrink-0">
            <div>
              <span className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase">Episode</span>
              <span className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 block">
                #{(status?.episode_count || 0) + (episodeActive ? 1 : 0)}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={toggleEpisode}
                disabled={isProcessing || status?.state === "human_active" || status?.state === "paused"}
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-all shadow-lg ${
                  isProcessing || status?.state === "human_active" || status?.state === "paused"
                    ? "opacity-50 cursor-not-allowed"
                    : episodeActive
                    ? "bg-white dark:bg-zinc-800 border-4 border-red-500 hover:scale-105"
                    : "bg-red-500 border-4 border-red-100 dark:border-red-900 hover:scale-105"
                }`}
              >
                {isProcessing ? (
                  <Loader2 className="w-6 h-6 text-neutral-400 animate-spin" />
                ) : episodeActive ? (
                  <div className="w-6 h-6 bg-red-500 rounded-sm" />
                ) : (
                  <Play className="w-6 h-6 text-white ml-1" />
                )}
              </button>

              {episodeActive && status?.state !== "paused" && (
                <button
                  onClick={nextEpisode}
                  disabled={isProcessing || status?.state === "human_active"}
                  className={`w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md ${
                    isProcessing || status?.state === "human_active"
                      ? "bg-blue-300 cursor-not-allowed"
                      : "bg-blue-500 hover:bg-blue-600 hover:scale-105"
                  }`}
                >
                  {isProcessing ? (
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                  ) : (
                    <SkipForward className="w-5 h-5 text-white" />
                  )}
                </button>
              )}
            </div>

            <button
              onClick={stopDeployment}
              disabled={isProcessing}
              className="px-4 py-2 rounded-lg text-sm font-medium text-neutral-500 dark:text-zinc-400 hover:bg-neutral-100 dark:hover:bg-zinc-800 transition-colors"
            >
              Stop
            </button>
          </div>
        )}
      </div>

      {/* Right: Sidebar */}
      <div className="w-60 flex flex-col gap-3 flex-shrink-0 overflow-y-auto">
        {/* Live Safety Panel */}
        <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-100 dark:border-zinc-700 space-y-3">
          <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Safety</h3>

          {/* Live speed slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-neutral-500 dark:text-zinc-400">Speed</span>
              <span className={`text-xs font-bold ${speedColor}`}>
                {Math.round(liveSpeedScale * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={liveSpeedScale}
              onChange={(e) => updateSpeedScale(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
              style={{
                background:
                  "linear-gradient(to right, #22c55e 0%, #22c55e 40%, #eab308 50%, #eab308 70%, #ef4444 80%, #ef4444 100%)",
              }}
            />
          </div>

          {/* Clamp counts */}
          {totalClamps > 0 && (
            <div className="space-y-1">
              {Object.entries(clampCounts).map(([type, count]) =>
                count > 0 ? (
                  <div key={type} className="flex items-center gap-1.5 text-xs">
                    {type.includes("velocity") ? <Zap className="w-3 h-3 text-amber-500" /> : <Lock className="w-3 h-3 text-red-500" />}
                    <span className="text-neutral-600 dark:text-zinc-400">{count} {type.replace(/_/g, " ")}</span>
                  </div>
                ) : null
              )}
            </div>
          )}

          {/* Peak readings */}
          {safety && Object.keys(safety.per_motor_velocity).length > 0 && (
            <div className="text-xs text-neutral-400 dark:text-zinc-500 space-y-0.5">
              <div className="flex justify-between">
                <span>Peak velocity</span>
                <span className="font-mono">
                  {Math.max(...Object.values(safety.per_motor_velocity), 0).toFixed(1)} rad/s
                </span>
              </div>
              <div className="flex justify-between">
                <span>Peak torque</span>
                <span className="font-mono">
                  {Math.max(...Object.values(safety.per_motor_torque), 0).toFixed(1)} Nm
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Metrics */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-neutral-100 dark:border-zinc-800 shadow-sm">
          <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest mb-2">Metrics</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-neutral-500 dark:text-zinc-400 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-500" /> Frames
              </span>
              <span className="font-mono font-bold text-neutral-900 dark:text-zinc-100">{status?.frame_count || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500 dark:text-zinc-400">Episodes</span>
              <span className="font-mono font-bold text-neutral-900 dark:text-zinc-100">{status?.episode_count || 0}</span>
            </div>
          </div>
        </div>

        {/* RL Metrics (SERL only) */}
        {isRlMode && status?.rl_metrics && (
          <div className="bg-orange-50 dark:bg-orange-950/30 rounded-xl p-4 border border-orange-200 dark:border-orange-800">
            <h3 className="text-xs font-bold text-orange-400 uppercase tracking-widest mb-2">RL Metrics</h3>
            <div className="space-y-1.5 text-xs">
              {status.rl_metrics.avg_reward != null && (
                <div className="flex justify-between">
                  <span className="text-orange-600 dark:text-orange-400">Avg Reward</span>
                  <span className="font-mono font-bold text-orange-700 dark:text-orange-300">
                    {status.rl_metrics.avg_reward.toFixed(2)}
                  </span>
                </div>
              )}
              {status.rl_metrics.online_buffer_size != null && (
                <div className="flex justify-between">
                  <span className="text-orange-600 dark:text-orange-400">Buffer</span>
                  <span className="font-mono font-bold text-orange-700 dark:text-orange-300">
                    {status.rl_metrics.online_buffer_size.toLocaleString()}
                  </span>
                </div>
              )}
              {status.rl_metrics.training_step != null && (
                <div className="flex justify-between">
                  <span className="text-orange-600 dark:text-orange-400">Train Step</span>
                  <span className="font-mono font-bold text-orange-700 dark:text-orange-300">
                    {status.rl_metrics.training_step.toLocaleString()}
                  </span>
                </div>
              )}
              {status.rl_metrics.intervention_rate != null && (
                <div className="flex justify-between">
                  <span className="text-orange-600 dark:text-orange-400">Intervention</span>
                  <span className="font-mono font-bold text-orange-700 dark:text-orange-300">
                    {(status.rl_metrics.intervention_rate * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Retrain button (HIL/SERL only) */}
        {isHilMode && (
          <button
            onClick={triggerRetrain}
            className="w-full py-2.5 bg-purple-600 text-white rounded-xl font-semibold text-sm hover:bg-purple-700 transition-all flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" /> Retrain on Data
          </button>
        )}

        {/* Policy config */}
        {status?.policy_config && (
          <div className="bg-purple-50 dark:bg-purple-950/30 rounded-xl p-3 border border-purple-100 dark:border-purple-900">
            <h3 className="text-xs font-bold text-purple-400 uppercase tracking-widest mb-1.5">Policy</h3>
            <div className="space-y-0.5 text-xs text-purple-700 dark:text-purple-300">
              {status.policy_config.type && <p><strong>Type:</strong> {status.policy_config.type}</p>}
              <p><strong>Cameras:</strong> {status.policy_config.cameras?.join(", ") || "All"}</p>
              <p><strong>Arms:</strong> {status.policy_config.arms?.join(", ") || "All"}</p>
            </div>
          </div>
        )}

        {/* Bottom buttons (when not HIL — inference mode has no episode bar) */}
        <div className="mt-auto space-y-2">
          {!isHilMode && status?.state === "running" && (
            <button
              onClick={restartDeployment}
              disabled={isProcessing}
              className="w-full py-2.5 rounded-xl font-semibold text-sm text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-950/50 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className="w-4 h-4" /> Restart Episode
            </button>
          )}
          {!isHilMode && (
            <button
              onClick={stopDeployment}
              className="w-full py-2.5 rounded-xl font-semibold text-sm text-neutral-700 dark:text-zinc-300 bg-neutral-100 dark:bg-zinc-800 hover:bg-neutral-200 dark:hover:bg-zinc-700 transition-all"
            >
              Stop Deployment
            </button>
          )}
          <button
            onClick={triggerEstop}
            className="w-full py-3 bg-red-600 text-white rounded-xl font-bold text-sm hover:bg-red-700 transition-all flex items-center justify-center gap-2 shadow-lg shadow-red-200 dark:shadow-red-900/30"
          >
            <OctagonX className="w-4 h-4" /> E-STOP
          </button>
        </div>
      </div>
    </div>
  );
}

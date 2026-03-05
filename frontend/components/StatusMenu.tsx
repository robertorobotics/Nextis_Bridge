"use client";

import { useState } from "react";
import { RefreshCw, Power, Loader2 } from "lucide-react";
import { API_BASE, systemApi, armsApi } from "../lib/api";
import { usePolling } from "../hooks/usePolling";

interface StatusMenuProps {
  onOpenArmManager: () => void;
}

export default function StatusMenu({ onOpenArmManager }: StatusMenuProps) {
  // Status State
  const [statusText, setStatusText] = useState("OFFLINE");
  const [connectionState, setConnectionState] = useState("DISCONNECTED");
  const [executionState, setExecutionState] = useState("IDLE");
  const [isRestarting, setIsRestarting] = useState(false);

  // UI States
  const [isStatusMenuOpen, setIsStatusMenuOpen] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [restartConfirm, setRestartConfirm] = useState(false);

  // Damiao Velocity Limiter State
  const [hasDamiao, setHasDamiao] = useState(false);
  const [velocityLimit, setVelocityLimit] = useState(0.1);
  const [isUpdatingVelocity, setIsUpdatingVelocity] = useState(false);

  // Arm status for quick view
  const [armsSummary, setArmsSummary] = useState<{
    total_arms: number;
    connected: number;
  }>({ total_arms: 0, connected: 0 });

  // Status polling
  usePolling(async () => {
    const data = await systemApi.status();
    if (data) {
      if (data.connection === "CONNECTED") {
        if (isRestarting) setIsRestarting(false);
        if (isReconnecting) setIsReconnecting(false);
      }
      setStatusText(data.status);
      setConnectionState(data.connection);
      setExecutionState(data.execution);
    } else {
      setConnectionState("DISCONNECTED");
      setStatusText("OFFLINE");
    }
  }, 1000, true);

  // Damiao velocity limit polling
  usePolling(() => {
    fetch(`${API_BASE}/robot/velocity-limit`)
      .then((res) => res.json())
      .then((data) => {
        setHasDamiao(data.has_velocity_limit || false);
        if (
          data.has_velocity_limit &&
          typeof data.velocity_limit === "number" &&
          !isNaN(data.velocity_limit)
        ) {
          setVelocityLimit(data.velocity_limit);
        }
      })
      .catch(() => {
        setHasDamiao(false);
      });
  }, 5000, true);

  // Arm registry summary polling
  usePolling(async () => {
    try {
      const data = await armsApi.list();
      if (data.summary) {
        setArmsSummary(data.summary);
      }
    } catch { /* ignore */ }
  }, 5000, true);

  const updateVelocityLimit = async (newLimit: number) => {
    setIsUpdatingVelocity(true);
    try {
      const res = await fetch(`${API_BASE}/robot/velocity-limit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: newLimit }),
      });
      if (res.ok) {
        setVelocityLimit(newLimit);
      } else {
        const err = await res.json().catch(() => ({}));
        console.warn("Velocity limit update rejected:", res.status, err);
      }
    } catch (e) {
      console.warn(
        "Velocity limit update failed (backend unreachable?):",
        e
      );
    } finally {
      setIsUpdatingVelocity(false);
    }
  };

  const getStatusColor = () => {
    if (isRestarting)
      return "bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]";

    switch (connectionState) {
      case "DISCONNECTED":
      case "ERROR":
        return "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]";
      case "INITIALIZING":
        return "bg-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.6)]";
      case "MOCK":
        return "bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.6)]";
      case "CONNECTED":
        if (executionState !== "IDLE")
          return "bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]";
        return "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]";
      default:
        return "bg-gray-400";
    }
  };

  const currentStatusText = isRestarting ? "RESTARTING..." : statusText;

  return (
    <div className="relative">
      <button
        onClick={() => {
          setIsStatusMenuOpen(!isStatusMenuOpen);
          setRestartConfirm(false);
        }}
        className={`px-4 py-1.5 flex items-center gap-2 rounded-full transition-all border border-transparent ${isStatusMenuOpen ? "bg-white dark:bg-zinc-800 border-black/5 dark:border-zinc-700 shadow-md" : "hover:bg-black/5 dark:hover:bg-white/5"}`}
      >
        <div
          className={`w-2.5 h-2.5 rounded-full ${getStatusColor()} transition-colors duration-500`}
        />
        <span className="text-xs font-medium text-neutral-500 dark:text-zinc-400 uppercase tracking-wide min-w-[50px] text-left">
          {currentStatusText}
        </span>
      </button>

      {/* STATUS MENU POPOVER */}
      {isStatusMenuOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsStatusMenuOpen(false)}
          />

          <div className="absolute top-12 right-0 w-64 max-w-[calc(100vw-2rem)] bg-white/90 dark:bg-zinc-900/90 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 shadow-2xl rounded-2xl p-4 z-50 animate-in fade-in slide-in-from-top-2 flex flex-col gap-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">
                System Control
              </span>
              <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
            </div>

            {/* Connection Details */}
            <div className="bg-neutral-50 dark:bg-zinc-800 rounded-lg p-3 border border-neutral-100 dark:border-zinc-700 mb-2">
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-neutral-500 dark:text-zinc-400">
                  Connection
                </span>
                <span className="font-medium text-neutral-800 dark:text-zinc-200">
                  {connectionState}
                </span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-neutral-500 dark:text-zinc-400">
                  Execution
                </span>
                <span className="font-medium text-neutral-800 dark:text-zinc-200">
                  {executionState}
                </span>
              </div>
            </div>

            {/* Damiao Velocity Limiter */}
            {hasDamiao && (
              <div className="bg-orange-50 dark:bg-orange-950/30 rounded-lg p-3 border border-orange-200 dark:border-orange-900/50 mb-2">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-orange-800 dark:text-orange-300">
                      Damiao Velocity
                    </span>
                    <span className="px-1.5 py-0.5 text-[9px] font-bold bg-orange-200 dark:bg-orange-900/50 text-orange-700 dark:text-orange-400 rounded uppercase">
                      Safety
                    </span>
                  </div>
                  <span className="text-sm font-bold text-orange-600 dark:text-orange-400">
                    {Math.round(
                      (isNaN(velocityLimit) ? 0.1 : velocityLimit) * 100
                    )}
                    %
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={isNaN(velocityLimit) ? 10 : velocityLimit * 100}
                  onChange={(e) => {
                    const newVal = Number(e.target.value) / 100;
                    setVelocityLimit(newVal);
                  }}
                  onMouseUp={() => {
                    updateVelocityLimit(velocityLimit);
                  }}
                  onTouchEnd={() => {
                    updateVelocityLimit(velocityLimit);
                  }}
                  disabled={isUpdatingVelocity}
                  className="w-full h-2 bg-orange-200 dark:bg-orange-900/50 rounded-full appearance-none cursor-pointer accent-orange-500 disabled:opacity-50"
                />
                <p className="text-[10px] text-orange-600/70 dark:text-orange-400/70 mt-1.5">
                  Default 10% for safety. Increase gradually. High torque
                  motors.
                </p>
              </div>
            )}

            {/* Arm Status Quick View */}
            {armsSummary.total_arms > 0 && (
              <div className="bg-neutral-50 dark:bg-zinc-800 rounded-lg p-3 border border-neutral-100 dark:border-zinc-700 mb-2">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase">
                    Arms
                  </span>
                  <span className="text-xs text-neutral-600 dark:text-zinc-300">
                    {armsSummary.connected}/{armsSummary.total_arms} connected
                  </span>
                </div>
                <button
                  onClick={() => {
                    onOpenArmManager();
                    setIsStatusMenuOpen(false);
                  }}
                  className="w-full py-1.5 text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950 rounded-lg transition-colors"
                >
                  Manage Arms →
                </button>
              </div>
            )}

            {/* Actions */}
            <button
              disabled={isReconnecting || isRestarting}
              onClick={async () => {
                setIsReconnecting(true);
                try {
                  const data = await systemApi.reconnect();
                  if ((data as any).status === "initializing") {
                    setConnectionState("INITIALIZING");
                    setStatusText("CONNECTING...");
                  }
                } catch (e) {
                  setStatusText("FAILED");
                  setIsReconnecting(false);
                }
              }}
              className="w-full py-2.5 rounded-xl bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-950 text-neutral-700 dark:text-zinc-300 font-medium text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {isReconnecting ? (
                <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              ) : (
                <RefreshCw className="w-4 h-4 text-blue-500" />
              )}
              {isReconnecting ? "Reconnecting..." : "Reconnect Hardware"}
            </button>

            {/* Restart with Confirmation */}
            {!restartConfirm ? (
              <button
                disabled={isRestarting}
                onClick={() => setRestartConfirm(true)}
                className="w-full py-2.5 rounded-xl bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 hover:border-red-300 dark:hover:border-red-600 hover:bg-red-50 dark:hover:bg-red-950 text-neutral-700 dark:text-zinc-300 font-medium text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                <Power className="w-4 h-4 text-red-500" />
                Restart System
              </button>
            ) : (
              <div className="flex gap-2 animate-in fade-in slide-in-from-right-2">
                <button
                  onClick={() => setRestartConfirm(false)}
                  className="flex-1 py-2.5 rounded-xl bg-neutral-100 dark:bg-zinc-800 hover:bg-neutral-200 dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-400 font-medium text-xs transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={async () => {
                    setIsRestarting(true);
                    setRestartConfirm(false);
                    setIsStatusMenuOpen(false);
                    try {
                      const res = await fetch(
                        `${API_BASE}/system/restart`,
                        { method: "POST" }
                      );
                      const data = await res.json();
                      if (data.status === "restarting") {
                        setStatusText("RESTARTING...");
                        setConnectionState("INITIALIZING");
                      }
                    } catch (e) {
                      setStatusText("RESTARTING...");
                    }
                  }}
                  className="flex-1 py-2.5 rounded-xl bg-red-500 hover:bg-red-600 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-lg shadow-red-200 dark:shadow-red-900/30 transition-all"
                >
                  Confirm
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

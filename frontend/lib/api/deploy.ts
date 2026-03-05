import { api } from "./client";
import type { DeploymentStatus, SafetyReadings } from "./types";

export const deployApi = {
  status: () => api.get<DeploymentStatus>("/deploy/status"),

  start: (config: Record<string, unknown>) =>
    api.post<{ status: string; mode: string; policy_id: string; active_arms: string[] }>(
      "/deploy/start",
      config
    ),

  stop: () =>
    api.post<{ status: string; summary: { frame_count: number; episode_count: number } }>(
      "/deploy/stop"
    ),

  settings: (data: Record<string, unknown>) =>
    api.patch<{ status: string; speed_scale: number }>("/deploy/settings", data),

  safety: () => api.get<SafetyReadings>("/deploy/safety"),

  resume: () => api.post<{ status: string }>("/deploy/resume"),

  estop: () => api.post<{ status: string }>("/deploy/estop"),

  reset: () => api.post<{ status: string }>("/deploy/reset"),

  restart: () =>
    api.post<{ status: string; state: string; mode: string }>("/deploy/restart"),

  retrain: (opts?: Record<string, unknown>) =>
    api.post<void>("/deploy/retrain", opts),

  startEpisode: () => api.post<void>("/deploy/episode/start"),

  stopEpisode: () => api.post<void>("/deploy/episode/stop"),

  nextEpisode: () =>
    api.post<{ status: string; previous_episode: unknown; new_episode: unknown }>(
      "/deploy/episode/next"
    ),
};

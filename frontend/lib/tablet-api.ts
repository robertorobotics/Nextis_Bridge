import { apiFetchSafe, api, API_BASE } from "@/lib/api/client";

export interface TabletCameraInfo {
  id: string;
  name: string;
  streaming: boolean;
}

export interface TabletStatus {
  connected: boolean;
  teleop_active: boolean;
  recording_active: boolean;
  deployment_active: boolean;
  connected_arms: number;
  connected_cameras: TabletCameraInfo[];
  episode_count: number;
  current_dataset: string | null;
  active_policy: string | null;
}

export const tabletApi = {
  /** Poll aggregated status — returns null on failure (never throws). */
  status: () => apiFetchSafe<TabletStatus>("/tablet/status"),

  startTeleop: () =>
    api.post<{ status: string }>("/teleop/start", {}),

  stopTeleop: () =>
    api.post<void>("/teleop/stop"),

  startEpisode: () =>
    api.post<{ status: string; message: string }>("/recording/episode/start"),

  stopEpisode: () =>
    api.post<{ status: string; message: string; episode_count?: number }>(
      "/recording/episode/stop"
    ),

  discardLastEpisode: () =>
    api.delete<{ status: string; message: string; episode_count?: number }>(
      "/recording/episode/last"
    ),

  stopDeploy: () =>
    api.post<{ status: string }>("/deploy/stop"),

  emergencyStop: () =>
    api.post<void>("/emergency/stop"),

  cameraStreamUrl: (cameraId: string) =>
    `${API_BASE}/video_feed/${cameraId}?max_width=960&quality=90`,
};

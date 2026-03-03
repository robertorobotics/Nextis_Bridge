import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { camerasApi } from '../../lib/api';
import type { CameraStatusEntry } from '../../lib/api/types';

export interface CameraFeedProps {
  cameraId: string;
  mode?: 'contain' | 'cover' | 'fill';
  className?: string;
  showOverlay?: boolean;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  label?: string;
  badge?: string;
  onFrameLoad?: () => void;
}

type FeedStatus = 'loading' | 'live' | 'error' | 'reconnecting';

const MAX_RETRIES = 5;

// Fixed stream parameters — never varies by layout, so the MJPEG URL is stable
const STREAM_MAX_WIDTH = 960;
const STREAM_QUALITY = 90;

const objectFitClass: Record<string, string> = {
  contain: 'object-contain',
  cover: 'object-cover',
  fill: 'object-fill',
};

export default function CameraFeed({
  cameraId,
  mode = 'contain',
  className = '',
  showOverlay = true,
  autoReconnect = true,
  reconnectInterval = 3000,
  label,
  badge,
  onFrameLoad,
}: CameraFeedProps) {
  const [status, setStatus] = useState<FeedStatus>('loading');
  const [retryCount, setRetryCount] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [resolution, setResolution] = useState<{ w: number; h: number } | null>(null);
  // Cache buster only set on error retry — not in initial URL
  const [retrySuffix, setRetrySuffix] = useState('');

  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stable MJPEG src — no cache buster on initial load, only appended on retry
  const src = `${camerasApi.videoFeedUrl(cameraId)}?max_width=${STREAM_MAX_WIDTH}&quality=${STREAM_QUALITY}${retrySuffix}`;

  // Fetch resolution once on mount
  useEffect(() => {
    let cancelled = false;
    camerasApi.status().then((statuses: Record<string, CameraStatusEntry>) => {
      if (cancelled) return;
      const entry = statuses[cameraId];
      if (entry?.actual_width && entry?.actual_height) {
        setResolution({ w: entry.actual_width, h: entry.actual_height });
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [cameraId]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, []);

  const startReconnect = useCallback(() => {
    if (!autoReconnect) return;

    setStatus('reconnecting');
    const seconds = Math.ceil(reconnectInterval / 1000);
    setCountdown(seconds);

    if (countdownRef.current) clearInterval(countdownRef.current);
    let remaining = seconds;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0 && countdownRef.current) {
        clearInterval(countdownRef.current);
        countdownRef.current = null;
      }
    }, 1000);

    if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    retryTimeoutRef.current = setTimeout(() => {
      setRetrySuffix(`&_t=${Date.now()}`);
      setStatus('loading');
    }, reconnectInterval);
  }, [autoReconnect, reconnectInterval]);

  const handleLoad = useCallback(() => {
    setStatus('live');
    setRetryCount(0);
    onFrameLoad?.();
  }, [onFrameLoad]);

  const handleError = useCallback(() => {
    setRetryCount(prev => {
      const next = prev + 1;
      if (next >= MAX_RETRIES) {
        setStatus('error');
      } else {
        startReconnect();
      }
      return next;
    });
  }, [startReconnect]);

  const handleManualRetry = useCallback(() => {
    setRetryCount(0);
    setRetrySuffix(`&_t=${Date.now()}`);
    setStatus('loading');
  }, []);

  const displayLabel = label || cameraId;

  const dotColor =
    status === 'live' ? 'bg-green-500' :
    status === 'error' ? 'bg-red-500' :
    'bg-yellow-500 animate-pulse';

  return (
    <div
      className={`w-full h-full bg-neutral-900 rounded-2xl overflow-hidden border border-white/5 relative group ${className}`}
    >
      {/* Loading shimmer */}
      {(status === 'loading' || status === 'reconnecting') && (
        <div className="absolute inset-0 bg-neutral-800 animate-pulse z-0" />
      )}

      {/* MJPEG stream — always rendered (unless permanent error) to keep connection alive */}
      {status !== 'error' && (
        <img
          src={src}
          alt={displayLabel}
          data-camera-id={cameraId}
          className={`w-full h-full ${objectFitClass[mode]} transition-opacity duration-300 ${status === 'live' ? 'opacity-100' : 'opacity-0'}`}
          onLoad={handleLoad}
          onError={handleError}
          draggable={false}
        />
      )}

      {/* Reconnecting overlay */}
      {status === 'reconnecting' && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-2">
            <RefreshCw className="w-6 h-6 text-white/30 animate-spin" />
            <span className="text-white/40 text-xs">Reconnecting in {countdown}s...</span>
          </div>
        </div>
      )}

      {/* Error state (after max retries) */}
      {status === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <AlertCircle className="w-8 h-8 text-white/20" />
            <span className="text-white/30 text-sm">Connection lost</span>
            <button
              onClick={handleManualRetry}
              className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-full text-xs text-white/70 transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        </div>
      )}

      {/* Top-left: Label badge */}
      {showOverlay && (
        <div className="absolute top-3 left-3 z-20 pointer-events-none">
          <span className="bg-black/50 backdrop-blur-md rounded-full px-3 py-1 text-xs text-white border border-white/10 font-medium flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${dotColor}`} />
            {displayLabel}
            {badge && (
              <span className="text-blue-400 text-[10px]">{badge}</span>
            )}
          </span>
        </div>
      )}

      {/* Bottom-right: Resolution badge */}
      {showOverlay && resolution && status === 'live' && (
        <div className="absolute bottom-3 right-3 z-20 pointer-events-none">
          <span className="bg-black/50 backdrop-blur-md rounded-full px-2.5 py-0.5 text-[10px] text-white/60 border border-white/5">
            {resolution.w}x{resolution.h}
          </span>
        </div>
      )}
    </div>
  );
}

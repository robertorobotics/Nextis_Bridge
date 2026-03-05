import { useState, useCallback, useRef } from "react";
import { usePolling } from "@/hooks/usePolling";
import { tabletApi, type TabletStatus } from "@/lib/tablet-api";

const OFFLINE_THRESHOLD = 2;

export function useTabletStatus(intervalMs = 1500) {
  const [status, setStatus] = useState<TabletStatus | null>(null);
  const [isOnline, setIsOnline] = useState(false);
  const failCount = useRef(0);

  const poll = useCallback(async () => {
    const result = await tabletApi.status();
    if (result !== null) {
      setStatus(result);
      setIsOnline(true);
      failCount.current = 0;
    } else {
      failCount.current += 1;
      if (failCount.current >= OFFLINE_THRESHOLD) {
        setIsOnline(false);
      }
    }
  }, []);

  usePolling(poll, intervalMs, true);

  return { status, isOnline };
}

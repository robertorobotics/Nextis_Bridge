"""Episode lifecycle management — start, stop, delete, sync.

Extracted from recording.py to keep file sizes manageable.
All functions take a TeleoperationService instance as their first argument.
"""

import logging
import time

logger = logging.getLogger(__name__)


def sync_to_disk(svc):
    """Flush all pending episode data to disk and close writers.

    MUST be called BEFORE any external deletion operation.

    This ensures:
    1. Metadata buffer is flushed to parquet (episodes saved to disk)
    2. Parquet writers are closed (prevents appending to wrong files)
    3. Disk state is consistent for external modifications

    Without this, the metadata_buffer may contain episode data that hasn't
    been written to disk yet. If deletion runs, it won't find the episode
    on disk, but the buffer still has it. When recording resumes, both
    the old buffered episode and new episode get saved = 2 episodes!
    """
    if not svc.dataset or not svc.session_active:
        print("[SYNC_TO_DISK] Skipped (no dataset or session not active)")
        return

    print(f"[SYNC_TO_DISK] BEFORE: meta.total_episodes={svc.dataset.meta.total_episodes}, episode_count={svc.episode_count}")

    try:
        # 1. Flush and close metadata writer (this flushes metadata_buffer to disk)
        if hasattr(svc.dataset, 'meta') and hasattr(svc.dataset.meta, '_close_writer'):
            svc.dataset.meta._close_writer()
            print("[SYNC_TO_DISK] Flushed metadata buffer and closed metadata writer")

        # 2. Close data parquet writer
        if hasattr(svc.dataset, '_close_writer'):
            svc.dataset._close_writer()
            print("[SYNC_TO_DISK] Closed data writer")

    except Exception as e:
        import traceback
        print(f"[SYNC_TO_DISK] Error: {e}")
        print(traceback.format_exc())


def refresh_metadata_from_disk(svc):
    """Re-read episode metadata from disk to sync after external modifications.

    Called AFTER external operations like delete_episode().
    Assumes sync_to_disk() was called BEFORE the external operation.

    CRITICAL: We reload meta.episodes from disk (via load_episodes) rather than
    setting it to None. When latest_episode is None but meta.episodes is loaded,
    LeRobot's _save_episode_data/_save_episode_metadata enter their resumption
    path which calls update_chunk_file_indices() — creating NEW parquet files
    instead of overwriting the existing ones. Setting episodes=None caused both
    branches to default to chunk=0/file=0, destroying all prior data.
    """
    if not svc.dataset or not svc.session_active:
        print("[REFRESH] Skipped (no dataset or session not active)")
        return

    import json
    from lerobot.datasets.utils import load_episodes

    info_path = svc.dataset.meta.root / "meta" / "info.json"
    print(f"[REFRESH] Reading from: {info_path}")

    if info_path.exists():
        try:
            with open(info_path, "r") as f:
                disk_info = json.load(f)

            old_memory_count = svc.dataset.meta.info.get("total_episodes", 0)
            disk_count = disk_info.get("total_episodes", 0)

            print(f"[REFRESH] Disk: {disk_count}, Memory: {old_memory_count}")

            # 1. Update info dict from disk
            svc.dataset.meta.info["total_episodes"] = disk_count
            svc.dataset.meta.info["total_frames"] = disk_info.get("total_frames", 0)

            verify_count = svc.dataset.meta.total_episodes
            print(f"[REFRESH] After update: meta.total_episodes = {verify_count}")
            if verify_count != disk_count:
                print(f"[REFRESH] ERROR: Update failed! Expected {disk_count}, got {verify_count}")

            # 2. Reset latest_episode so LeRobot enters the resumption path
            svc.dataset.meta.latest_episode = None
            if hasattr(svc.dataset, 'latest_episode'):
                svc.dataset.latest_episode = None

            # 3. Clear metadata buffer (should be empty after sync_to_disk)
            if hasattr(svc.dataset.meta, 'metadata_buffer'):
                svc.dataset.meta.metadata_buffer = []

            # 4. CRITICAL: Reload episodes from disk instead of setting to None.
            #    This enables LeRobot's resumption logic in _save_episode_data and
            #    _save_episode_metadata to create new parquet files (via
            #    update_chunk_file_indices) instead of overwriting chunk-000/file-000.
            try:
                svc.dataset.meta.episodes = load_episodes(svc.dataset.meta.root)
                ep_count = len(svc.dataset.meta.episodes) if svc.dataset.meta.episodes else 0
                print(f"[REFRESH] Reloaded {ep_count} episodes from disk")
            except Exception as e:
                print(f"[REFRESH] WARNING: Could not reload episodes: {e}")
                svc.dataset.meta.episodes = None

            # 5. Clear episode_buffer for fresh creation on next start_episode()
            if hasattr(svc.dataset, 'episode_buffer') and svc.dataset.episode_buffer is not None:
                svc.dataset.episode_buffer = None
                print("[REFRESH] Cleared stale episode_buffer")

            # 6. Data writer: already closed by sync_to_disk(). LeRobot's resumption
            #    logic will create a new writer at the next file index, preserving
            #    existing data. _current_file_start_frame is reset by resumption.

            # 7. Sync local episode counter
            svc.episode_count = disk_count

            print(f"[REFRESH] Complete: episode_count={svc.episode_count}, meta.total_episodes={svc.dataset.meta.total_episodes}")

        except Exception as e:
            import traceback
            print(f"[REFRESH] Error: {e}")
            print(traceback.format_exc())
    else:
        print(f"[REFRESH] ERROR: info.json not found at {info_path}")


def start_episode(svc):
    """Starts recording a new episode."""
    print("=" * 60)
    print("[START_EPISODE] Called!")
    print(f"  session_active: {svc.session_active}")
    print(f"  recording_active: {svc.recording_active}")
    print(f"  _episode_saving: {svc._episode_saving}")
    print(f"  dataset: {svc.dataset is not None}")
    print("=" * 60)

    if not svc.session_active:
        print("[START_EPISODE] ERROR: No active session!")
        raise Exception("No active recording session")

    if svc.recording_active:
        print("[START_EPISODE] Already recording, skipping")
        return

    # Wait for any ongoing episode save to complete before starting new episode
    if svc._episode_saving:
        print("[START_EPISODE] Waiting for previous episode to finish saving...")
        wait_start = time.time()
        max_wait = 10.0  # Maximum 10 seconds wait
        while svc._episode_saving and (time.time() - wait_start) < max_wait:
            time.sleep(0.1)
        if svc._episode_saving:
            raise Exception("Previous episode save timed out. Please try again.")
        print(f"[START_EPISODE] Previous save completed after {time.time() - wait_start:.1f}s")

    # Warn if teleop isn't running - recording needs teleop for actions
    if not svc.is_running:
        print("[START_EPISODE] WARNING: Teleop is NOT running!")
        print("[START_EPISODE] Recording requires teleop to be active for action/state data.")
        print("[START_EPISODE] Will use robot state fallback if available.")

    print("[START_EPISODE] Starting Episode Recording...")

    if svc.dataset:
        # Log current state BEFORE buffer creation (critical for debugging)
        meta_total = svc.dataset.meta.total_episodes
        print(f"[START_EPISODE] BEFORE buffer: meta.total_episodes={meta_total}, episode_count={svc.episode_count}")

        # Check for count mismatch and warn
        if svc.episode_count != meta_total:
            print("[START_EPISODE] WARNING: Count mismatch detected!")
            print(f"[START_EPISODE]   episode_count={svc.episode_count} != meta.total_episodes={meta_total}")

        # Initialize episode buffer - handle case where buffer is None on first use
        try:
            if svc.dataset.episode_buffer is None:
                print("[START_EPISODE] Creating new episode buffer (first episode)")
                svc.dataset.episode_buffer = svc.dataset.create_episode_buffer()
            else:
                print("[START_EPISODE] Clearing existing episode buffer")
                svc.dataset.clear_episode_buffer()
        except Exception as e:
            print(f"[START_EPISODE] Error with buffer, recreating: {e}")
            svc.dataset.episode_buffer = svc.dataset.create_episode_buffer()

        # Log the episode index that will be used
        buffer_ep_idx = svc.dataset.episode_buffer.get("episode_index", "N/A")
        print(f"[START_EPISODE] Episode buffer ready, episode_index={buffer_ep_idx}")
        print(f"[START_EPISODE] Dataset features: {list(svc.dataset.features.keys())[:5]}...")

    # Reset frame counter for new episode
    svc._recording_frame_counter = 0

    svc.recording_active = True
    print("[START_EPISODE] recording_active = True, ready to capture frames!")


def stop_episode(svc):
    """Stops current episode and saves it."""
    print("=" * 60)
    print("[STOP_EPISODE] Called!")
    print(f"  recording_active: {svc.recording_active}")
    print(f"  session_active: {svc.session_active}")
    print(f"  dataset: {svc.dataset is not None}")
    if svc.dataset:
        print(f"  dataset type: {type(svc.dataset)}")
    print("=" * 60)

    if not svc.recording_active:
        print("[STOP_EPISODE] WARNING: recording_active is False, returning")
        return

    # Acquire lock to prevent race with stop_session
    # This ensures save_episode() completes before finalize() can be called
    print("[STOP_EPISODE] Acquiring episode save lock...")
    with svc._episode_save_lock:
        svc._episode_saving = True
        print("[STOP_EPISODE] Lock acquired, proceeding with save")

        # Capture dataset reference BEFORE changing state
        # This prevents race conditions where dataset could be set to None
        current_dataset = svc.dataset

        print("[STOP_EPISODE] Stopping Episode Recording...")
        svc.recording_active = False

        # Wait for frame queue to drain before saving episode
        print(f"[STOP_EPISODE] Waiting for frame queue to drain ({len(svc._frame_queue)} frames)...")
        drain_timeout = 5.0  # seconds
        drain_start = time.time()
        while len(svc._frame_queue) > 0 and (time.time() - drain_start) < drain_timeout:
            time.sleep(0.05)
        if len(svc._frame_queue) > 0:
            print(f"[STOP_EPISODE] WARNING: Queue not fully drained ({len(svc._frame_queue)} remaining)")
        else:
            print("[STOP_EPISODE] Queue drained successfully")

        # Reset first frame logging flag for next episode
        if hasattr(svc, '_first_frame_logged'):
            delattr(svc, '_first_frame_logged')
        if hasattr(svc, '_last_rec_error'):
            delattr(svc, '_last_rec_error')

        if current_dataset is not None:
            try:
                # Check episode buffer before saving
                if hasattr(current_dataset, 'episode_buffer') and current_dataset.episode_buffer:
                    buffer_size = current_dataset.episode_buffer.get('size', 0)
                    print(f"[STOP_EPISODE] Episode buffer has {buffer_size} frames (captured {svc._recording_frame_counter})")

                    if buffer_size == 0:
                        print("[STOP_EPISODE] WARNING: Buffer size is 0, no frames were recorded!")
                        print("[STOP_EPISODE] This means observations weren't captured during recording.")
                        svc._episode_saving = False
                        return
                else:
                    print("[STOP_EPISODE] WARNING: Episode buffer is empty or missing!")
                    print(f"  has episode_buffer attr: {hasattr(current_dataset, 'episode_buffer')}")
                    if hasattr(current_dataset, 'episode_buffer'):
                        print(f"  episode_buffer value: {current_dataset.episode_buffer}")
                    svc._episode_saving = False
                    return

                # Log state BEFORE save
                print(f"[STOP_EPISODE] BEFORE save: meta.total_episodes={current_dataset.meta.total_episodes}, episode_count={svc.episode_count}")

                # Diagnostic: Check image writer status before save
                if hasattr(current_dataset, 'image_writer') and current_dataset.image_writer:
                    try:
                        queue_size = current_dataset.image_writer.queue.qsize()
                        print(f"[STOP_EPISODE] Image writer queue size: {queue_size}")
                    except Exception:
                        print("[STOP_EPISODE] Image writer queue size: (unable to check)")

                print("[STOP_EPISODE] Calling save_episode()...")
                save_start = time.time()
                # Note: task is already included in each frame, no need to pass to save_episode
                current_dataset.save_episode()
                save_duration = time.time() - save_start
                mode_str = "streaming" if getattr(svc, '_streaming_encoding', False) else "batch"
                print(f"[STOP_EPISODE] save_episode() completed in {save_duration:.1f}s (mode={mode_str})")
                # Log state AFTER save
                print(f"[STOP_EPISODE] AFTER save: meta.total_episodes={current_dataset.meta.total_episodes}")
                svc.episode_count += 1
                print(f"[STOP_EPISODE] SUCCESS! Episode {svc.episode_count} Saved! ({svc._recording_frame_counter} frames)")
            except Exception as e:
                import traceback
                print(f"[STOP_EPISODE] ERROR saving episode: {e}")
                print(traceback.format_exc())
        else:
            print("[STOP_EPISODE] WARNING: No dataset available (current_dataset is None)!")

        svc._episode_saving = False
        print("[STOP_EPISODE] Episode save lock released")


def delete_last_episode(svc):
    """Deletes the last recorded episode (if possible/implemented)."""
    logger.warning("Delete Last Episode not fully supported yet.")

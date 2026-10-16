from . import config_store as _config_store
from .logger import log

get_config_snapshot = _config_store.get_config_snapshot
get_config_value = _config_store.get_config_value
invalidate_config_cache = _config_store.invalidate_config_cache
update_config_value = _config_store.update_config_value
update_config_values = _config_store.update_config_values


def _ensure_config_exists() -> bool:
    return _config_store.ensure_config_exists()


def make_comfy_tqdm_class(
    desc: str | None = None,
    log_prefix: str | None = None,
    heartbeat_interval_seconds: float = 30.0,
):
    import threading
    import time

    import comfy.utils  # type: ignore

    fallback_desc = desc
    prefix = log_prefix

    def format_bytes(value) -> str:
        size = float(value or 0)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size < 1024 or unit == "TiB":
                precision = 0 if unit == "B" else 1
                return f"{size:.{precision}f} {unit}"
            size /= 1024
        return f"{size:.1f} TiB"

    def format_duration(seconds) -> str:
        elapsed = max(0, int(seconds or 0))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    class ComfyTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0) or 0
            self.n = kwargs.get("initial", 0)
            self.desc = kwargs.get("desc") or fallback_desc or "Download"
            self._last_logged_percent = int((self.n / self.total) * 100) if self.total else 0
            self._started_at = time.monotonic()
            self._last_logged_at = self._started_at
            self._last_status_at = self._started_at
            self._has_byte_progress = self.n > 0
            self._heartbeat_interval = max(0.0, heartbeat_interval_seconds)
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = None
            self._comfy_progress_disabled = False
            self.pbar = self._make_comfy_progress(self.total)
            if self.n > 0:
                self._update_comfy_progress()
            if prefix is not None:
                total_text = f" ({format_bytes(self.total)})" if self.total else ""
                log.msg(prefix, f"Downloading {self.desc}{total_text}")
            self._start_heartbeat()

        def _make_comfy_progress(self, total):
            if self._comfy_progress_disabled:
                return None
            try:
                return comfy.utils.ProgressBar(max(total, 1))
            except Exception as error:  # noqa: BLE001 -- optional UI telemetry
                self._disable_comfy_progress(error)
                return None

        def _disable_comfy_progress(self, error):
            self._comfy_progress_disabled = True
            self.pbar = None
            log.debug(
                "Download",
                "ComfyUI progress events are unavailable outside an active "
                f"prompt; continuing without them ({error})",
            )

        def _update_comfy_progress(self):
            if self.pbar is None:
                return
            try:
                self.pbar.update_absolute(self.n, self.total)
            except Exception as error:  # noqa: BLE001 -- optional UI telemetry
                self._disable_comfy_progress(error)

        def _start_heartbeat(self):
            if prefix is None or self._heartbeat_interval <= 0:
                return
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="smartllm-download-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

        def _stop_heartbeat(self):
            heartbeat_thread = self._heartbeat_thread
            if heartbeat_thread is None:
                return
            self._heartbeat_stop.set()
            if heartbeat_thread is not threading.current_thread():
                heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None

        def _heartbeat_loop(self):
            while not self._heartbeat_stop.is_set():
                remaining = max(
                    0.0,
                    self._heartbeat_interval - (time.monotonic() - self._last_status_at),
                )
                if self._heartbeat_stop.wait(remaining):
                    return
                now = time.monotonic()
                if now - self._last_status_at < self._heartbeat_interval:
                    continue
                if self._has_byte_progress and self.total:
                    progress_text = f"latest {format_bytes(self.n)}/{format_bytes(self.total)}"
                elif self._has_byte_progress:
                    progress_text = f"latest {format_bytes(self.n)} downloaded"
                else:
                    progress_text = "waiting for byte progress"
                log.msg(
                    prefix,
                    f"{self.desc}: still working "
                    f"(elapsed {format_duration(now - self._started_at)}, {progress_text})",
                )
                self._last_status_at = now

        def _log_progress(self):
            if prefix is None:
                return
            now = time.monotonic()
            if self.total:
                percent = min(100, int((self.n / self.total) * 100))
                if (
                    percent < 100
                    and percent < self._last_logged_percent + 5
                    and now - self._last_logged_at < 30
                ):
                    return
                if percent == self._last_logged_percent:
                    return
                log.msg(
                    prefix,
                    f"{self.desc}: {percent}% "
                    f"({format_bytes(self.n)}/{format_bytes(self.total)}, "
                    f"elapsed {format_duration(now - self._started_at)})",
                )
                self._last_logged_percent = percent
            elif now - self._last_logged_at >= 30:
                log.msg(
                    prefix,
                    f"{self.desc}: {format_bytes(self.n)} downloaded "
                    f"(elapsed {format_duration(now - self._started_at)})",
                )
            else:
                return
            self._last_logged_at = now
            self._last_status_at = now

        def update(self, n=1):
            increment = n or 0
            self.n += increment
            if increment > 0:
                self._has_byte_progress = True
            self._update_comfy_progress()
            self._log_progress()

        def close(self):
            self._stop_heartbeat()
            self._update_comfy_progress()
            self._log_progress()

        def set_postfix_str(self, _value, **_kwargs):
            pass

        def reset(self, total=None):
            self._stop_heartbeat()
            if total is not None:
                self.total = total
                self.pbar = self._make_comfy_progress(self.total)
            self.n = 0
            self._last_logged_percent = 0
            self._started_at = time.monotonic()
            self._last_logged_at = self._started_at
            self._last_status_at = self._started_at
            self._has_byte_progress = False
            self._start_heartbeat()

        def refresh(self):
            if self.n > 0:
                self._update_comfy_progress()

        def set_description(self, desc=None, refresh=True):
            if desc:
                self.desc = desc
            if refresh:
                self.refresh()

        def set_description_str(self, desc=None, refresh=True):
            self.set_description(desc, refresh=refresh)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    return ComfyTqdm

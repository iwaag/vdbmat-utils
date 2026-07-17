"""designlab: a viser GUI that fills in a generator config form, saves/loads
it, and runs the one-shot generate pipeline to publish a canonical run
bundle for the existing Mitsuba stage viewer to pick up.

This module is the GUI "thin skin" only (roadmap: "GUI 薄皮 + browser-free
core"). It wires ``designlab_registry`` / ``designlab_configs`` /
``designlab_pipeline`` / ``designlab_jobs`` to viser widgets and contains no
generation, validation, or pipeline logic of its own — everything it calls
is already covered by the browser-free unit/integration tests in Steps 1-2.
There is intentionally no automated test for this module (this
environment cannot drive a browser); see ``docs/designlab.md`` for the
manual verification procedure.

Run with (from ``vdbmat-utils``)::

    uv run --group designlab python examples/designlab/designlab_app.py -- \\
        --config-root <CONFIG_ROOT> --output-root <OUTPUT_ROOT> \\
        [--work-root <WORK_ROOT>] [--port 8081]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from designlab_configs import (
    DesignlabConfigError,
    load_config,
    save_config,
    scan_config_catalog,
)
from designlab_jobs import JobBusyError, JobWorker
from designlab_pipeline import (
    PublishError,
    check_roots,
    default_work_root,
    run_generate_job,
    sweep_stale_work_dirs,
)
from designlab_registry import REGISTRY

_NO_CONFIG_SENTINEL = "(none)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="designlab_app")
    parser.add_argument(
        "--config-root",
        type=Path,
        required=True,
        help="server-local directory the config catalog scans/saves into",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="canonical run-bundle publish root (point the existing "
        "mitsuba_stage_viewer's --input-root here)",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="scratch directory for in-progress jobs (default: an "
        "--output-root sibling, '<basename>.designlab-work')",
    )
    parser.add_argument("--port", type=int, default=8081)
    return parser.parse_args(argv)


def _check_environment() -> None:
    """Fail fast, with an actionable message, on a venv mismatch.

    Both ``vdbmat_utils`` (this package) and ``vdbmat`` (imported via
    subprocess only, but its CLI must be reachable from this interpreter)
    are expected to be importable in the same venv (roadmap risk:
    "サブプロセスが別 venv で走る事故").
    """
    missing = []
    for module_name in ("vdbmat_utils", "vdbmat"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "designlab: cannot import "
            + ", ".join(missing)
            + " in this interpreter. Run via `uv run --group designlab "
            "python examples/designlab/designlab_app.py -- ...` from "
            "vdbmat-utils."
        )


class DesignlabApp:
    def __init__(self, args: argparse.Namespace) -> None:
        import viser

        _check_environment()

        self.config_root = args.config_root.resolve()
        if not self.config_root.is_dir():
            raise SystemExit(
                f"designlab: --config-root is not a directory: {args.config_root}"
            )

        output_root = args.output_root.resolve()
        if not output_root.is_dir():
            raise SystemExit(
                f"designlab: --output-root is not a directory: {args.output_root}"
            )

        work_root = args.work_root if args.work_root is not None else default_work_root(
            output_root
        )
        work_root.mkdir(parents=True, exist_ok=True)
        self.output_root, self.work_root = check_roots(output_root, work_root)
        sweep_stale_work_dirs(self.work_root)

        # Registry has exactly one method in Phase 2; the dropdown below is
        # display-only until Phase 3 adds a second one and this needs to
        # rebuild the form panel on selection change.
        self.method = REGISTRY[0]
        self._catalog: dict[str, object] = {}

        self.job_worker = JobWorker(on_error=self._on_job_error)
        self.job_worker.start()

        self.server = viser.ViserServer(host="127.0.0.1", port=args.port)
        gui = self.server.gui
        gui.set_panel_label("designlab")

        gui.add_dropdown(
            "method",
            tuple(m.title for m in REGISTRY),
            initial_value=self.method.title,
        )

        with gui.add_folder(f"{self.method.title} config"):
            self.form_binding = self.method.build_form(gui)

        with gui.add_folder("Config catalog"):
            options = self._catalog_options()
            self.config_dropdown = gui.add_dropdown(
                "config", options, initial_value=options[0]
            )
            self.config_refresh_button = gui.add_button("Refresh")
            self.config_load_button = gui.add_button("Load")
            self.config_save_name = gui.add_text("save as (name)", "demo")
            self.config_save_button = gui.add_button("Save")

        with gui.add_folder("Generate"):
            self.name_input = gui.add_text("name", "demo")
            self.generate_button = gui.add_button("Generate")

        self.status = gui.add_markdown("ready.")

        self.config_refresh_button.on_click(lambda _event: self._refresh_catalog())
        self.config_load_button.on_click(lambda _event: self._load_config())
        self.config_save_button.on_click(lambda _event: self._save_config())
        self.generate_button.on_click(lambda _event: self._on_generate())

    # -- config catalog ------------------------------------------------------

    def _catalog_options(self) -> tuple[str, ...]:
        candidates = scan_config_catalog(self.config_root)
        self._catalog = {c.root_relative: c for c in candidates}
        if not candidates:
            return (_NO_CONFIG_SENTINEL,)
        return tuple(c.root_relative for c in candidates)

    def _refresh_catalog(self) -> None:
        options = self._catalog_options()
        self.config_dropdown.options = options
        if self.config_dropdown.value not in options:
            self.config_dropdown.value = options[0]
        self.status.content = f"catalog refreshed ({len(self._catalog)} config(s))"

    def _load_config(self) -> None:
        selection = self.config_dropdown.value
        candidate = self._catalog.get(selection)
        if candidate is None:
            self.status.content = "**load failed**: no config selected"
            return
        try:
            config = load_config(candidate.path, self.config_root, candidate.method)
        except DesignlabConfigError as error:
            self.status.content = f"**load failed**: {error}"
            return
        candidate.method.config_to_form(self.form_binding, config)
        self.status.content = f"loaded: {selection}"

    def _save_config(self) -> None:
        try:
            config = self.method.form_to_config(self.form_binding)
        except Exception as error:
            self.status.content = f"**save failed**: {error}"
            return
        try:
            target = save_config(
                config, self.config_root, self.method, self.config_save_name.value
            )
        except DesignlabConfigError as error:
            self.status.content = f"**save failed**: {error}"
            return
        self.status.content = f"saved: {target.name}"
        self._refresh_catalog()

    # -- generate --------------------------------------------------------------

    def _on_generate(self) -> None:
        try:
            config = self.method.form_to_config(self.form_binding)
        except Exception as error:
            self.status.content = f"**generate failed**: {error}"
            return
        name = self.name_input.value
        method = self.method

        def job() -> None:
            def on_stage(stage: str) -> None:
                self.status.content = f"generate: {stage}…"

            try:
                result = run_generate_job(
                    method=method,
                    config=config,
                    name=name,
                    output_root=self.output_root,
                    work_root=self.work_root,
                    seq=self.job_worker.next_seq(),
                    on_stage=on_stage,
                )
            except PublishError as error:
                self.status.content = (
                    f"**generate failed at {error.stage}**\n```\n{error.message}\n```"
                )
                return
            reused_note = " (reused existing bundle)" if result.reused else ""
            self.status.content = f"published: {result.publish_path}{reused_note}"

        try:
            self.job_worker.submit(job)
        except JobBusyError as error:
            self.status.content = f"**cannot start**: {error}"
            return
        self.status.content = "generate: queued…"

    def _on_job_error(self, error: BaseException) -> None:
        self.status.content = f"**unexpected error**: {error}"


def main() -> None:
    args = _parse_args()
    app = DesignlabApp(args)
    print(f"designlab ready: http://127.0.0.1:{args.port} (work root: {app.work_root})")
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

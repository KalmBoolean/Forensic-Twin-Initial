"""Workspace model for the Forensic Digital Twin app.

On-disk layout:

    <workspaces root>/
        <WorkspaceName>/
            workspace.json        metadata + run history
            inbox/                images waiting to be reconstructed
            project/              what the bottom "file explorer" shows
                Run_001/
                    Images/       the images that were consumed by the build
                    insta_output/ InstantSplat output (+ gsplat/ with the .ply)
                Run_002/ ...
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ROOT = Path.home() / "ForensicTwinWorkspaces"
_NAME_RE = re.compile(r"[\w\- ]{1,48}")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Run:
    id: str
    created: str
    image_count: int
    status: str = "running"  # running | done
    ply: str | None = None  # path relative to the workspace root
    images: list = field(default_factory=list)


class Workspace:
    META = "workspace.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.name = self.root.name
        self.created = _now()
        self.runs: list[Run] = []

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        if self.meta_path.exists():
            self._load()
        else:
            self._save()

    # ------------------------------------------------------------ paths
    @property
    def inbox_dir(self) -> Path:
        return self.root / "inbox"

    @property
    def project_dir(self) -> Path:
        return self.root / "project"

    @property
    def meta_path(self) -> Path:
        return self.root / self.META

    def run_dir(self, run: Run) -> Path:
        return self.project_dir / run.id

    def run_images_dir(self, run: Run) -> Path:
        return self.run_dir(run) / "Images"

    def run_output_dir(self, run: Run) -> Path:
        return self.run_dir(run) / "insta_output"

    # ------------------------------------------------------ persistence
    def _load(self):
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}

        self.created = data.get("created", self.created)
        self.runs = []

        for raw in data.get("runs", []):
            run = Run(**raw)
            if run.status == "running":
                # app died mid-build: throw the half-finished run away
                shutil.rmtree(self.run_dir(run), ignore_errors=True)
                continue
            self.runs.append(run)

        self._save()

    def _save(self):
        data = {
            "name": self.name,
            "created": self.created,
            "runs": [asdict(r) for r in self.runs],
        }
        self.meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ inbox
    def inbox_images(self) -> list[Path]:
        return sorted(
            p for p in self.inbox_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def add_images(self, paths) -> int:
        """Copy images into the inbox. Returns how many were added."""
        added = 0
        for src in map(Path, paths):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            if src.parent.resolve() == self.inbox_dir.resolve():
                continue

            dest = self.inbox_dir / src.name
            i = 1
            while dest.exists():
                dest = self.inbox_dir / f"{src.stem}_{i}{src.suffix}"
                i += 1

            shutil.copy2(src, dest)
            added += 1
        return added

    def remove_inbox_images(self, names) -> None:
        for name in names:
            (self.inbox_dir / name).unlink(missing_ok=True)

    # ------------------------------------------------------------- runs
    def begin_run(self) -> Run:
        """Snapshot the inbox into a fresh run folder, ready for the pipeline."""
        images = self.inbox_images()
        if not images:
            raise ValueError("No images queued in this workspace.")

        n = len(self.runs) + 1
        while (self.project_dir / f"Run_{n:03d}").exists():
            n += 1

        run = Run(
            id=f"Run_{n:03d}",
            created=_now(),
            image_count=len(images),
            images=[p.name for p in images],
        )

        img_dir = self.run_images_dir(run)
        img_dir.mkdir(parents=True)
        for p in images:
            shutil.copy2(p, img_dir / p.name)

        self.runs.append(run)
        self._save()
        return run

    def complete_run(self, run: Run) -> Path | None:
        """Mark the run done and consume the inbox images (they now live in the run)."""
        self.remove_inbox_images(run.images)

        plys = sorted(
            self.run_dir(run).rglob("*.ply"),
            key=lambda p: p.stat().st_mtime,
        )
        ply = plys[-1] if plys else None

        run.status = "done"
        run.ply = ply.relative_to(self.root).as_posix() if ply else None
        self._save()
        return ply

    def fail_run(self, run: Run) -> None:
        """Discard a failed run; inbox images are untouched."""
        shutil.rmtree(self.run_dir(run), ignore_errors=True)
        self.runs = [r for r in self.runs if r.id != run.id]
        self._save()

    # --------------------------------------------------- reconstruction
    def latest_ply(self) -> Path | None:
        for run in reversed(self.runs):
            if run.ply and (self.root / run.ply).exists():
                return self.root / run.ply
        return None

    @property
    def has_reconstruction(self) -> bool:
        return self.latest_ply() is not None


class WorkspaceManager:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_names(self) -> list[str]:
        return sorted(
            (p.name for p in self.root.iterdir()
             if p.is_dir() and (p / Workspace.META).exists()),
            key=str.lower,
        )

    def create(self, name: str) -> Workspace:
        name = name.strip()
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                "Use 1-48 letters, numbers, spaces, dashes or underscores."
            )
        if (self.root / name).exists():
            raise ValueError(f'A workspace called "{name}" already exists.')
        return Workspace(self.root / name)

    def open(self, name: str) -> Workspace:
        return Workspace(self.root / name)

    def delete(self, name: str) -> None:
        shutil.rmtree(self.root / name, ignore_errors=True)

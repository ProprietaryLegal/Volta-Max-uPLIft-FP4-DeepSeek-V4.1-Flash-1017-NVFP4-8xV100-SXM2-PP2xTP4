"""Decode route counters of the three-tier layers (campaign card 8.2; plan 00, M3).

Off unless ``VLLM_NVFP4_8R_STATS=1``. With the switch on every three-tier
layer owns a ``RouteStats``: device tensors updated by device operations only,
so the updates are captured in the decode CUDA graphs and run on every replay.
Per layer and TP rank:

  stat       int64 [5]    calls of ``forward`` (decode rounds), routes,
                          HOST routes, HOST experts fetched (distinct per
                          call), WARM routes
  gap_hist   int64 [7]    HOST routes by rounds since the expert's last use:
                          <= 1, 2, 4, 16, 64, 256, more (or never)
  routes     int64 [384]  routes per expert
  last_round int32 [385]  the round of each expert's last HOST use

``gap_hist`` up to a bucket, over the HOST routes, is the hit rate of a cache
that keeps every HOST expert used in that many rounds; ``recent`` in a
snapshot is the number of experts such a cache would hold.

Data boundary: the per-expert arrays (``routes``, the tier arrays) derive from
the text the server was sent. Snapshots are written only into the directory
the loader names (the tally's own private directory); nothing here logs a
per-expert value. A worker thread writes one snapshot per card each time the
text in ``DIRECTORY/mark`` changes to a new label; a run is bracketed by
two labels and read as their difference.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

STATS_ENV = "VLLM_NVFP4_8R_STATS"
STATS_DIR = "stats"  # under the tally's directory
MARK_FILE = "mark"
STAT_FIELDS = ("calls", "routes", "host_routes", "host_groups", "warm_routes")
GAP_EDGES = (1, 2, 4, 16, 64, 256)
RECENT = (4, 16, 64, 256)
NEVER = -(2**30)
LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
POLL_SECONDS = 1.0
_E = tiers.NUM_EXPERTS
_MAX_ROUTES = 36  # six rows of six routes: the decode batch


def stats_from_env(raw: str | None) -> bool:
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{STATS_ENV} must be 0 or 1, got {raw!r}")


class RouteStats:
    """The counters of one layer on one card; ``tier`` is its tier array."""

    def __init__(self, tier: np.ndarray, device: torch.device) -> None:
        maps = tiers.slot_maps(tier)  # validates the array
        self.is_host = torch.from_numpy(maps.host >= 0).to(device)
        self.is_warm = torch.from_numpy(maps.iq >= 0).to(device)
        self.stat = torch.zeros(len(STAT_FIELDS), dtype=torch.int64, device=device)
        self.gap_hist = torch.zeros(len(GAP_EDGES) + 1, dtype=torch.int64, device=device)
        self.routes = torch.zeros(_E, dtype=torch.int64, device=device)
        # Entry 384 takes the writes of every route that is not a HOST route.
        self.last_round = torch.full((_E + 1,), NEVER, dtype=torch.int32, device=device)
        self.edges = torch.tensor(GAP_EDGES, dtype=torch.int64, device=device)
        self.ones = torch.ones(_MAX_ROUTES, dtype=torch.int64, device=device)

    def count(self, topk_ids: torch.Tensor) -> None:
        """One decode call's routes [M, 6]. Device operations only."""
        flat = topk_ids.reshape(-1).long()
        n = flat.numel()
        if n > _MAX_ROUTES:
            raise ValueError(f"a decode call has at most {_MAX_ROUTES} routes, got {n}")
        is_host = self.is_host[flat]
        slot = torch.where(is_host, flat, _E)
        r = self.stat[0]  # a view: the round number when the graph replays
        gap = r - self.last_round[slot]
        self.gap_hist.scatter_add_(0, torch.bucketize(gap, self.edges), is_host.long())
        # Duplicate slots all write the same value, so the result is defined.
        self.last_round[slot] = torch.where(is_host, r, NEVER).int()
        self.routes.scatter_add_(0, flat, self.ones[:n])
        self.stat[2] += is_host.sum()
        self.stat[3] += (self.last_round[:_E] == r).sum()
        self.stat[4] += self.is_warm[flat].sum()
        self.stat[1] += n
        self.stat[0] += 1

    def read(self) -> dict[str, np.ndarray]:
        r = self.stat[0]
        age = r - self.last_round[:_E]
        recent = torch.stack([(age <= g).sum() for g in RECENT])
        return {
            "stat": self.stat.cpu().numpy(),
            "gap_hist": self.gap_hist.cpu().numpy(),
            "recent": recent.cpu().numpy(),
            "routes": self.routes.cpu().numpy(),
        }


def check_label(label: str) -> str:
    if LABEL.fullmatch(label) is None:
        raise ValueError(f"a stats label is lower-case letters, digits and '-', got {label!r}")
    return label


def snapshot_path(directory: Path, label: str, first_layer: int, rank: int) -> Path:
    return directory / f"snap-{check_label(label)}-s{first_layer:02d}-r{rank}.npz"


@dataclass(frozen=True)
class Snapshot:
    """The cumulative counters of one card (a stage's TP rank) at a label."""

    label: str
    boot: str  # names the worker process: equal in two snapshots of one boot
    first_layer: int
    rank: int
    unix_time: float
    layers: np.ndarray  # int32 [L]
    stat: np.ndarray  # int64 [L, 5]
    gap_hist: np.ndarray  # int64 [L, 7]
    recent: np.ndarray  # int64 [L, 4]
    routes: np.ndarray  # int64 [L, 384]
    tier: np.ndarray  # int8 [L, 384]

    def __post_init__(self) -> None:
        check_label(self.label)
        n = len(self.layers)
        want = {
            "stat": (n, len(STAT_FIELDS)), "gap_hist": (n, len(GAP_EDGES) + 1),
            "recent": (n, len(RECENT)), "routes": (n, _E), "tier": (n, _E),
        }
        for name, shape in want.items():
            got = getattr(self, name)
            if got.shape != shape or got.dtype.kind != "i":
                raise ValueError(f"snapshot {name} is {got.dtype} {got.shape}, want integer {shape}")
        if n == 0 or self.layers.tolist() != list(range(self.first_layer, self.first_layer + n)):
            raise ValueError("snapshot layers are not one run from the stage's first layer")

    def save(self, directory: Path) -> Path:
        path = snapshot_path(directory, self.label, self.first_layer, self.rank)
        meta = {"label": self.label, "boot": self.boot, "first_layer": self.first_layer,
                "rank": self.rank, "unix_time": self.unix_time}
        buf = io.BytesIO()
        np.savez(buf, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), layers=self.layers,
                 stat=self.stat, gap_hist=self.gap_hist, recent=self.recent, routes=self.routes,
                 tier=self.tier)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> Snapshot:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            return cls(meta["label"], meta["boot"], meta["first_layer"], meta["rank"],
                       meta["unix_time"], *(np.asarray(data[k]) for k in
                                            ("layers", "stat", "gap_hist", "recent", "routes", "tier")))


def take_snapshot(
    label: str, boot: str, first_layer: int, rank: int,
    stats: dict[int, RouteStats], tier: dict[int, np.ndarray],
) -> Snapshot:
    layers = sorted(stats)
    if layers != sorted(tier):
        raise ValueError("counters and tier arrays cover different layers")
    read = [stats[layer].read() for layer in layers]
    return Snapshot(
        label, boot, first_layer, rank, time.time(), np.asarray(layers, np.int32),
        *(np.stack([r[k] for r in read]) for k in ("stat", "gap_hist", "recent", "routes")),
        np.stack([np.asarray(tier[layer], np.int8) for layer in layers]),
    )


def read_mark(directory: Path) -> str | None:
    """The label in the mark file, or None when there is no mark file."""
    path = directory / MARK_FILE
    if not path.is_file():
        return None
    return check_label(path.read_text().strip())


class StatsWriter(threading.Thread):
    """One per worker: a snapshot of this card's layers at every new mark.

    The label already in the mark file when the worker starts belongs to an
    earlier boot and is not answered."""

    def __init__(
        self, directory: Path, first_layer: int, rank: int,
        stats: dict[int, RouteStats], tier: dict[int, np.ndarray],
    ) -> None:
        super().__init__(name=f"nvfp4-8r-stats-s{first_layer}-r{rank}", daemon=True)
        directory.mkdir(exist_ok=True)
        self.directory, self.first_layer, self.rank = directory, first_layer, rank
        self.stats, self.tier = stats, tier
        self.boot = f"{os.getpid()}-{time.time_ns()}"
        self.answered = read_mark(directory)

    def poll(self) -> Path | None:
        """Write a snapshot if the mark names a label not yet answered."""
        label = read_mark(self.directory)
        if label is None or label == self.answered:
            return None
        path = take_snapshot(
            label, self.boot, self.first_layer, self.rank, self.stats, self.tier
        ).save(self.directory)
        self.answered = label
        return path

    def run(self) -> None:
        while True:
            time.sleep(POLL_SECONDS)
            self.poll()

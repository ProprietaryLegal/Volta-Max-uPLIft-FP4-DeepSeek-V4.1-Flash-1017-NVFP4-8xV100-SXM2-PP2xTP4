#!/usr/bin/env python3
"""Build the item plan of the NVFP4 closeness test from public sources.

  build_items.py fetch --data DIR                       download every source (pinned revisions) into DIR/<source>/
  build_items.py build --data DIR --tokenizer FILE --out PLAN --ids IDS
                                                        cut the contexts, sample the MMLU items, write the plan
  build_items.py --dry                                  print the sources, licences, counts and outputs; no network
  build_items.py --self-test                            CPU checks on hand-made strings; no network

Every source is public domain or under a licence that allows redistribution (SOURCES below, and SOURCES.md).
None of it is text that was used to build the routing tally of the hot_iq build (that used Python 3.12 standard
library source, US Code title 28, six short prompts and private text): no CPython source, no US Code.

The plan holds the context texts and MC prompts (inputs only, never model output). The ids file is the same plan
without any text: item ids, sources, offsets and sha256 of every text, so anyone can rebuild the texts with this
script and check them. The build is deterministic: the same fetched files give the same plan, byte for byte.
Fetch fails loudly on any HTTP error and never retries.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closeness_lib as cl  # noqa: E402

SEED = 20261110
UA = {"User-Agent": "closeness-test-builder/1.0"}
TARGETS = (96, 160, 224)  # context lengths in tokens, cycled within each source
SKIP_EDGE = 0.05  # contexts start after the first 5 % and before the last 5 % of a document (licence headers, contents)
GH = "https://raw.githubusercontent.com"
MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
MMLU_PER_SUBJECT = 9  # 57 subjects x 9 = 513 items
MMLU_PAGE = 100  # the datasets-server's largest page
MMLU_PAUSE_S = 3.0  # it answered 429 to requests 0.5 s apart; 141 pages at 3 s is about 7 minutes
DOLLY_CATEGORIES = ("brainstorming", "classification", "closed_qa", "creative_writing", "general_qa",
                    "information_extraction", "open_qa", "summarization")
DOLLY_PER_CATEGORY = 8
DOLLY_MAX_TOKENS = 224


@dataclass(frozen=True)
class Source:
    key: str
    kind: str
    how: str  # raw | gutenberg | ecfr | cap
    url: str
    licence: str
    quota: int
    split: str  # line | para : where a context may start


SOURCES: tuple[Source, ...] = (
    # code (64)
    Source("go-strings", "code", "raw", f"{GH}/golang/go/go1.23.0/src/strings/strings.go", "BSD-3-Clause (Go authors)", 7, "line"),
    Source("go-bufio", "code", "raw", f"{GH}/golang/go/go1.23.0/src/bufio/bufio.go", "BSD-3-Clause (Go authors)", 6, "line"),
    Source("lodash", "code", "raw", f"{GH}/lodash/lodash/4.17.21/lodash.js", "MIT (OpenJS Foundation and contributors)", 13, "line"),
    Source("requests-sessions", "code", "raw", f"{GH}/psf/requests/v2.32.3/src/requests/sessions.py", "Apache-2.0 (Kenneth Reitz and contributors)", 7, "line"),
    Source("click-core", "code", "raw", f"{GH}/pallets/click/8.1.7/src/click/core.py", "BSD-3-Clause (Pallets)", 6, "line"),
    Source("ripgrep-glob", "code", "raw", f"{GH}/BurntSushi/ripgrep/14.1.0/crates/globset/src/glob.rs", "MIT or Unlicense (Andrew Gallant)", 13, "line"),
    Source("redis-sds", "code", "raw", f"{GH}/redis/redis/7.2.4/src/sds.c", "BSD-3-Clause (Salvatore Sanfilippo and contributors)", 12, "line"),
    # prose (64): Project Gutenberg, public domain in the United States
    Source("pg944", "prose", "gutenberg", "https://www.gutenberg.org/cache/epub/944/pg944.txt", "public domain in the US (Project Gutenberg 944, Darwin, The Voyage of the Beagle)", 16, "para"),
    Source("pg205", "prose", "gutenberg", "https://www.gutenberg.org/cache/epub/205/pg205.txt", "public domain in the US (Project Gutenberg 205, Thoreau, Walden)", 16, "para"),
    Source("pg2641", "prose", "gutenberg", "https://www.gutenberg.org/cache/epub/2641/pg2641.txt", "public domain in the US (Project Gutenberg 2641, Forster, A Room with a View)", 16, "para"),
    Source("pg3300", "prose", "gutenberg", "https://www.gutenberg.org/cache/epub/3300/pg3300.txt", "public domain in the US (Project Gutenberg 3300, Smith, The Wealth of Nations)", 16, "para"),
    # legal-style (64): US federal regulations at a fixed date and US Supreme Court opinions, US government works
    Source("cfr16-310", "legal", "ecfr", "https://www.ecfr.gov/api/versioner/v1/full/2026-10-30/title-16.xml?part=310", "public domain (US government work, eCFR as of 2026-10-30)", 8, "para"),
    Source("cfr24-100", "legal", "ecfr", "https://www.ecfr.gov/api/versioner/v1/full/2026-10-28/title-24.xml?part=100", "public domain (US government work, eCFR as of 2026-10-28)", 8, "para"),
    Source("cfr29-825", "legal", "ecfr", "https://www.ecfr.gov/api/versioner/v1/full/2026-11-06/title-29.xml?part=825", "public domain (US government work, eCFR as of 2026-11-06)", 8, "para"),
    Source("cfr45-164", "legal", "ecfr", "https://www.ecfr.gov/api/versioner/v1/full/2026-11-05/title-45.xml?part=164", "public domain (US government work, eCFR as of 2026-11-05)", 8, "para"),
    Source("cap-us-550", "legal", "cap", "https://static.case.law/us/550", "public domain (US Supreme Court opinions, Caselaw Access Project static files)", 16, "para"),
    Source("cap-us-560", "legal", "cap", "https://static.case.law/us/560", "public domain (US Supreme Court opinions, Caselaw Access Project static files)", 16, "para"),
    # structured (64)
    Source("csv-recent-grads", "structured", "raw", f"{GH}/fivethirtyeight/data/4c1ff5e3aef1816ae04af63218015066e186c147/college-majors/recent-grads.csv", "CC-BY-4.0 (FiveThirtyEight)", 10, "line"),
    Source("csv-candy", "structured", "raw", f"{GH}/fivethirtyeight/data/4c1ff5e3aef1816ae04af63218015066e186c147/candy-power-ranking/candy-data.csv", "CC-BY-4.0 (FiveThirtyEight)", 6, "line"),
    Source("yaml-linguist", "structured", "raw", f"{GH}/github-linguist/linguist/v7.30.0/lib/linguist/languages.yml", "MIT (GitHub, Inc.)", 16, "line"),
    Source("json-vscode-ts", "structured", "raw", f"{GH}/microsoft/vscode/1.90.0/extensions/typescript-language-features/package.json", "MIT (Microsoft Corporation)", 16, "line"),
    Source("xml-commons-lang", "structured", "raw", f"{GH}/apache/commons-lang/rel/commons-lang-3.14.0/pom.xml", "Apache-2.0 (Apache Software Foundation)", 16, "line"),
)
CAP_CASES_PER_VOLUME = 4  # each chosen case gives quota / 4 contexts
DOLLY_URL = f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{DOLLY_REV}/databricks-dolly-15k.jsonl"
DOLLY_LICENCE = "CC-BY-SA-3.0 (Databricks, databricks-dolly-15k)"
MMLU_LICENCE = "MIT (Hendrycks et al., cais/mmlu)"
DS = "https://datasets-server.huggingface.co"


# ------------------------------------------------------------------------------------------- network (fetch only)

def get(url: str, headers: dict | None = None, pause: float = 0.5) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:  # an HTTP error raises: no retry
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    time.sleep(pause)
    return data


def save(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:  # never overwritten
        f.write(data)
    return {"file": path.name, "bytes": len(data), "sha256": cl.sha256_bytes(data)}


def fetch(data_dir: Path) -> None:
    for s in SOURCES:
        d = data_dir / s.key
        if (d / "FETCH.json").exists():
            print(f"have {s.key}")
            continue
        rec = {"source": s.key, "url": s.url, "licence": s.licence, "files": []}
        if s.how == "ecfr":
            rec["files"].append(save(d / "part.xml", get(s.url, {"Accept": "application/xml", "Accept-Encoding": "gzip"})))
        elif s.how == "cap":
            meta = get(s.url + "/CasesMetadata.json")
            rec["files"].append(save(d / "CasesMetadata.json", meta))
            names = sorted(c["file_name"] for c in json.loads(meta))
            chosen = sorted(random.Random(f"{SEED}:cap:{s.key}").sample(names, CAP_CASES_PER_VOLUME))
            for name in chosen:
                rec["files"].append(save(d / f"{name}.json", get(f"{s.url}/cases/{name}.json")))
        else:
            rec["files"].append(save(d / "source.txt", get(s.url)))
        (d / "FETCH.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"fetched {s.key}: {sum(f['bytes'] for f in rec['files'])} B")
    d = data_dir / "dolly"
    if not (d / "FETCH.json").exists():
        rec = {"source": "dolly", "url": DOLLY_URL, "licence": DOLLY_LICENCE,
               "files": [save(d / "dolly.jsonl", get(DOLLY_URL))]}
        (d / "FETCH.json").write_text(json.dumps(rec, indent=1) + "\n")
        print("fetched dolly")
    d = data_dir / "mmlu"
    if not (d / "FETCH.json").exists():
        info = json.loads(get("https://huggingface.co/api/datasets/cais/mmlu"))
        if info["sha"] != MMLU_REV:
            raise RuntimeError(f"cais/mmlu main is {info['sha']}, the plan pins {MMLU_REV}: stop")
        sizes = json.loads(get(f"{DS}/size?dataset=cais/mmlu"))
        subjects = {x["config"]: x["num_rows"] for x in sizes["size"]["splits"]
                    if x["split"] == "test" and x["config"] not in ("all", "auxiliary_train")}
        if len(subjects) != 57:
            raise RuntimeError(f"cais/mmlu has {len(subjects)} test subjects, want 57")
        files = []
        for subject in sorted(subjects):  # one file per subject, so a stopped fetch goes on where it stopped
            path = d / f"test-{subject}.jsonl"
            if not path.exists():
                rows = []
                for off in range(0, subjects[subject], MMLU_PAGE):
                    q = urllib.parse.urlencode({"dataset": "cais/mmlu", "config": subject, "split": "test",
                                                "offset": off, "length": MMLU_PAGE})
                    page = json.loads(get(f"{DS}/rows?{q}", pause=MMLU_PAUSE_S))
                    for row in page["rows"]:
                        if row["truncated_cells"]:
                            raise RuntimeError(f"{subject} {row['row_idx']}: truncated cells {row['truncated_cells']}")
                        rows.append({"subject": subject, "index": row["row_idx"], "row": row["row"]})
                if [r["index"] for r in rows] != list(range(subjects[subject])):
                    raise RuntimeError(f"{subject}: got rows {len(rows)} of {subjects[subject]} or out of order")
                save(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
            files.append({"file": path.name, "bytes": path.stat().st_size,
                          "sha256": cl.sha256_bytes(path.read_bytes()), "rows": subjects[subject]})
        rec = {"source": "mmlu", "dataset": "cais/mmlu", "revision": MMLU_REV, "licence": MMLU_LICENCE,
               "split": "test", "files": files}
        (d / "FETCH.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"fetched mmlu: {sum(f['rows'] for f in files)} rows in {len(files)} subjects")


# ------------------------------------------------------------------------------------------- text extraction (pure)

def gutenberg_body(text: str) -> str:
    text = text.replace("\r\n", "\n")
    start = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    end = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError("no Project Gutenberg start and end markers")
    return text[start.end():end.start()].strip("\n") + "\n"


def ecfr_text(xml: str) -> str:
    xml = re.sub(r"</(P|HEAD|FP|HED|PSPACE|EXTRACT|NOTE)>", "\n\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    text = html.unescape(xml)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def cap_text(case: dict) -> str:
    ops = case["casebody"]["opinions"]
    major = [o for o in ops if o.get("type") == "majority"] or ops
    lines = major[0]["text"].replace("\r\n", "\n").split("\n")  # CAP puts one paragraph per line
    return "\n\n".join(x.strip() for x in lines if x.strip()) + "\n"


def candidate_starts(text: str, split: str) -> list[int]:
    sep = "\n\n" if split == "para" else "\n"
    lo, hi = int(len(text) * SKIP_EDGE), int(len(text) * (1 - SKIP_EDGE))
    starts = [m.end() for m in re.finditer(re.escape(sep), text)]
    if split == "para":  # a paragraph starts with a non-space character
        return [s for s in starts if lo <= s < hi and text[s:s + 1].strip()]
    # a line may be indented (code, YAML, JSON); it must not be blank
    return [s for s in starts if lo <= s < hi and text[s:text.find("\n", s) if "\n" in text[s:] else len(text)].strip()]


def windows(text: str, split: str, count: int, targets: list[int], tok, rng: random.Random) -> list[tuple[int, int, int]]:
    """count non-overlapping windows (start, end, tokens): window i starts at a candidate start and is exactly
    targets[i] tokens long by tok (the model's tokenizer). Candidates are tried in a seeded random order."""
    cands = candidate_starts(text, split)
    rng.shuffle(cands)
    taken: list[tuple[int, int, int]] = []
    for s in cands:
        if len(taken) == count:
            break
        want = targets[len(taken)]
        enc = tok.encode(text[s:s + want * 16], add_special_tokens=False)
        if len(enc.ids) < want:
            continue
        e = s + enc.offsets[want - 1][1]
        if any(s < b and a < e for a, b, _ in taken):
            continue
        if len(text[s:e].strip()) < 100:
            continue
        taken.append((s, e, want))
    if len(taken) != count:
        raise ValueError(f"only {len(taken)} of {count} windows fit")
    return taken


def round_robin(groups: list[list[str]]) -> list[str]:
    out, i = [], 0
    while any(i < len(g) for g in groups):
        out.extend(g[i] for g in groups if i < len(g))
        i += 1
    return out


# ------------------------------------------------------------------------------------------- build

def load_source_text(data_dir: Path, s: Source) -> list[tuple[str, str]]:
    """(document name, text) pairs of one source."""
    d = data_dir / s.key
    rec = json.loads((d / "FETCH.json").read_text())
    for f in rec["files"]:
        if cl.sha256_bytes((d / f["file"]).read_bytes()) != f["sha256"]:
            raise ValueError(f"{d / f['file']}: sha256 differs from FETCH.json")
    if s.how == "raw":
        return [("source", (d / "source.txt").read_text(encoding="utf-8").replace("\r\n", "\n"))]
    if s.how == "gutenberg":
        return [("source", gutenberg_body((d / "source.txt").read_text(encoding="utf-8")))]
    if s.how == "ecfr":
        return [("part", ecfr_text((d / "part.xml").read_text(encoding="utf-8")))]
    docs = []
    for f in rec["files"]:
        if f["file"] != "CasesMetadata.json":
            docs.append((f["file"][:-5], cap_text(json.loads((d / f["file"]).read_text(encoding="utf-8")))))
    return sorted(docs)


def build(data_dir: Path, tokenizer: Path, out: Path, ids_out: Path) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    contexts: list[dict] = []
    sources_rec = []
    for s in SOURCES:
        docs = load_source_text(data_dir, s)
        per_doc = [s.quota // len(docs) + (1 if i < s.quota % len(docs) else 0) for i in range(len(docs))]
        n = 0
        for (name, text), count in zip(docs, per_doc):
            targets = [TARGETS[(n + i) % len(TARGETS)] for i in range(count)]
            rng = random.Random(f"{SEED}:ctx:{s.key}:{name}")
            for start, end, want in windows(text, s.split, count, targets, tok, rng):
                body = text[start:end]
                contexts.append({"id": f"{s.kind}-{s.key}-{n:02d}", "kind": s.kind, "source": s.key,
                                 "locator": {"doc": name, "char_start": start, "char_end": end, "text_of": s.how},
                                 "target_tokens": want, "text": body, "text_sha256": cl.sha256_text(body)})
                n += 1
        fetch_rec = json.loads((data_dir / s.key / "FETCH.json").read_text())
        sources_rec.append({"key": s.key, "kind": s.kind, "url": s.url, "licence": s.licence,
                            "files": fetch_rec["files"], "contexts": n})
    # chat: Dolly user turns, 8 per category, at most DOLLY_MAX_TOKENS tokens
    d = data_dir / "dolly"
    rec = json.loads((d / "FETCH.json").read_text())
    raw = (d / "dolly.jsonl").read_bytes()
    if cl.sha256_bytes(raw) != rec["files"][0]["sha256"]:
        raise ValueError("dolly.jsonl sha256 differs from FETCH.json")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    for cat in DOLLY_CATEGORIES:
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        random.Random(f"{SEED}:dolly:{cat}").shuffle(idx)
        took = 0
        for i in idx:
            r = rows[i]
            body = r["instruction"].strip() + (("\n\n" + r["context"].strip()) if r["context"].strip() else "")
            ntok = len(tok.encode(body, add_special_tokens=False).ids)
            if not 8 <= ntok <= DOLLY_MAX_TOKENS:
                continue
            contexts.append({"id": f"chat-dolly-{cat}-{took:02d}", "kind": "chat", "source": "dolly",
                             "locator": {"line": i, "fields": "instruction + context"}, "target_tokens": ntok,
                             "text": body, "text_sha256": cl.sha256_text(body)})
            took += 1
            if took == DOLLY_PER_CATEGORY:
                break
        if took != DOLLY_PER_CATEGORY:
            raise ValueError(f"dolly {cat}: only {took} records fit")
    sources_rec.append({"key": "dolly", "kind": "chat", "url": DOLLY_URL, "licence": DOLLY_LICENCE,
                        "files": rec["files"], "contexts": DOLLY_PER_CATEGORY * len(DOLLY_CATEGORIES)})
    for c in contexts:
        c["user_tokens"] = len(tok.encode(cl.context_message(c), add_special_tokens=False).ids)
    # MC: MMLU, 9 items per subject drawn with a seeded generator from the whole test split
    d = data_dir / "mmlu"
    rec = json.loads((d / "FETCH.json").read_text())
    if rec["revision"] != MMLU_REV:
        raise ValueError(f"mmlu fetched at {rec['revision']}, the plan pins {MMLU_REV}")
    mc = []
    for f in rec["files"]:
        raw = (d / f["file"]).read_bytes()
        if cl.sha256_bytes(raw) != f["sha256"]:
            raise ValueError(f"{f['file']} sha256 differs from FETCH.json")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        subject = rows[0]["subject"]
        for i in sorted(random.Random(f"{SEED}:mmlu:{subject}").sample(range(len(rows)), MMLU_PER_SUBJECT)):
            r = rows[i]
            row = r["row"]
            prompt = cl.mc_message(subject, row["question"], row["choices"])
            mc.append({"id": f"mmlu-{subject}-{i:04d}", "subject": subject, "index": i,
                       "row_sha256": cl.sha256_bytes(cl.canonical(row)), "gold": int(row["answer"]),
                       "prompt": prompt, "prompt_sha256": cl.sha256_text(prompt),
                       "user_tokens": len(tok.encode(prompt, add_special_tokens=False).ids)})
    sources_rec.append({"key": "mmlu", "kind": "mc", "url": "https://huggingface.co/datasets/cais/mmlu",
                        "revision": MMLU_REV, "licence": MMLU_LICENCE, "files": rec["files"], "items": len(mc)})
    # orders
    by_kind = []
    for k in cl.KINDS:
        ids = [c["id"] for c in contexts if c["kind"] == k]
        random.Random(f"{SEED}:order:{k}").shuffle(ids)
        by_kind.append(ids)
    by_subject = []
    for subj in sorted({m["subject"] for m in mc}):
        ids = [m["id"] for m in mc if m["subject"] == subj]
        random.Random(f"{SEED}:order:{subj}").shuffle(ids)
        by_subject.append(ids)
    main_c, main_m = round_robin(by_kind), round_robin(by_subject)
    plan = {"format": cl.PLAN_FORMAT, "seed": SEED,
            "tokenizer_sha256": cl.sha256_bytes(tokenizer.read_bytes()),
            "request": {"temperature": 0, "top_logprobs": cl.TOP_K, "context_tokens": cl.CONT_TOKENS,
                        "context_ignore_eos": True, "mc_max_tokens": cl.MC_MAX_TOKENS, "mc_ignore_eos": False,
                        "thinking": False, "endpoint": "/v1/chat/completions", "cache_salt": "fresh per request",
                        "context_instruction": cl.CONTEXT_INSTRUCTION, "mc_instruction": cl.MC_INSTRUCTION},
            "sources": sources_rec, "contexts": contexts, "mc": mc,
            "order": {"main_contexts": main_c, "floor_contexts": main_c[:cl.FLOOR_CONTEXTS],
                      "bridge_contexts": main_c[:cl.BRIDGE_CONTEXTS],
                      "main_mc": main_m, "floor_mc": main_m[:cl.FLOOR_MC]}}
    body = json.dumps(plan, indent=1, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    with out.open("xb") as f:
        f.write(body)
    plan_sha = cl.sha256_bytes(body)
    cl.Plan.load(out)  # every invariant the capture and the scorer rely on
    public = json.loads(body)
    for c in public["contexts"]:
        del c["text"]
    for m in public["mc"]:
        del m["prompt"]
    public["plan_sha256"] = plan_sha
    with ids_out.open("x") as f:
        f.write(json.dumps(public, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return {"plan": str(out), "plan_sha256": plan_sha, "ids": str(ids_out), "contexts": len(contexts),
            "mc": len(mc), "per_kind": {k: sum(1 for c in contexts if c["kind"] == k) for k in cl.KINDS}}


# ------------------------------------------------------------------------------------------- dry and self-test

def dry() -> None:
    print("SOURCES (kind, key, contexts, licence, url)")
    for s in SOURCES:
        print(f"  {s.kind:10s} {s.key:18s} {s.quota:3d}  {s.licence}\n      {s.url}")
    print(f"  {'chat':10s} {'dolly':18s} {DOLLY_PER_CATEGORY * len(DOLLY_CATEGORIES):3d}  {DOLLY_LICENCE}\n      {DOLLY_URL}")
    print(f"  {'mc':10s} {'mmlu':18s} {57 * MMLU_PER_SUBJECT:3d}  {MMLU_LICENCE} @ {MMLU_REV}")
    per = {k: sum(s.quota for s in SOURCES if s.kind == k) for k in cl.KINDS}
    per["chat"] = DOLLY_PER_CATEGORY * len(DOLLY_CATEGORIES)
    print(f"contexts per kind {per}, total {sum(per.values())}; MC items {57 * MMLU_PER_SUBJECT}")
    print(f"context lengths cycle {TARGETS} tokens; floor {cl.FLOOR_CONTEXTS} contexts + {cl.FLOOR_MC} MC; "
          f"bridge {cl.BRIDGE_CONTEXTS}")
    print("fetch writes DIR/<source>/ (one directory per source, never overwritten); build writes the plan and the "
          "ids file with mode x (never overwritten)")


class _WsTok:
    """Whitespace tokenizer with the two methods windows() uses; for the self-test only."""

    class _Enc:
        def __init__(self, ids, offsets):
            self.ids, self.offsets = ids, offsets

    def encode(self, text, add_special_tokens=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return self._Enc(list(range(len(spans))), spans)


def self_test() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    pg = "Title\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\nBody one.\r\n\r\nBody two.\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nLicence"
    check("gutenberg_body", gutenberg_body(pg), "Body one.\n\nBody two.\n")
    xml = "<DIV5><HEAD>PART 1&#x2014;X</HEAD><P>(a) One &amp; two.</P><P>(b) Three.</P></DIV5>"
    check("ecfr_text", ecfr_text(xml), "PART 1—X\n\n(a) One & two.\n\n(b) Three.\n")
    check("cap_text", cap_text({"casebody": {"opinions": [{"type": "dissent", "text": "D"},
                                                         {"type": "majority", "text": "P1.\nP2.\r\n\nP3."}]}}),
          "P1.\n\nP2.\n\nP3.\n")
    check("round_robin", round_robin([["a1", "a2", "a3"], ["b1"], ["c1", "c2"]]), ["a1", "b1", "c1", "a2", "c2", "a3"])
    # 200 paragraphs of 30 words: candidates lie in [5 %, 95 %) of the text, windows do not overlap and are exact
    text = "\n\n".join(" ".join(f"w{p}x{i}" for i in range(30)) for p in range(200)) + "\n"
    ws = windows(text, "para", 6, [40, 40, 40, 40, 40, 40], _WsTok(), random.Random(1))
    check("windows count", len(ws), 6)
    check("windows tokens", [len(text[a:b].split()) for a, b, _ in ws], [40] * 6)
    check("windows disjoint", all(b1 <= a2 or b2 <= a1 for i, (a1, b1, _) in enumerate(ws)
                                  for j, (a2, b2, _) in enumerate(ws) if i < j), True)
    check("windows inside edges", all(len(text) * SKIP_EDGE <= a < len(text) * (1 - SKIP_EDGE) for a, _, _ in ws), True)
    code = "".join(f"    line {i} of an indented block\n" if i % 3 else "\n" for i in range(100))
    starts = candidate_starts(code, "line")
    check("line starts: indented lines count, blank lines do not",
          all(code[s:s + 4] == "    " for s in starts) and len(starts) > 0, True)
    msg = cl.mc_message("high_school_physics", "Q?", ["w", "x", "y", "z"])
    check("mc_message", msg.endswith("Q?\nA. w\nB. x\nC. y\nD. z") and "high school physics" in msg, True)
    check("quotas", sum(s.quota for s in SOURCES) + DOLLY_PER_CATEGORY * len(DOLLY_CATEGORIES), 320)
    check("kinds 64 each", {k: sum(s.quota for s in SOURCES if s.kind == k) for k in cl.KINDS if k != "chat"},
          {"code": 64, "prose": 64, "legal": 64, "structured": 64})
    check("no CPython, no US Code", any("python/cpython" in s.url or "uscode" in s.url.lower() for s in SOURCES), False)
    for f in fails:
        print("FAIL", f)
    print("SELF-TEST PASS" if not fails else f"SELF-TEST FAIL ({len(fails)})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="mode")
    f = sub.add_parser("fetch")
    f.add_argument("--data", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--ids", type=Path, required=True)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.dry:
        dry()
        return 0
    if a.mode == "fetch":
        fetch(a.data)
        return 0
    if a.mode == "build":
        print(json.dumps(build(a.data, a.tokenizer, a.out, a.ids), indent=1))
        return 0
    ap.error("give fetch, build, --dry or --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())

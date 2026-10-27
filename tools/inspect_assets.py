import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "model_training" / "yolo_dataset"
RUN20 = ROOT / "runs" / "segment" / "data" / "model_training" / "runs" / "baseline_cpu_20epochs"
RUN1 = ROOT / "data" / "model_training" / "runs" / "baseline_cpu_1epochs"
SPLITS = ["train", "val", "test"]


def base_frame(name):
    stem = Path(name).stem
    return re.split(r"_jpg\.rf\.|\.rf\.", stem)[0]


def sequence_key(base):
    if re.fullmatch(r"\d+", base):
        return "numeric"
    m = re.match(r"^(.*?)-\d+-?$", base)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)_\d+$", base)
    if m:
        return m.group(1)
    return base


def show_text(path, keys=None):
    print("\n=== %s ===" % path)
    if not path.exists():
        print("MISSING")
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if keys is None or any(line.startswith(k + ":") for k in keys):
            print(line)


def main():
    print("project root:", ROOT)

    show_text(DS / "data.yaml")
    show_text(RUN20 / "args.yaml", ["model", "data", "epochs", "imgsz", "batch", "device", "task", "seed"])

    print("\n=== results.csv (last row of 20-epoch run) ===")
    rc = RUN20 / "results.csv"
    if rc.exists():
        with rc.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = {k.strip(): v.strip() for k, v in rows[-1].items()}
            print("epochs logged:", len(rows))
            for k, v in last.items():
                if "metrics" in k or k == "epoch":
                    print("  %s = %s" % (k, v))
    else:
        print("MISSING")

    print("\n=== weights ===")
    for p in [RUN20 / "weights" / "best.pt", RUN1 / "weights" / "best.pt", ROOT / "yolo11n-seg.pt"]:
        print("%s : %s" % (p.relative_to(ROOT), ("%.1f MB" % (p.stat().st_size / 1e6)) if p.exists() else "MISSING"))

    print("\n=== images / labels per split ===")
    bases = defaultdict(set)
    seqs = defaultdict(Counter)
    for s in SPLITS:
        imgs = sorted((DS / "images" / s).glob("*.jpg"))
        lbls = sorted((DS / "labels" / s).glob("*.txt"))
        print("%-5s images=%d labels=%d" % (s, len(imgs), len(lbls)))
        for p in imgs:
            b = base_frame(p.name)
            bases[b].add(s)
            seqs[s][sequence_key(b)] += 1

    cls = defaultdict(Counter)
    empty = Counter()
    for s in SPLITS:
        for p in (DS / "labels" / s).glob("*.txt"):
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            if not lines:
                empty[s] += 1
            for ln in lines:
                cls[s][ln.split()[0]] += 1
    print("\n=== instances per class id ===")
    for s in SPLITS:
        print(s, dict(sorted(cls[s].items())), "| empty label files:", empty[s])

    print("\n=== leakage check ===")
    multi = [b for b, ss in bases.items() if len(ss) > 1]
    print("unique base frames:", len(bases))
    print("base frames present in >1 split:", len(multi))
    all_seq = set()
    for s in SPLITS:
        all_seq |= set(seqs[s])
    print("source sequences (filename-derived):", len(all_seq))
    for k in sorted(all_seq):
        print("  %-16s train=%-4d val=%-4d test=%-4d" % (k, seqs["train"][k], seqs["val"][k], seqs["test"][k]))


if __name__ == "__main__":
    main()
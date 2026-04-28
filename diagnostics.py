from collections import Counter

def _safe_int(x):
    try:
        return int(x)
    except:
        return 0

def summarize_trades(folds):
    train, val, test = [], [], []
    for f in folds:
        train.append(_safe_int((f.get("train") or {}).get("trades", 0)))
        val.append(_safe_int((f.get("val") or {}).get("trades", 0)))
        test.append(_safe_int((f.get("test") or {}).get("trades", 0)))

    def mean(x): return sum(x)/len(x) if x else 0

    return {
        "mean": {
            "train": mean(train),
            "val": mean(val),
            "test": mean(test)
        },
        "zero_folds": {
            "train": sum(1 for x in train if x == 0),
            "val": sum(1 for x in val if x == 0),
            "test": sum(1 for x in test if x == 0),
        }
    }

def summarize_reasons(wf):
    reasons = wf.get("reasons", [])
    short = [r.split(":")[-1] for r in reasons]
    return dict(Counter(short))

def build_candidate_diagnostics(result):
    wf = result.get("walk_forward", {})
    folds = wf.get("split_results", {})

    # reconstruct folds
    fold_reports = []
    count = wf.get("fold_count", 0)

    for i in range(count):
        fold_reports.append({
            "train": (folds.get("train") or [{}])[min(i, len(folds.get("train", []))-1)],
            "val": (folds.get("val") or [{}])[min(i, len(folds.get("val", []))-1)],
            "test": (folds.get("test") or [{}])[min(i, len(folds.get("test", []))-1)],
        })

    return {
        "score": wf.get("score", 0),
        "passed": wf.get("passed", False),
        "trade_activity": summarize_trades(fold_reports),
        "top_fail_reasons": summarize_reasons(wf)
    }
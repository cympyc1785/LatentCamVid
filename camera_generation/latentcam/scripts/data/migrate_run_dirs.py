"""Migrate existing run folders to the unified naming + estimated config.yaml (Hydra port).

(1) results/<ts>_<exp_name>/  -> write config.yaml estimated from the exp_name suffix
    (map exp_name -> conf/experiment via load_cfg; fall back to base). Skips if config.yaml exists.
(2) main/wandb/(offline-)run-<ts>-<hash>/ -> rename to <ts>_<exp_name> matched by EXACT timestamp
    to a results/ folder, and copy that folder's config.yaml in. Unmatched / active-run / non-run
    entries are left untouched and reported.

--dry-run prints the plan without touching anything. --skip-ts <ts> excludes a live run's wandb dir.
Run: python scripts/migrate_run_dirs.py --dry-run   (then without --dry-run to apply)
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, glob, shutil, argparse, re
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'main'))
from hydra_cfg import load_cfg, save_cfg_yaml

LC = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
RESULTS = osp.join(LC, 'results')
WANDB = osp.join(LC, 'main', 'wandb')
TS_RE = re.compile(r'(\d{8}_\d{6})')


def build_expname_map():
    """exp_name (e.g. 'dl3dv_rolling') -> experiment config name ('config_rolling')."""
    import glob as _g
    m = {}
    for p in sorted(_g.glob(osp.join(LC, 'main', 'configs', 'config*.py'))):
        name = osp.basename(p)[:-3]
        if name in ('config_large', 'config_vae', 'config_vae32'):
            continue
        try:
            cfg, _ = load_cfg(name, overrides=[] if name != 'config' else [])
            m[cfg.exp_name] = name
        except Exception:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-ts', default='20260721_032652', help='live run ts to leave untouched')
    args = ap.parse_args()

    expmap = build_expname_map()
    print(f"exp_name->config map ({len(expmap)}): sample {list(expmap.items())[:3]}")

    # (1) results/ config.yaml
    res_by_ts = {}
    print("\n=== (1) results/ config.yaml ===")
    for d in sorted(glob.glob(osp.join(RESULTS, '*'))):
        if not osp.isdir(d):
            continue
        base = osp.basename(d); mts = TS_RE.match(base)
        if not mts:
            continue
        ts = mts.group(1); exp_name = base[len(ts) + 1:]
        res_by_ts.setdefault(ts, base)
        cfgpath = osp.join(d, 'config.yaml')
        cfgname = expmap.get(exp_name)
        if osp.isfile(cfgpath):
            print(f"  skip (exists): {base}"); continue
        if cfgname is None:
            print(f"  NO MATCH exp_name='{exp_name}' -> skip {base}"); continue
        if args.dry_run:
            print(f"  would write {base}/config.yaml  (from {cfgname})")
        else:
            _, cd = load_cfg(cfgname, overrides=[])
            save_cfg_yaml(cd, cfgpath); print(f"  wrote {base}/config.yaml  (from {cfgname})")

    # (2) wandb dirs -> rename by exact ts match to a results folder
    print("\n=== (2) main/wandb/ rename ===")
    renamed = unmatched = skipped = 0
    used = set()
    for d in sorted(glob.glob(osp.join(WANDB, '*run-*'))):
        if not osp.isdir(d):
            continue
        b = osp.basename(d); mts = TS_RE.search(b)
        ts = mts.group(1) if mts else None
        if ts == args.skip_ts:
            print(f"  SKIP active run: {b}"); skipped += 1; continue
        target = res_by_ts.get(ts)
        if target is None:
            unmatched += 1; continue
        newname = target
        dest = osp.join(WANDB, newname)
        i = 2
        while osp.exists(dest) or newname in used:
            newname = f"{target}__{i}"; dest = osp.join(WANDB, newname); i += 1
        used.add(newname)
        cfg_src = osp.join(RESULTS, target, 'config.yaml')
        if args.dry_run:
            print(f"  would rename {b} -> {newname}" + ("  (+config.yaml)" if osp.isfile(cfg_src) else ""))
        else:
            shutil.move(d, dest)
            if osp.isfile(cfg_src) and not osp.isfile(osp.join(dest, 'config.yaml')):
                shutil.copy(cfg_src, osp.join(dest, 'config.yaml'))
            print(f"  renamed {b} -> {newname}")
        renamed += 1
    print(f"\nwandb: renamed {renamed}, unmatched(left) {unmatched}, skipped-active {skipped}")


if __name__ == '__main__':
    main()

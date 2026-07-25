#!/usr/bin/env python3
"""
저장된 지점의 신호등 설정을 고친다 (경로는 건드리지 않는다).

  # 좌회전 화살표 있음으로 표시 → 7종 수집 대상이 된다
  python3 tools/edit_spot.py sig_004 --has-left

  # 신호등 ID 를 여러 개로 (한 진입로에 신호등이 둘 이상일 때)
  python3 tools/edit_spot.py sig_004 --tl C1256W000016,C1256W000018

  # 여러 지점에 한꺼번에
  python3 tools/edit_spot.py sig_001 sig_002 sig_003 --no-left

  # 전부
  python3 tools/edit_spot.py --all --states auto
"""

import argparse
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spot import schema     # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="지점의 신호등 설정 수정",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("spot_ids", nargs="*", help="수정할 spot_id 들")
    ap.add_argument("--all", action="store_true", help="모든 지점")
    ap.add_argument("--spots-dir", default=os.path.join(WS_ROOT, "spots"))
    ap.add_argument("--tl", default=None, help="신호등 ID (쉼표 구분) 로 교체")
    ap.add_argument("--add-tl", default=None, help="신호등 ID 추가 (쉼표 구분)")
    ap.add_argument("--has-left", dest="has_left", action="store_true", default=None)
    ap.add_argument("--no-left", dest="has_left", action="store_false")
    ap.add_argument("--note", default=None)
    ap.add_argument("--verified-no-tl", dest="verified", action="store_true",
                    default=None,
                    help="unknown 지점을 화면으로 확인했다고 표시 (화각 경고 억제)")
    ap.add_argument("--not-verified", dest="verified", action="store_false")
    ap.add_argument("--states", default=None,
                    help="수집 상태를 직접 지정 (쉼표 구분). 'auto' 면 자동판정으로 되돌린다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.all:
        targets = [s.spot_id for s in schema.load_all(args.spots_dir)]
    else:
        targets = args.spot_ids
    if not targets:
        print("수정할 지점을 지정할 것 (또는 --all).")
        return 1

    changed = 0
    for spot_id in targets:
        path = os.path.join(args.spots_dir, "%s.yaml" % spot_id)
        if not os.path.exists(path):
            print("%-12s 없음: %s" % (spot_id, path))
            continue
        spot = schema.load(path, strict=False)
        data = spot.data
        tl = data.setdefault("traffic_light", {})
        before = list(spot.states())

        if args.tl is not None:
            tl["ids"] = [t.strip() for t in args.tl.split(",") if t.strip()]
        if args.add_tl:
            ids = list(tl.get("ids") or [])
            for t in args.add_tl.split(","):
                t = t.strip()
                if t and t not in ids:
                    ids.append(t)
            tl["ids"] = ids
        if args.has_left is not None:
            tl["has_left"] = bool(args.has_left)
        if args.note is not None:
            data["note"] = args.note
        if args.verified is not None:
            data["verified_no_tl"] = bool(args.verified)
        if args.states is not None:
            if args.states.strip().lower() == "auto":
                tl.pop("states", None)
            else:
                names = [s.strip() for s in args.states.split(",") if s.strip()]
                from tl import states as tl_states
                for n in names:
                    tl_states.get(n)
                tl["states"] = names

        # 예전에 만든 파일에 없을 수 있으니 명시적으로 채워 넣는다
        tl.setdefault("has_left", False)

        after = list(schema.Spot(data).states())
        mark = "→" if before != after else " ="
        print("%-12s tl=%-34s left=%-5s  %s %d종%s"
              % (spot_id, ",".join(tl.get("ids") or []) or "-", tl.get("has_left"),
                 mark, len(before),
                 " → %d종" % len(after) if before != after else ""))
        if before != after:
            print("             %s  →  %s" % (", ".join(before), ", ".join(after)))

        if not args.dry_run:
            schema.save(data, args.spots_dir)
            changed += 1

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았다.")
    else:
        print("\n%d개 저장." % changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import csv, sys, pathlib

def main(path):
    p = pathlib.Path(path)
    bad = []
    with p.open("r", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, None)
        for i,row in enumerate(r, start=2):
            if len(row) != 4:
                bad.append(i)
    print(f"{p}: bad rows={len(bad)}")
    if bad[:10]:
        print("first 10 bad row numbers:", bad[:10])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/check_bad_rows.py inputs/SOLUSDT/SOLUSDT-ticks-2025-05.csv")
        sys.exit(2)
    main(sys.argv[1])

"""
generate index ranges according to the number of data points given query size
"""

import sys
import random
import argparse
import os

"""gen ranges [L,R] s.t. R - L + 1 = f * n
"""
def gen_iranges(n, q, f):
    if f != 17:
        f = 2 ** (-f)
        ranges = []
        cnt = int(n * f)
        for i in range(q):
            l = random.randint(0, n - cnt)
            r = l + cnt - 1
            ranges.append((l, r))
        return ranges
    else:
        # generate ranges from 0-10
        ranges = []
        f_i = 0
        for i in range(q):
            cnt =int (n * 2 ** (-f_i))
            l = random.randint(0, n - cnt)
            r = l + cnt - 1
            ranges.append((l, r))
            f_i = (f_i + 1) % 11
        return ranges
    
"""gen ranges (,L], [R,) s.t. L + n - R = f * n
"""
def gen_iranges_open(n, q, f):
    if f != 17:
        f = 2 ** (-f)
        ranges = []
        cnt = int(n * f)
        for i in range(q):
            r = random.randint(0, n)
            l = (r + cnt - 1) % n
            ranges.append((l, r))
        return ranges
    else:
        # generate ranges from 0-10
        ranges = []
        f_i = 0
        for i in range(q):
            cnt =int (n * 2 ** (-f_i))
            r = random.randint(0, n)
            l = (r + cnt - 1) % n
            ranges.append((l, r))
            f_i = (f_i + 1) % 11
        return ranges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of data points")
    parser.add_argument("--q", type=int, help="number of queries")
    # parser.add_argument("--f", type=int, help="fraction of data points to be queried")
    parser.add_argument("--o", type=str, help="output dir")
    parser.add_argument("--open", type=int, help="open or closed ranges", default=0)
    args = parser.parse_args()
    # f = args.f
    n = args.n
    q = args.q
    op = args.o
    out_dir = op
    os.makedirs(out_dir, exist_ok=True)
    for f in range(18):
        print(f"generating index ranges for {n} data points, {q} queries, fraction {f}, is_open {args.open}")
        if args.open:
            ranges = gen_iranges_open(n, q, f)
        else:
            ranges = gen_iranges(n, q, f)
        out_file = os.path.join(out_dir, f"{f}.bin")
        with open(out_file, "wb") as f:
            for l, r in ranges:
                f.write(l.to_bytes(4, "little"))
                f.write(r.to_bytes(4, "little"))
        print(f"index ranges saved to {out_file}")
        print(f"example ranges: {ranges[:5]}")

def printer(title: str, dt: dict, max_depth=3, depth=0):
    if depth == 0:
        print(f"{'=' * 10} {title} {'=' * 10}")

    if not isinstance(dt, dict):
        return print("\t" * depth + str(dt))

    max_len = max((len(str(a)) + 1 for a in dt.keys()))

    for k, v in dt.items():
        prefix = "\t" * depth + f"{str(k):<{max_len}}: "

        if isinstance(v, dict) and depth < max_depth - 1:
            print(prefix)
            printer("", v, max_depth=max_depth, depth=depth + 1)
        else:
            print(prefix + str(v))

    if depth == 0:
        print("=" * (22 + len(title)))
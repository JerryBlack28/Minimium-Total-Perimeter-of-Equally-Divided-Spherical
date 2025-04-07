import numpy as np
import argparse
from uctsearch1 import Node1, State1, uctsearch1, delete1
from uctsearch2 import Node2, State2, uctsearch2, delete2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--size", type=int, default=5, help="Input a number >= 5!")
    args = parser.parse_args()
    print(f"size = {args.size}")
    assert args.size >= 5, "Size must >= 5!"
    with open(f"./results/result{args.size}.txt", "w") as file:
        file.write("")
    best_result = None
    best_value = 1e18

    for i in range(3):
        root = Node1(State1(size=args.size, alpha=[args.size * 5, 1, 0]))
        tmp_result, tmp_value = uctsearch1(300000, root)
        if tmp_value < best_value:
            best_value, best_result = tmp_value, tmp_result
        delete1(root)

    voronoi = best_result
    with open(f"./results/result{args.size}.txt", "a") as file:
        file.write(f"Regions: {voronoi.sv.regions}\n")

    root = Node2(State2(vertices=np.array([list(pair) for pair in zip(*[voronoi.latv, voronoi.lonv])]),
                        regions=voronoi.sv.regions,
                        alpha=[args.size * 20, 1, args.size * 10]))
    best_result = uctsearch2(50000, root)
    with open(f"./results/result{args.size}.txt", "a") as file:
        file.write(f"Vertices: {np.round(best_result.state.vertices, 6).tolist()}\n")
        file.write(f"Total perimeter: {best_result.state.polygon.perimeter / 2}\n")
    delete2(root)
    best_result.state.polygon.savefig(f"{args.size}")

"""
ssp.py: model for the stable set problem（安定集合問題 / 最大独立集合問題）

目的（何を解くか）
- グラフ G=(V,E) に対し、互いに隣接しない頂点集合（stable set / independent set）を最大化する。
- 「隣接しない」＝ 選んだ頂点同士の間に辺が存在しない、という条件。
- 最大独立集合（Maximum Independent Set, MIS）は NP困難で、典型的な 0-1 組合せ最適化である。

安定集合（独立集合）の定義
- S ⊆ V が安定集合であるとは、任意の (i,j) ∈ E について i と j が同時に S に含まれないこと。
- つまり、S 内のどの2頂点も辺で結ばれていない。

数理最適化としての形式
- 変数: x_i ∈ {0,1}（頂点 i を選ぶなら1）
- 制約: 各辺 (i,j) に対して x_i + x_j ≤ 1（隣接頂点の同時選択禁止）
- 目的: Σ x_i を最大化（選ぶ頂点数最大）
- よって **0-1 混合整数線形計画（0-1 MIP / ILP）** として解く。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *


def ssp(V, E):
    """
    ssp -- stable set problem（最大独立集合）を解くMIPモデルを構築する

    Parameters:
        - V: 頂点集合（list）
        - E: 辺集合（list of (i,j), i<j を想定）

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """
    model = Model("ssp")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # x[i] ∈ {0,1}: 頂点 i を独立集合に入れるなら1
    # Math: x_i \in \{0,1\}\quad(\forall i\in V)
    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)" % i)

    model.update()

    # ------------------------------------------------------------
    # 制約（独立集合条件）
    # ------------------------------------------------------------
    # 各辺 (i,j) について、両端点を同時に選べない。
    # Math: x_i + x_j \le 1\quad(\forall (i,j)\in E)
    #
    # 直感：
    # - i と j が隣接しているなら、独立集合では両方同時に入れられない。
    # - この制約だけで「選ばれた頂点集合は独立集合」になる。
    for i, j in E:
        model.addConstr(x[i] + x[j] <= 1, "Edge(%s,%s)" % (i, j))

    # ------------------------------------------------------------
    # 目的関数（独立集合のサイズ最大化）
    # ------------------------------------------------------------
    # Math: \max \sum_{i\in V} x_i
    #
    # x_i は0/1なので、この総和は「選んだ頂点数」を表す。
    model.setObjective(quicksum(x[i] for i in V), GRB.MAXIMIZE)

    model.update()
    model.__data = x
    return model


import random


def make_data(n, prob):
    """
    make_data: ランダムグラフを作る（Erdos–Renyi的）

    Parameters:
        - n: 頂点数
        - prob: 各ペア (i<j) が辺になる確率

    Returns:
        - V: 頂点リスト [1..n]
        - E: 辺リスト（無向、i<j）
    """
    V = range(1, n + 1)
    E = [(i, j) for i in V for j in V if i < j and random.random() < prob]
    return V, E


if __name__ == "__main__":
    random.seed(1)

    # n=100, prob=0.5 はかなり密なグラフになりやすい（辺数が多くなる）。
    # 独立集合は小さくなりやすいが、MIPとしては制約が多くなって重くなる可能性がある。
    V, E = make_data(100, 0.5)

    model = ssp(V, E)
    model.optimize()

    # print は Python3 形式に統一
    print("Opt.value=", model.ObjVal)

    x = model.__data
    print("maximum stable set:")
    print([i for i in V if x[i].X > 0.5])

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: x_i \in \{0,1\}
# Math: x_i + x_j \le 1
# Math: \max \sum_{i\in V} x_i

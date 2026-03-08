"""
gcp.py: model for the graph coloring problem（グラフ彩色問題: GCP）

目的（何を解くか）
- グラフ G=(V,E) の各頂点に色を割り当てる。
- 隣接する頂点（辺で結ばれた頂点）は同じ色を持てない。
- 使う色数を最小化したい（彩色数の最小化）だが、彩色数（chromatic number）の厳密最小化はNP困難である。
- そこで「色数の上界 K を与え、0..K-1 の範囲の色を使って彩色しつつ、実際に使った色数を最小化する」MIPを解く。

入力の意味
- V: 頂点集合（list）
- E: 辺集合（list of (i,j)）
- K: 色数の上界（最大でK色まで使える）

数理最適化としての形式
- x[i,k] ∈ {0,1}: 頂点 i に色 k を割り当てるなら1
- y[k] ∈ {0,1}: 色 k を「使用する」なら1
- 制約・目的は線形（ただし SOS版はSOS制約が入る）
- よって **0-1 MIP（MILP）** として解く

このファイルの3モデル
1) gcp
   - 基本形。使用色 y[k] を導入して Σ y[k] を最小化する。
2) gcp_low
   - 対称性（色ラベルの置換）を壊すため、y[0] ≥ y[1] ≥ ... を追加して「低い色番号から使う」形に固定する。
3) gcp_sos
   - gcp_low に加え、各頂点 i の {x[i,0],...,x[i,K-1]} に SOS1 を付ける。
   - すでに Σ x[i,k] = 1 があるので数学的には冗長だが、探索が速くなる場合がある。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。
- 以降、レビュー本文は LaTeX（$...$ / $$...$$）だが、ソースコード内は Mathover 形式。

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *


def gcp(V, E, K):
    """
    gcp -- model for minimizing the number of colors in a graph

    Parameters:
        - V: 頂点集合
        - E: 辺集合
        - K: 色数上界

    Returns:
        - model: Gurobi Model（解ける状態）
    """
    model = Model("gcp")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # y[k] ∈ {0,1}: 色kを使うなら1
    # Math: y_k \in \{0,1\}\quad(\forall k\in\{0,\dots,K-1\})
    #
    # x[i,k] ∈ {0,1}: 頂点iに色kを割当てるなら1
    # Math: x_{ik} \in \{0,1\}\quad(\forall i\in V,\ \forall k\in\{0,\dots,K-1\})
    x, y = {}, {}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)" % k)
        for i in V:
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))

    model.update()

    # ------------------------------------------------------------
    # 制約（1）各頂点はちょうど1色
    # ------------------------------------------------------------
    # Math: \sum_{k=0}^{K-1} x_{ik} = 1\quad(\forall i\in V)
    #
    # これで「必ずどれかの色を選ぶ」かつ「複数色を同時に持たない」を保証する。
    for i in V:
        model.addConstr(quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)

    # ------------------------------------------------------------
    # 制約（2）隣接頂点の同色禁止 + 色使用yへの連結
    # ------------------------------------------------------------
    # 本来の同色禁止は
    #   x_{ik} + x_{jk} <= 1
    # で十分である。
    #
    # ただし「使用色数」を数えるため y[k] を導入しているので、
    # x が立つなら y も立つように連結する必要がある。
    #
    # この実装は次を使う：
    # Math: x_{ik}+x_{jk}\le y_k\quad(\forall (i,j)\in E,\ \forall k)
    #
    # 効果：
    # - y_k=0 なら、辺で結ばれた頂点同士には色kを割当てられない（実質色kが使えない）
    # - y_k=1 なら、x_{ik}+x_{jk}<=1 となり同色禁止として機能する
    #
    # 注意（重要）：
    # - この形は「辺が存在する頂点ペア」に対してしか x と y を連結しない。
    # - もし孤立点（次数0）があると、その頂点は辺制約に登場しないため、
    #   y_k=0 でも x_{ik}=1 が可能になりうる（yに連結されない）。
    # - ランダムグラフでは孤立点が出ることがあるので、堅牢にするなら
    #   x_{ik} <= y_k を全 i,k に対して明示するのが安全。
    for i, j in E:
        for k in range(K):
            model.addConstr(
                x[i, k] + x[j, k] <= y[k], "NotSameColor(%s,%s,%s)" % (i, j, k)
            )

    # ------------------------------------------------------------
    # 目的関数：使用色数の最小化
    # ------------------------------------------------------------
    # Math: \min \sum_{k=0}^{K-1} y_k
    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def gcp_low(V, E, K):
    """
    gcp_low -- low-index colors（低い色番号から使う）版

    対称性（色ラベルの置換）を壊す理由
    - 彩色は「色のラベル」を入れ替えても同じ解が大量に存在する（対称性）。
    - MIPは同型解を何度も探索して遅くなりやすい。
    - y[0] >= y[1] >= ... を入れることで、「使う色は0,1,2,...の順に詰める」ことを強制し、
      対称性を大きく削る。
    """
    model = Model("gcp - low colors")

    x, y = {}, {}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)" % k)
        for i in V:
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))

    model.update()

    # Math: \sum_{k=0}^{K-1} x_{ik} = 1\quad(\forall i\in V)
    for i in V:
        model.addConstr(quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)

    # Math: x_{ik}+x_{jk}\le y_k\quad(\forall (i,j)\in E,\ \forall k)
    for i, j in E:
        for k in range(K):
            model.addConstr(
                x[i, k] + x[j, k] <= y[k], "NotSameColor(%s,%s,%s)" % (i, j, k)
            )

    # 対称性破壊：低い色番号から使う
    # Math: y_k \ge y_{k+1}\quad(\forall k\in\{0,\dots,K-2\})
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], "LowColor(%s)" % k)

    # Math: \min \sum_{k=0}^{K-1} y_k
    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def gcp_sos(V, E, K):
    """
    gcp_sos -- gcp_low + SOS1 版

    SOS1 を付ける意図
    - 各頂点 i は色を1つだけ選ぶ（選択構造）。
    - その構造を SOS1 として渡すと、ソルバが分枝や伝播を効率化できる場合がある。
    - すでに Σ x[i,k] = 1 があるので冗長だが、性能比較用として意味がある。
    """
    model = Model("gcp - sos constraints")

    x, y = {}, {}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)" % k)
        for i in V:
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))

    model.update()

    for i in V:
        # Math: \sum_{k=0}^{K-1} x_{ik} = 1\quad(\forall i\in V)
        model.addConstr(quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)

        # SOS1: 高々1変数だけ非ゼロ
        # Math: \mathrm{SOS1}(x_{i0},x_{i1},\dots,x_{i,K-1})
        model.addSOS(1, [x[i, k] for k in range(K)])

    # Math: x_{ik}+x_{jk}\le y_k\quad(\forall (i,j)\in E,\ \forall k)
    for i, j in E:
        for k in range(K):
            model.addConstr(
                x[i, k] + x[j, k] <= y[k], "NotSameColor(%s,%s,%s)" % (i, j, k)
            )

    # Math: y_k \ge y_{k+1}\quad(\forall k\in\{0,\dots,K-2\})
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], "LowColor(%s)" % k)

    # Math: \min \sum_{k=0}^{K-1} y_k
    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


import random


def make_data(n, prob):
    """
    make_data: prepare data for a random graph

    Parameters:
        - n: number of vertices
        - prob: probability of existence of an edge for each pair (i<j)

    Returns:
        - V: vertices [1..n]
        - E: edges list
    """
    V = range(1, n + 1)
    E = [(i, j) for i in V for j in V if i < j and random.random() < prob]
    return V, E


if __name__ == "__main__":
    random.seed(1)
    V, E = make_data(20, 0.5)
    K = 10
    print("n,K=", len(V), K)

    # 例：低い色番号から使う対称性破壊版
    model = gcp_low(V, E, K)
    model.optimize()
    print("Opt.value=", model.ObjVal)

    # 解の復元：各頂点の色 k を抜き出す
    x = model.__data
    color = {}
    for i in V:
        for k in range(K):
            if x[i, k].X > 0.5:
                color[i] = k
    print("colors:", color)

    # ------------------------------------------------------------
    # ベンチマーク（元コードはPython2前提の記法があるので注意）
    # ------------------------------------------------------------
    # ここは「モデル比較」のための実験ループ。
    # - size を変えてランダムグラフを生成
    # - 3モデルを解いて平均時間を取る
    #
    # Python3で動かすなら：
    # - time.clock() -> time.perf_counter()
    # - dict.has_key -> (key in dict)
    # - print の整形
    import time, sys

    setParam(GRB.Param.Threads, 1)
    models = [gcp, gcp_low, gcp_sos]
    cpu = {}
    N = 25
    print("#size\t%s\t%s\t%s" % tuple(m.__name__ for m in models))

    for size in range(250):
        print(size, "\t", end="")
        K = size
        for prob in [0.1]:
            for m in models:
                name = m.__name__

                prev_key = (name, size - 1, prob)
                cur_key = (name, size, prob)

                if (prev_key not in cpu) or (
                    cpu[prev_key] != "-" and cpu[prev_key] < 100
                ):
                    cpu[cur_key] = 0.0
                    for t in range(N):
                        tinit = time.perf_counter()
                        random.seed(t)
                        V, E = make_data(size, prob)
                        model = m(V, E, K)
                        model.Params.OutputFlag = 0
                        model.optimize()
                        assert model.ObjVal >= 0 and model.ObjVal <= K
                        tend = time.perf_counter()
                        cpu[cur_key] += tend - tinit
                    cpu[cur_key] /= N
                else:
                    cpu[cur_key] = "-"

                print(cpu[cur_key], "\t", end="")
        print()
        sys.stdout.flush()

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: x_{ik} \in \{0,1\}
# Math: y_k \in \{0,1\}
# Math: \sum_{k=0}^{K-1} x_{ik} = 1
# Math: x_{ik}+x_{jk}\le y_k
# Math: \min \sum_{k=0}^{K-1} y_k
# Math: y_k \ge y_{k+1}
# Math: \mathrm{SOS1}(x_{i0},x_{i1},\dots,x_{i,K-1})

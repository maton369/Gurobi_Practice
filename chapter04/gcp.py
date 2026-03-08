"""
gcp.py: model for the graph coloring problem（グラフ彩色問題: GCP）

目的（何を解くか）
- グラフ G=(V,E) の各頂点に色を割り当てる。
- 隣接する頂点（辺で結ばれた頂点）は同じ色にできない。
- 使う色数（chromatic number に相当）を最小化したいが、これは NP 困難。
- そこで「色数の上界 K を与え、その範囲内で使用色数を最小化する」MIPを解く。

入力の意味
- V: 頂点集合
- E: 辺集合（無向、(i,j) で i<j を想定）
- K: 色数の上界（最大で K 色まで使ってよい）

数理最適化としての形式
- x[i,k] ∈ {0,1}: 頂点 i に色 k を割り当てるなら1
- y[k] ∈ {0,1}: 色 k を「使用する」なら1（少なくとも1頂点に割り当てたら使用）
- 制約・目的は線形（ただしSOS版は SOS1 制約を追加）
- よって **0-1 MIP（MILP）** として解く。

本ファイルの3モデル
1) gcp     : 基本モデル（色使用 y を入れ、Σ y を最小化）
2) gcp_low : 対称性（色ラベルの入れ替え）を壊すため y[0]≥y[1]≥... を追加
3) gcp_sos : gcp_low に加え、各頂点の色選択に SOS1 を与える（等式と同義だがソルバに有利な場合がある）

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。
"""

from gurobipy import *


def gcp(V, E, K):
    """
    gcp -- model for minimizing the number of colors in a graph

    Parameters:
        - V: 頂点集合
        - E: 辺集合
        - K: 色数の上界

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """
    model = Model("gcp")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # y[k] ∈ {0,1}: 色 k を使用するなら1
    # Math: y_k \in \{0,1\}\quad(\forall k\in\{0,\dots,K-1\})
    #
    # x[i,k] ∈ {0,1}: 頂点 i に色 k を割り当てるなら1
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
    # これにより「必ずどれかの色を選ぶ」かつ「複数色を同時に持たない」が保証される。
    for i in V:
        model.addConstr(
            quicksum(x[i, k] for k in range(K)) == 1,
            "AssignColor(%s)" % i,
        )

    # ------------------------------------------------------------
    # 制約（2）隣接頂点は同色禁止
    # ------------------------------------------------------------
    # 本来の彩色制約は
    #   x_{ik} + x_{jk} <= 1
    # で十分（同じ色 k を両方が取れない）。
    #
    # ただしこのモデルは y_k を導入して「使う色数」を数えるので、
    # x の存在を y に連結する（色を使ったら y=1 になる）必要がある。
    #
    # ここでは次を課している：
    # Math: x_{ik} + x_{jk} \le y_k\quad(\forall (i,j)\in E,\ \forall k)
    #
    # この形だと：
    # - もし y_k=0 なら左辺も 0 でないといけない → その色 k は誰にも割り当てられない
    # - もし y_k=1 なら x_{ik}+x_{jk} <= 1 に戻る → 同色禁止
    #
    # つまり「同色禁止」と「yへの連結」を同時に表現している。
    #
    # 注意：
    # - 連結としては x_{ik} <= y_k を各 i,k に入れるのが定石だが、
    #   ここでは辺制約の中に組み込んでいるため、孤立点（degree=0）の頂点がいると
    #   x_{ik} <= y_k が不足しうる点に注意（この実装ではランダムグラフで孤立が出ることもある）。
    #   ただし AssignColor により必ずどこかの色を取るので、孤立点がある場合は yが立つ誘因が弱くなる。
    #
    # 一般に堅牢にするなら x_{ik} <= y_k を明示するのが安全。
    for i, j in E:
        for k in range(K):
            model.addConstr(
                x[i, k] + x[j, k] <= y[k],
                "NotSameColor(%s,%s,%s)" % (i, j, k),
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
    gcp_low -- use colors with low indices（低い色番号から使うように対称性を破壊）

    ポイント
    - 彩色問題は「色ラベルの置換」が大量に存在し、MIP探索が遅くなりやすい。
    - y[0] >= y[1] >= ... >= y[K-1] を入れることで、
      “使う色は低い番号から” という規則を強制し、同型解を減らす。
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

    # 対称性破壊（低い色番号を優先）
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
    gcp_sos -- gcp_low + SOS1 を付与した版

    SOS1（Special Ordered Set type 1）
    - 指定した変数集合のうち、非ゼロになれるのは高々1つ、という制約。
    - ここでは各頂点 i について {x[i,0],...,x[i,K-1]} に SOS1 を追加する。
    - 既に Σ x[i,k] = 1 があるので数学的には冗長だが、
      ソルバが「選択」構造を理解して探索が速くなるケースがある。
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

        # Math: \text{SOS1}(x_{i0},x_{i1},\dots,x_{i,K-1})
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
    make_data: ランダムグラフを作る（無向、i<j）

    Parameters:
        - n: 頂点数
        - prob: 各ペア (i<j) が辺になる確率

    Returns:
        - V: 頂点リスト [1..n]
        - E: 辺リスト
    """
    V = range(1, n + 1)
    E = [(i, j) for i in V for j in V if i < j and random.random() < prob]
    return V, E


if __name__ == "__main__":
    random.seed(1)
    V, E = make_data(20, 0.5)
    K = 10
    print("n,K=", len(V), K)

    # 例：対称性破壊ありの gcp_low を解く
    model = gcp_low(V, E, K)
    model.optimize()
    print("Opt.value=", model.ObjVal)

    # 解の復元：各頂点がどの色 k を取ったか
    x = model.__data
    color = {}
    for i in V:
        for k in range(K):
            if x[i, k].X > 0.5:
                color[i] = k
    print("colors:", color)

    # ------------------------------------------------------------
    # 速度比較実験（元コードをPython3寄せに軽く整形したいならここも修正対象）
    # ------------------------------------------------------------
    import time, sys

    setParam(GRB.Param.Threads, 1)
    models = [gcp, gcp_low, gcp_sos]
    cpu = {}
    N = 25
    print("#size\t%s\t%s\t%s" % tuple(m.__name__ for m in models))

    # 注意：
    # - このループは size を 0..249 まで回しており、かなり重い。
    # - time.clock() はPython3で非推奨なので、time.perf_counter() の方が安全。
    for size in range(250):
        print(size, "\t", end="")
        K = size
        for prob in [0.1]:
            for m in models:
                name = m.__name__
                # Python2の has_key は Python3 では使えないため、ここは dict in を使うのが正しい。
                # 元コードの意図をコメントとして残す。
                key_prev = (name, size - 1, prob)
                if (key_prev not in cpu) or (
                    cpu[key_prev] != "-" and cpu[key_prev] < 100
                ):
                    cpu[(name, size, prob)] = 0.0
                    for t in range(N):
                        tinit = time.perf_counter()
                        random.seed(t)
                        V, E = make_data(size, prob)
                        model = m(V, E, K)
                        model.Params.OutputFlag = 0
                        model.optimize()
                        assert model.ObjVal >= 0 and model.ObjVal <= K
                        tend = time.perf_counter()
                        cpu[(name, size, prob)] += tend - tinit
                    cpu[(name, size, prob)] /= N
                else:
                    cpu[(name, size, prob)] = "-"
                print(cpu[(name, size, prob)], "\t", end="")
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
# Math: \text{SOS1}(x_{i0},x_{i1},\dots,x_{i,K-1})

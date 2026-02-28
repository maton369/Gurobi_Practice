"""
kcenter.py: model for solving the k-center problem（k-センター問題）

目的（何を解くか）
- 候補施設集合 J から「ちょうど k 個」施設を選ぶ。
- 各顧客（頂点）i ∈ I は、選ばれた施設のうち1つに割り当てられる。
- そのとき「顧客→割当施設までの距離（コスト）」の最大値を最小化する。
  つまり “最悪ケース距離” を最小化するミニマックス問題である。

k-median との違い（重要）
- k-median: 総コスト（平均的な良さ）を最小化
- k-center: 最大コスト（最悪ケース）を最小化
  → 公平性/サービス品質の下限保証に向く（どの顧客も遠すぎないようにする）。

数理最適化としての形式
- 施設選択: 二値変数 y[j]
- 割当: 二値変数 x[i,j]
- 最大距離の上界: 連続変数 z
- 制約・目的は線形（ただし min-max を z で線形化）
- よって **0-1混合整数線形計画（0-1 MIP / MILP）** として解ける。

数式コメント方針（ユーザー指定）
- ソースコード内の数式コメントは Mathover に合わせて `# Math: <LaTeX>` の1行形式に統一する。
- レビュー本文はLaTeXだが、ここはソースコードなので Mathover 形式。

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *


def kcenter(I, J, c, k):
    """
    kcenter -- minimize the maximum travel cost from customers to k facilities.

    Parameters:
        - I: 顧客集合
        - J: 候補施設集合
        - c[i,j]: 顧客 i を施設 j が担当するときのコスト（距離）
        - k: 選ぶ施設数（ちょうど k）

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """

    # ------------------------------------------------------------
    # モデル作成
    # ------------------------------------------------------------
    model = Model("k-center")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # z: 「最大距離（最大コスト）」の上界を表す連続変数
    # 目的関数で z を最小化することで、max 距離を最小化する。
    # Math: z \in \mathbb{R}
    z = model.addVar(vtype="C", name="z")

    # y[j] ∈ {0,1}: 施設 j を選ぶなら1
    # Math: y_j \in \{0,1\}\quad(\forall j\in J)
    #
    # x[i,j] ∈ {0,1}: 顧客 i を施設 j に割り当てるなら1
    # Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 制約
    # ------------------------------------------------------------
    for i in I:
        # (1) 割当制約: 各顧客 i は必ず1つの施設に割り当てられる
        # Math: \sum_{j\in J} x_{ij} = 1\quad(\forall i\in I)
        model.addConstr(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)

        # (2) k-center の“max”を線形化する制約（z を上界にする）
        #
        # 本来やりたいのは
        #   max_i ( 顧客 i の割当距離 ) を最小化
        #
        # 顧客 i の割当距離は、x[i,j] が 0/1 で「どれか1つだけ1」になるので
        #   sum_j c[i,j] * x[i,j]
        # がその顧客の距離（選ばれた施設への距離）になる。
        #
        # そして「すべての顧客 i に対し、その距離 ≤ z」とすれば、
        # z は “全顧客の割当距離の最大値” の上界になる。
        #
        # Math: \sum_{j\in J} c_{ij}x_{ij} \le z\quad(\forall i\in I)
        model.addConstr(
            quicksum(c[i, j] * x[i, j] for j in J) <= z,
            "Max_x(%s)" % i,
        )

        # (3) 連結（強化）制約: 選ばれていない施設には割当できない
        # Math: x_{ij}\le y_j\quad(\forall i\in I,\ \forall j\in J)
        for j in J:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    # (4) 施設数制約: ちょうど k 個の施設を選ぶ
    # Math: \sum_{j\in J} y_j = k
    model.addConstr(quicksum(y[j] for j in J) == k, "Facilities")

    # ------------------------------------------------------------
    # 目的関数
    # ------------------------------------------------------------
    # z を最小化すると、全顧客の割当距離の最大値（最悪ケース）が最小になる。
    # Math: \min z
    model.setObjective(z, GRB.MINIMIZE)

    model.update()
    model.__data = x, y
    return model


# ------------------------------------------------------------
# データ生成（平面上のランダム点→距離コスト）
# ------------------------------------------------------------
import math
import random


def distance(x1, y1, x2, y2):
    # ユークリッド距離
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_data(n, m, same=True):
    """
    n: 顧客数
    m: 施設候補数
    same=True なら I と J が同じ点集合（インデックス領域も重なりやすい）
    same=False なら 顧客点と施設候補点を別集合にする
    """
    if same == True:
        I = range(n)
        J = range(m)
        x = [random.random() for i in range(max(m, n))]  # positions
        y = [random.random() for i in range(max(m, n))]
    else:
        I = range(n)
        J = range(n, n + m)
        x = [random.random() for i in range(n + m)]
        y = [random.random() for i in range(n + m)]

    # コスト行列 c[i,j]
    c = {}
    for i in I:
        for j in J:
            c[i, j] = distance(x[i], y[i], x[j], y[j])

    return I, J, c, x, y


if __name__ == "__main__":
    random.seed(67)
    n = 100
    m = n
    I, J, c, x_pos, y_pos = make_data(n, m, same=True)

    # 選択施設数
    k = 10

    model = kcenter(I, J, c, k)
    model.optimize()

    EPS = 1.0e-6
    x, y = model.__data

    # 割当（x[i,j]=1）を抽出
    edges = [(i, j) for (i, j) in x if x[i, j].X > EPS]

    # 選ばれた施設（y[j]=1）を抽出
    facilities = [j for j in y if y[j].X > EPS]

    # print は Python3 形式に統一
    print("Optimal value=", model.ObjVal)  # ここでは z の最小値（= 最悪距離の最小値）
    print("Selected facilities:", facilities)
    print("Edges:", edges)
    print("max c:", max([c[i, j] for (i, j) in edges]))

    # ------------------------------------------------------------
    # 可視化（networkx + matplotlib）
    # ------------------------------------------------------------
    try:
        import networkx as NX
        import matplotlib.pyplot as P

        P.clf()
        G = NX.Graph()

        facilities = set(j for j in J if y[j].X > EPS)
        other = set(j for j in J if j not in facilities)
        client = set(i for i in I if i not in facilities and i not in other)

        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        G.add_nodes_from(other)

        for i, j in edges:
            G.add_edge(i, j)

        position = {}
        for i in range(len(x_pos)):
            position[i] = (x_pos[i], y_pos[i])

        NX.draw(G, position, with_labels=False, node_color="w", nodelist=facilities)
        NX.draw(
            G, position, with_labels=False, node_color="c", nodelist=other, node_size=50
        )
        NX.draw(
            G,
            position,
            with_labels=False,
            node_color="g",
            nodelist=client,
            node_size=50,
        )
        P.show()

    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")

# ------------------------------------------------------------
# このモデルのまとめ（Mathover用の数式コメント）
# ------------------------------------------------------------
# Math: y_j \in \{0,1\}\quad(\forall j\in J)
# Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
# Math: \sum_{j\in J} x_{ij} = 1\quad(\forall i\in I)
# Math: \sum_{j\in J} c_{ij}x_{ij} \le z\quad(\forall i\in I)
# Math: x_{ij}\le y_j\quad(\forall i\in I,\ \forall j\in J)
# Math: \sum_{j\in J} y_j = k
# Math: \min z

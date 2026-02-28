"""
gpp.py: model for the graph partitioning problem（グラフ分割問題: GPP）

目的（何を解くか）
- 頂点集合 V を、サイズが等しい 2 つのグループ（|V|/2 と |V|/2）に分割する。
- 辺集合 E のうち「異なるグループをまたぐ辺（cut edge）」の本数（または重み）を最小化する。
- これは balanced minimum cut の基本形で、NP困難な組合せ最適化問題である。

本ファイルにある4つの定式化
1) gpp      : 標準の 0-1 MIP（カット辺指示変数 y を導入）
2) gpp_qo   : 二次目的（x_i(1-x_j)+x_j(1-x_i)）で cut を表現する MIQP
3) gpp_qo_ps: (x_i-x_j)^2 を用いた半正定値（実質凸）っぽい形の MIQP
4) gpp_soco : 二次制約を SOC（Second Order Cone）で扱う形のモデル（教育用の変形）

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。

"""

from gurobipy import *


def gpp(V, E):
    """
    gpp -- standard MIP model for graph partitioning

    Parameters:
        - V: 頂点集合（list）
        - E: 辺集合（list of (i,j) with i<j を想定）

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """
    model = Model("gpp")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # x[i] ∈ {0,1}: 頂点 i の所属グループを表す（二部に分ける）
    # - x[i]=1 をグループA、x[i]=0 をグループBと解釈する
    # Math: x_i \in \{0,1\}\quad(\forall i\in V)
    x = {}

    # y[i,j] ∈ {0,1}: 辺 (i,j) がカット（異なるグループ間）なら1
    # Math: y_{ij} \in \{0,1\}\quad(\forall (i,j)\in E)
    y = {}

    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)" % i)
    for i, j in E:
        y[i, j] = model.addVar(vtype="B", name="y(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 制約（1）バランス制約（等分割）
    # ------------------------------------------------------------
    # ちょうど半分の頂点がグループA（x=1）になるようにする。
    # Math: \sum_{i\in V} x_i = |V|/2
    #
    # 注意:
    # - Python2の整数除算だと len(V)/2 は切り捨てになる。
    # - |V| が偶数であることが前提。
    model.addConstr(quicksum(x[i] for i in V) == len(V) / 2, "Partition")

    # ------------------------------------------------------------
    # 制約（2）カット辺の定義（|x_i - x_j| を y_ij で上から抑える）
    # ------------------------------------------------------------
    # x_i, x_j が0/1のとき、
    # - 同じグループなら x_i - x_j = 0
    # - 異なるグループなら x_i - x_j = ±1
    #
    # よって y_ij が 1 であれば異なるグループ、0 であれば同一グループ、という対応を作りたい。
    #
    # 典型的には次の2本で |x_i - x_j| ≤ y_ij を実現する。
    # Math: x_i - x_j \le y_{ij}\quad(\forall (i,j)\in E)
    # Math: x_j - x_i \le y_{ij}\quad(\forall (i,j)\in E)
    #
    # これにより
    # - x_i=x_j のとき y_ij は 0 でもOK
    # - x_i≠x_j のとき y_ij は 1 でないと不可能（±1 ≤ y）
    for i, j in E:
        model.addConstr(x[i] - x[j] <= y[i, j], "Edge(%s,%s)" % (i, j))
        model.addConstr(x[j] - x[i] <= y[i, j], "Edge(%s,%s)" % (j, i))

    # ------------------------------------------------------------
    # 目的関数：カット辺の本数を最小化
    # ------------------------------------------------------------
    # Math: \min \sum_{(i,j)\in E} y_{ij}
    model.setObjective(quicksum(y[i, j] for (i, j) in E), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def gpp_qo(V, E):
    """
    gpp_qo -- quadratic optimization model (MIQP)

    カット辺を二次式で直接数え上げる形。
    x_i(1-x_j)+x_j(1-x_i) は、x_i≠x_j のとき 1、等しいとき 0 になる。
    """
    model = Model("gpp")
    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)" % i)
    model.update()

    # Math: \sum_{i\in V} x_i = |V|/2
    model.addConstr(quicksum(x[i] for i in V) == len(V) / 2, "Partition")

    # 辺ごとのカット指標：
    # Math: x_i(1-x_j)+x_j(1-x_i) =
    #       \begin{cases}
    #       1 & (x_i\ne x_j)\\
    #       0 & (x_i=x_j)
    #       \end{cases}
    #
    # 目的（カット辺最小化）：
    # Math: \min \sum_{(i,j)\in E}\left(x_i(1-x_j)+x_j(1-x_i)\right)
    model.setObjective(
        quicksum(x[i] * (1 - x[j]) + x[j] * (1 - x[i]) for (i, j) in E),
        GRB.MINIMIZE,
    )

    model.update()
    model.__data = x
    return model


def gpp_qo_ps(V, E):
    """
    gpp_qo_ps -- quadratic optimization, positive semidefinite flavored model (MIQP)

    (x_i - x_j)^2 は 0/1 変数に対して、同一なら0、異なれば1になる。
    また平方は凸なので、連続緩和の見通しが良くなることがある（目的が凸方向に寄る）。
    """
    model = Model("gpp")
    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)" % i)
    model.update()

    # Math: \sum_{i\in V} x_i = |V|/2
    model.addConstr(quicksum(x[i] for i in V) == len(V) / 2, "Partition")

    # Math: (x_i-x_j)^2 =
    #       \begin{cases}
    #       1 & (x_i\ne x_j)\\
    #       0 & (x_i=x_j)
    #       \end{cases}
    #
    # 目的：
    # Math: \min \sum_{(i,j)\in E}(x_i-x_j)^2
    model.setObjective(
        quicksum((x[i] - x[j]) * (x[i] - x[j]) for (i, j) in E),
        GRB.MINIMIZE,
    )

    model.update()
    model.__data = x
    return model


def gpp_soco(V, E):
    """
    gpp_soco -- SOC（Second-Order Cone）っぽい形のモデル

    ポイント
    - 0/1 の二次式を補助変数 s,z で上から抑え、その和でカットを表す形。
    - ここでは (x_i+x_j-1)^2 と (x_j-x_i)^2 を使い、s+z=1 で整合を取っている。
    - 教材的には「二次式→補助変数→（SOC制約として扱える）」流れを示す例。

    注意
    - Gurobi の SOCP は連続変数に強いが、ここは二値も混ざるので結局MIP（MISOCP）になる。
    """
    model = Model("gpp model -- soco")
    x, s, z = {}, {}, {}

    # x[i] は0/1
    # Math: x_i \in \{0,1\}\quad(\forall i\in V)
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)" % i)

    # s[i,j], z[i,j] は連続（ここでは二次式の上界として使う）
    # Math: s_{ij}\ge 0,\ z_{ij}\ge 0\quad(\forall (i,j)\in E)
    for i, j in E:
        s[i, j] = model.addVar(vtype="C", name="s(%s,%s)" % (i, j))
        z[i, j] = model.addVar(vtype="C", name="z(%s,%s)" % (i, j))

    model.update()

    # バランス制約
    # Math: \sum_{i\in V} x_i = |V|/2
    model.addConstr(quicksum(x[i] for i in V) == len(V) / 2, "Partition")

    for i, j in E:
        # (x_i + x_j - 1)^2 <= s_{ij}
        # Math: (x_i+x_j-1)^2 \le s_{ij}
        model.addConstr(
            (x[i] + x[j] - 1) * (x[i] + x[j] - 1) <= s[i, j], "S(%s,%s)" % (i, j)
        )

        # (x_j - x_i)^2 <= z_{ij}
        # Math: (x_j-x_i)^2 \le z_{ij}
        model.addConstr((x[j] - x[i]) * (x[j] - x[i]) <= z[i, j], "Z(%s,%s)" % (i, j))

        # s_{ij} + z_{ij} == 1
        # Math: s_{ij}+z_{ij}=1
        model.addConstr(s[i, j] + z[i, j] == 1, "P(%s,%s)" % (i, j))

    # 目的：z の合計を最小化（ここでは cut を z 側に寄せて数える意図）
    # Math: \min \sum_{(i,j)\in E} z_{ij}
    model.setObjective(quicksum(z[i, j] for (i, j) in E), GRB.MINIMIZE)

    model.update()
    model.__data = x, s, z
    return model


import random


def make_data(n, prob):
    """
    make_data: ランダムグラフを作る

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

    V, E = make_data(4, 0.5)
    print("edges:", E)

    print("\n\n\nStandard model:")
    model = gpp(V, E)
    model.optimize()
    print("Opt.value=", model.ObjVal)
    x = model.__data
    print("partition:")
    print([i for i in V if x[i].X >= 0.5])
    print([i for i in V if x[i].X < 0.5])

    print("\n\n\nQuadratic optimization")
    model = gpp_qo(V, E)
    model.optimize()
    model.write("gpp_qo.lp")
    print("Opt.value=", model.ObjVal)
    x = model.__data
    print("partition:")
    print([i for i in V if x[i].X >= 0.5])
    print([i for i in V if x[i].X < 0.5])

    print("\n\n\nQuadratic optimization - positive semidefinite")
    model = gpp_qo_ps(V, E)
    model.optimize()
    model.write("gpp_qo.lp")
    print("Opt.value=", model.ObjVal)
    x = model.__data
    print("partition:")
    print([i for i in V if x[i].X >= 0.5])
    print([i for i in V if x[i].X < 0.5])

    print("\n\n\nSecond order cone optimization")
    model = gpp_soco(V, E)
    model.optimize()
    model.write("tmp.lp")
    status = model.Status
    if status == GRB.Status.OPTIMAL:
        print("Opt.value=", model.ObjVal)
        x, s, z = model.__data
        print("partition:")
        print([i for i in V if x[i].X >= 0.5])
        print([i for i in V if x[i].X < 0.5])
    else:
        model.computeIIS()
        for cst in model.getConstrs():
            if cst.IISConstr:
                print(cst.ConstrName)

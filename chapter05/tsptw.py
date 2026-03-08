"""
tsptw.py: solve the asymmetric traveling salesman problem with time window constraints
（時間窓付き非対称TSP: ATSP-TW / TSPTW）

問題設定（TSP with Time Windows）
- ノード（顧客）1..n をちょうど1回ずつ訪問する巡回路（有向）を作る。
- アーク (i,j) の移動時間/コスト c[i,j] があり、一般に非対称（c[i,j] != c[j,i]）。
- 各ノード i には時間窓 [e_i, l_i] があり、その範囲内の時刻に訪問開始（到着）しなければならない。
- 目的は総移動コスト（あるいは総移動時間）を最小化する。

このファイルにある3つの定式化
1) mtztw
   - MTZ（1-index potential）ベースの時間窓付き定式化（Big-Mで時間の整合を付ける）
2) mtz2tw
   - mtztw を強化した lifted 版（より強い制約でLP緩和が締まりやすい）
3) tsptw2
   - 2-index potential（アークごとに時間変数 u[i,j] を持つ）による別定式化

共通する「ツアーの骨格」制約（ATSP）
- 各ノード i から出るアークはちょうど1本（out-degree=1）
- 各ノード i に入るアークはちょうど1本（in-degree=1）
これだけではサブツアーが可能なので、MTZ系/2-index系でサブツアーや時間整合を崩せない構造を入れている。

時間窓の意味（重要）
- u_i（または到着時刻）を「訪問時刻」とすると、

  # Math: e_i \le u_i \le l_i

が必須である。
- さらに (i→j) を使うなら、

  # Math: u_j \ge u_i + c_{ij}

という時間の因果関係が必要で、アーク選択 x_{ij} と Big-M で線形化するのが定石である。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。
"""

import math
import random
from gurobipy import *


# ============================================================
# 1) MTZ + Time Window（1-index potential）
# ============================================================
def mtztw(n, c, e, l):
    """
    mtztw: TSPTW model based on MTZ one-index potential formulation

    変数の意味
    - x_{ij} ∈ {0,1}: アーク (i,j) を選ぶなら1
    - u_i: ノード i の訪問時刻（到着時刻、またはサービス開始時刻）

    時間窓
    - u_i は [e_i, l_i] 内に入らなければならない。

      # Math: e_i \le u_i \le l_i

    時間整合（アーク選択と連動）
    - (i→j) を選ぶなら u_j ≥ u_i + c_{ij} を強制する。
    - 選ばないならこの制約は無効化したいので Big-M を使う。

      # Math: u_j \ge u_i + c_{ij} - M(1-x_{ij})

    これを移項すると

      # Math: u_i - u_j + M x_{ij} \le M - c_{ij}

    となり、コードの MTZ(i,j) 制約の形になる。
    """
    model = Model("tsptw - mtz")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # u[i] : 訪問時刻（連続）
    # Math: e_i \le u_i \le l_i
    #
    # x[i,j] : アーク選択（二値）
    # Math: x_{ij}\in\{0,1\}\quad(\forall i\ne j)
    x, u = {}, {}
    for i in range(1, n + 1):
        u[i] = model.addVar(lb=e[i], ub=l[i], vtype="C", name="u(%s)" % i)
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 次数制約（ATSPの骨格）
    # ------------------------------------------------------------
    # Math: \sum_{j\ne i} x_{ij}=1\quad(\forall i)
    # Math: \sum_{j\ne i} x_{ji}=1\quad(\forall i)
    #
    # これで「各ノードをちょうど1回訪問」する形（ただしサブツアーはまだ可能）。
    for i in range(1, n + 1):
        model.addConstr(
            quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1, "Out(%s)" % i
        )
        model.addConstr(
            quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1, "In(%s)" % i
        )

    # ------------------------------------------------------------
    # 時間整合 + サブツアー除去（MTZ型Big-M）
    # ------------------------------------------------------------
    # (i→j) を使うなら u_j >= u_i + c_{ij} を強制する。
    #
    # Big-M の選び方（重要）
    # - きつい M を選ぶほど LP 緩和が強くなり探索が速くなりやすい。
    # - ここではペアごとに最小限の M を作る：
    #
    #   u_i <= l_i
    #   u_j >= e_j
    #
    # なので、u_i + c_{ij} - u_j の最大値は l_i + c_{ij} - e_j。
    # よって
    #
    #   M = max(l_i + c_{ij} - e_j, 0)
    #
    # が “その制約を無効化するのに十分な M” になる。
    #
    # 制約（移項後）：
    # Math: u_i - u_j + M x_{ij} \le M - c_{ij}
    #
    # 実装では j を 2..n にしている（基準ノード1を特別扱いする名残）。
    for i in range(1, n + 1):
        for j in range(2, n + 1):
            if i != j:
                M = max(l[i] + c[i, j] - e[j], 0)
                model.addConstr(
                    u[i] - u[j] + M * x[i, j] <= M - c[i, j],
                    "MTZ(%s,%s)" % (i, j),
                )

    # ------------------------------------------------------------
    # 目的関数（総移動コスト最小化）
    # ------------------------------------------------------------
    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, u
    return model


# ============================================================
# 2) 強化MTZ + Time Window（lifted constraints）
# ============================================================
def mtz2tw(n, c, e, l):
    """
    mtz2tw: stronger MTZ-based model for TSPTW（lifted / tightened）

    狙い
    - 標準の Big-M 時間整合は緩和が弱くなりやすい。
    - そこで逆向きアーク x_{ji} の情報なども使い、制約をリフティングして締める。
    - さらに u_i の下限/上限を x によって強化（lifted bounds）して、時間変数の自由度を減らす。

    注意
    - ここで使う M1, M2 は “無効化” のための Big-M をより細かく作っている。
    - 数式としては文献依存の lifted MTZ の一種で、理解のコアは
      「x の追加情報で Big-M を実質的に小さくする＝緩和を締める」点にある。
    """
    model = Model("tsptw - mtz-strong")

    x, u = {}, {}
    for i in range(1, n + 1):
        u[i] = model.addVar(lb=e[i], ub=l[i], vtype="C", name="u(%s)" % i)
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # 次数制約（Out/In）
    # Math: \sum_{j\ne i} x_{ij}=1,\ \sum_{j\ne i} x_{ji}=1
    for i in range(1, n + 1):
        model.addConstr(
            quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1, "Out(%s)" % i
        )
        model.addConstr(
            quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1, "In(%s)" % i
        )

        # Lifted time relation（代表形）
        # Math: u_i + c_{ij} - M_1(1-x_{ij}) + M_2 x_{ji} \le u_j
        #
        # - x_{ij}=1 のとき、基本の u_j >= u_i + c_{ij} を強く押す
        # - x_{ij}=0 のときは -M1 で無効化
        # - さらに x_{ji} の状況に応じて M2 で追加 tightening
        for j in range(2, n + 1):
            if i != j:
                M1 = max(l[i] + c[i, j] - e[j], 0)
                M2 = max(l[i] + min(-c[j, i], e[j] - e[i]) - e[j], 0)
                model.addConstr(
                    u[i] + c[i, j] - M1 * (1 - x[i, j]) + M2 * x[j, i] <= u[j],
                    "LiftedMTZ(%s,%s)" % (i, j),
                )

    # 追加の lifted bounds（uの下限/上限を x で締める）
    #
    # 下限側（例）
    # - i に入るアーク j→i が選ばれるとき、u_i は u_j + c_{ji} 以上になり得る。
    # - それを e から持ち上げる形で下限を強化する。
    #
    # Math: e_i + \sum_{j\ne i} \max(e_j+c_{ji}-e_i,0)\,x_{ji} \le u_i
    for i in range(2, n + 1):
        model.addConstr(
            e[i]
            + quicksum(
                max(e[j] + c[j, i] - e[i], 0) * x[j, i]
                for j in range(1, n + 1)
                if i != j
            )
            <= u[i],
            "LiftedLB(%s)" % i,
        )

        # 上限側（例）
        # - i→j が選ばれるとき、u_i は u_j - c_{ij} 以下の制約を間接的に持つ。
        # - それを l_i から引き下げる形で上限を強化する。
        #
        # Math: u_i \le l_i - \sum_{j\ne i} \max(l_i-l_j+c_{ij},0)\,x_{ij}
        model.addConstr(
            u[i]
            <= l[i]
            - quicksum(
                max(l[i] - l[j] + c[i, j], 0) * x[i, j]
                for j in range(2, n + 1)
                if i != j
            ),
            "LiftedUB(%s)" % i,
        )

    # 目的
    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, u
    return model


# ============================================================
# 3) 2-index potential（アークごとに時刻変数 u[i,j] を持つ）
# ============================================================
def tsptw2(n, c, e, l):
    """
    tsptw2: two-index potential model for TSPTW

    発想
    - ノード時刻 u_i を直接持つ代わりに、アーク (i,j) に対応した「出発時刻（または到着時刻寄与）」のような
      変数 u_{ij} を持たせる。
    - x_{ij} と u_{ij} を結びつけて、選ばれたアークにだけ時刻が乗るようにする。

    変数
    - x_{ij} ∈ {0,1}
    - u_{ij} ≥ 0（連続）

    時間窓を u_{ij} に埋め込む
    - もし (i,j) を使うなら u_{ij} は [e_i, l_i] の範囲の値を取り得る（iの時刻を表す）
    - 使わないなら u_{ij}=0 にできるようにする

      # Math: e_i x_{ij} \le u_{ij} \le l_i x_{ij}

    関係制約（Relate）
    - 各ノード j について、j に入ってくるアークの「(iの時刻 + 移動時間)」と、
      j から出ていくアークの「(jの時刻)」を整合させる形の差分制約。
    - これにより選択アークの連鎖に沿って時刻が伝播する。
    """
    model = Model("tsptw2")

    x, u = {}, {}

    # 変数作成
    # Math: x_{ij}\in\{0,1\}\quad(\forall i\ne j)
    # Math: u_{ij}\ge 0\quad(\forall i\ne j)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
                u[i, j] = model.addVar(vtype="C", name="u(%s,%s)" % (i, j))

    model.update()

    # 次数制約
    # Math: \sum_{j\ne i} x_{ij}=1,\ \sum_{j\ne i} x_{ji}=1
    for i in range(1, n + 1):
        model.addConstr(
            quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1, "Out(%s)" % i
        )
        model.addConstr(
            quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1, "In(%s)" % i
        )

    # 関係制約（時刻の伝播）
    # Math: \sum_{i\ne j}(u_{ij}+c_{ij}x_{ij}) - \sum_{k\ne j} u_{jk} \le 0\quad(\forall j=2..n)
    #
    # 直感：
    # - j に入る選択アーク (i→j) が1本だけ立つので、その u_{ij} が「iの時刻」を表す
    # - そこに移動時間 c_{ij} を足したものが「jに到着/開始する時刻」になり、
    #   j から出る選択アーク (j→k) が1本だけ立つので、その u_{jk} が「jの時刻」になってほしい
    # - それらの整合を “総和” で縛っている（教育用の簡潔表現）
    for j in range(2, n + 1):
        model.addConstr(
            quicksum(u[i, j] + c[i, j] * x[i, j] for i in range(1, n + 1) if i != j)
            - quicksum(u[j, k] for k in range(1, n + 1) if k != j)
            <= 0,
            "Relate(%s)" % j,
        )

    # 時間窓を u_{ij} に埋め込む（選ばれたときだけ範囲内）
    # Math: e_i x_{ij} \le u_{ij}
    # Math: u_{ij} \le l_i x_{ij}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                model.addConstr(e[i] * x[i, j] <= u[i, j], "LB(%s,%s)" % (i, j))
                model.addConstr(u[i, j] <= l[i] * x[i, j], "UB(%s,%s)" % (i, j))

    # 目的
    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, u
    return model


# ============================================================
# 付随：データ生成（距離と時間窓）
# ============================================================
def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_data(n, width):
    """
    make_data: compute matrix distance and time windows（簡易生成）

    生成の雰囲気
    - 平面上にランダム点を置き、距離をコスト c_{ij} にする（非対称ではなく対称寄りになるが例としてOK）。
    - 1→2→3→...→n の順に進むときの累積距離を "start" とし、
      各ノードの時間窓 [e_j, l_j] を start±delta で作る。
    - width を大きくすると delta が大きくなり、時間窓が緩くなる。

    注意
    - 本来のTSPTWではサービス時間や待ち時間の扱いもあることが多いが、ここでは簡略化している。
    """
    x = dict([(i, 100 * random.random()) for i in range(1, n + 1)])
    y = dict([(i, 100 * random.random()) for i in range(1, n + 1)])
    c = {}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if j != i:
                c[i, j] = distance(x[i], y[i], x[j], y[j])

    # 時間窓
    e = {1: 0}
    l = {1: 0}

    start = 0
    delta = int(76.0 * math.sqrt(n) / n * width) + 1
    for i in range(1, n):
        j = i + 1
        start += c[i, j]
        e[j] = max(start - delta, 0)
        l[j] = start + delta

    return c, x, y, e, l


if __name__ == "__main__":
    EPS = 1.0e-6

    # 例：小さな n=5 の手書きデータ
    n = 5
    c = {
        (1, 1): 0,
        (1, 2): 9,
        (1, 3): 10,
        (1, 4): 10,
        (1, 5): 10,
        (2, 1): 10,
        (2, 2): 0,
        (2, 3): 9,
        (2, 4): 10,
        (2, 5): 10,
        (3, 1): 10,
        (3, 2): 10,
        (3, 3): 0,
        (3, 4): 9,
        (3, 5): 10,
        (4, 1): 10,
        (4, 2): 10,
        (4, 3): 10,
        (4, 4): 0,
        (4, 5): 9,
        (5, 1): 9,
        (5, 2): 10,
        (5, 3): 10,
        (5, 4): 10,
        (5, 5): 0,
    }
    e = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    l = {1: 100, 2: 100, 3: 10, 4: 100, 5: 100}

    print(c)
    print(e)
    print(l)

    # ------------------------------------------------------------
    # mtztw
    # ------------------------------------------------------------
    model = mtztw(n, c, e, l)
    model.optimize()
    x, u = model.__data

    sol = [i for (v, i) in sorted([(u[i].X, i) for i in u])]
    print("mtz:")
    print(sol)
    print("Opt.value =", model.ObjVal)

    # ------------------------------------------------------------
    # mtz2tw（強化）
    # ------------------------------------------------------------
    model = mtz2tw(n, c, e, l)
    model.optimize()
    x, u = model.__data

    sol = [i for (v, i) in sorted([(u[i].X, i) for i in u])]
    print("mtz2:")
    print(sol)
    print("Opt.value =", model.ObjVal)

    # ------------------------------------------------------------
    # two-index model
    # ------------------------------------------------------------
    print("TWO INDEX MODEL")
    model = tsptw2(n, c, e, l)
    model.optimize()
    print("Opt.value=", model.ObjVal)

    x, u = model.__data

    # 選択アークの表示
    for i, j in x:
        if x[i, j].X > EPS:
            print(x[i, j].VarName, i, j, x[i, j].X)

    # u[i,j] を “到着時刻寄与” とみなして集計し、順序らしきものを作る（簡易表示）
    start_time = [0] * (n + 1)
    for i, j in u:
        if u[i, j].X > EPS:
            print(u[i, j].VarName, i, j, u[i, j].X)
            start_time[j] += u[i, j].X

    start = [i for v, i in sorted([(start_time[i], i) for i in range(1, n + 1)])]
    print(start)

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: x_{ij}\in\{0,1\}
# Math: \sum_{j\ne i} x_{ij}=1
# Math: \sum_{j\ne i} x_{ji}=1
# Math: e_i \le u_i \le l_i
# Math: u_i - u_j + M x_{ij} \le M - c_{ij}
# Math: u_i + c_{ij} - M_1(1-x_{ij}) + M_2 x_{ji} \le u_j
# Math: e_i x_{ij} \le u_{ij} \le l_i x_{ij}
# Math: \min \sum_{i\ne j} c_{ij}x_{ij}

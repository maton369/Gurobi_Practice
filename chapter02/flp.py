"""
flp.py: model for solving the capacitated facility location problem（容量制約付き施設配置問題: CFLP）

目的（何を解くか）
- n人（n地点）の顧客（customers）を、いくつかの施設（facilities）に割り当てる。
- 施設を開くと固定費 f[j] が発生し、施設ごとに容量 M[j] がある。
- 顧客 i の需要 d[i] を、どの施設 j からどれだけ供給するか（割当量 x[i,j]）で決める。
- 輸送コスト（単位コスト c[i,j] × 供給量 x[i,j]）と、施設の固定費の合計を最小化する。

数理最適化としての形式
- 施設を開く/開かない：二値変数 y[j]（0/1）
- 供給量：連続変数 x[i,j]（>=0 を暗黙に持つことが多い）
- 需要の充足（等式）、容量制約（不等式）、連結（強化）制約（不等式）
- よって **混合整数線形計画（MILP / MIP）** になる。

データの意味
- I: 顧客集合
- J: 施設集合
- d[i]: 顧客 i の需要
- M[j]: 施設 j の容量
- f[j]: 施設 j を開設したときの固定費
- c[i,j]: 顧客 i を施設 j から供給するときの単位コスト

# 数式のコメント方針
# - 数式は TeX の $$...$$ を使う
# - $$ の前後は空行を入れる
# - $$ の中身は詰める（空行なし）
# - さらにユーザー要望に合わせ、数式は Python コメント（#）としてコメントアウトして載せる

"""

from gurobipy import *


def flp(I, J, d, M, f, c):
    """
    flp -- capacitated facility location problem（容量制約付き施設配置問題）を解くモデルを構築する

    Parameters:
        - I: 顧客集合
        - J: 施設集合
        - d[i]: 顧客 i の需要
        - M[j]: 施設 j の容量
        - f[j]: 施設 j を開く固定費
        - c[i,j]: 顧客 i を施設 j が供給するときの単位コスト

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """

    # ------------------------------------------------------------
    # モデル作成
    # ------------------------------------------------------------
    model = Model("flp")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # y[j] ∈ {0,1}: 施設 j を開くなら1、開かないなら0
    #
    # # 数式（コメント）
    #
    # $$
    # y_j \in \{0,1\} \quad (\forall j \in J)
    # $$
    #
    # x[i,j] ≥ 0: 顧客 i の需要を施設 j からどれだけ供給するか（供給量）
    #
    # # 数式（コメント）
    #
    # $$
    # x_{ij} \ge 0 \quad (\forall i \in I,\ \forall j \in J)
    # $$
    #
    # 注意:
    # - x[i,j] を連続変数としているため、需要を分割して複数施設から供給することが可能なモデルになっている。
    #   もし「各顧客は1施設にだけ割当て（割当問題）」にしたいなら、x を 0/1 にするか、別の割当変数が必要になる。
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="C", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 需要制約（Demand）
    # ------------------------------------------------------------
    # 各顧客 i の需要 d[i] は、すべての施設からの供給量の合計でちょうど満たす。
    #
    # # 数式（コメント）
    #
    # $$
    # \sum_{j \in J} x_{ij} = d_i \quad (\forall i \in I)
    # $$
    #
    # この制約により、需要が不足しない（= 必ず満たす）だけでなく、過剰供給も許さない（等式）。
    for i in I:
        model.addConstr(quicksum(x[i, j] for j in J) == d[i], "Demand(%s)" % i)

    # ------------------------------------------------------------
    # 容量制約（Capacity）
    # ------------------------------------------------------------
    # 施設 j の総供給量（顧客へ送った合計）が、施設容量 M[j] を超えない。
    # ただし施設が開いていない（y[j]=0）なら供給量は 0 でなければならない。
    #
    # その連結を Big-M 形で書くと次：
    #
    # # 数式（コメント）
    #
    # $$
    # \sum_{i \in I} x_{ij} \le M_j\,y_j \quad (\forall j \in J)
    # $$
    #
    # y_j=0 なら右辺0なので x_{ij} は全て0を強制される。
    # y_j=1 なら通常の容量制約になる。
    #
    # 注意（このコードのバグ/改善点）
    # - ループが `for j in M:` になっているが、M は辞書なので「キー集合」としては機能するものの、
    #   意図としては `for j in J:` が自然（Jと一致させた方が読みやすい）。
    # - 制約名が `"Capacity(%s)"%i` になっており、ここで i は未定義（直前の i ループの残骸）なのでバグ。
    #   本来は `"Capacity(%s)"%j` が正しい。
    #
    # 現状コードのままだと、制約の name 部分が誤っており、デバッグ時に非常に混乱する。
    for j in M:
        model.addConstr(quicksum(x[i, j] for i in I) <= M[j] * y[j], "Capacity(%s)" % i)

    # ------------------------------------------------------------
    # 強化制約（Strong linking constraints）
    # ------------------------------------------------------------
    # x_{ij} が正になるのは施設 j が開いているときだけ、という性質をより強く入れる。
    # 需要制約が等式であるため、x_{ij} <= d_i y_j を入れると「開いていない施設には顧客 i を割当てられない」が明確になる。
    #
    # # 数式（コメント）
    #
    # $$
    # x_{ij} \le d_i\,y_j \quad (\forall i \in I,\ \forall j \in J)
    # $$
    #
    # これは容量制約（Σ_i x_{ij} ≤ M_j y_j）とは別の方向の tightening（強化）になることが多い。
    # - 容量制約は施設単位の合計を縛る。
    # - 強化制約は顧客単位で「各アークの上限」を縛る。
    #
    # 結果として探索空間が縮まり、MIPの探索が速くなることがある。
    for i, j in x:
        model.addConstr(x[i, j] <= d[i] * y[j], "Strong(%s,%s)" % (i, j))

    # ------------------------------------------------------------
    # 目的関数（固定費 + 輸送費）
    # ------------------------------------------------------------
    # 施設固定費：開設した施設の固定費の合計
    #
    # # 数式（コメント）
    #
    # $$
    # \sum_{j \in J} f_j\,y_j
    # $$
    #
    # 輸送費：単位コスト×供給量の合計
    #
    # # 数式（コメント）
    #
    # $$
    # \sum_{i \in I}\sum_{j \in J} c_{ij}\,x_{ij}
    # $$
    #
    # 総コスト最小化：
    #
    # # 数式（コメント）
    #
    # $$
    # \min \left(\sum_{j \in J} f_j\,y_j + \sum_{i \in I}\sum_{j \in J} c_{ij}\,x_{ij}\right)
    # $$
    #
    model.setObjective(
        quicksum(f[j] * y[j] for j in J)
        + quicksum(c[i, j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE,
    )

    model.update()

    # ------------------------------------------------------------
    # 後段で解を取り出しやすいように model.__data に変数辞書を格納
    # ------------------------------------------------------------
    # こうしておくと、main側で x,y を参照して施設やエッジを抽出できる。
    model.__data = x, y
    return model


def make_data():
    # 顧客集合 I と需要 d
    I, d = multidict({1: 80, 2: 270, 3: 250, 4: 160, 5: 180})

    # 施設集合 J、容量 M、固定費 f
    # multidict の値を [capacity, fixed_cost] の形で持たせ、J,M,f に分解している。
    J, M, f = multidict({1: [500, 1000], 2: [500, 1000], 3: [500, 1000]})

    # 単位輸送コスト c[i,j]
    c = {
        (1, 1): 4,
        (1, 2): 6,
        (1, 3): 9,
        (2, 1): 5,
        (2, 2): 4,
        (2, 3): 7,
        (3, 1): 6,
        (3, 2): 3,
        (3, 3): 4,
        (4, 1): 8,
        (4, 2): 5,
        (4, 3): 3,
        (5, 1): 10,
        (5, 2): 8,
        (5, 3): 4,
    }
    return I, J, d, M, f, c


if __name__ == "__main__":
    # ------------------------------------------------------------
    # データ作成→モデル構築→最適化
    # ------------------------------------------------------------
    I, J, d, M, f, c = make_data()
    model = flp(I, J, d, M, f, c)
    model.optimize()

    # ------------------------------------------------------------
    # 解の抽出
    # ------------------------------------------------------------
    EPS = 1.0e-6
    x, y = model.__data

    # 輸送が発生しているエッジ（顧客 i から施設 j への供給が正）
    edges = [(i, j) for (i, j) in x if x[i, j].X > EPS]

    # 開設された施設
    facilities = [j for j in y if y[j].X > EPS]

    # print は Python3 形式に統一する方針。
    print("Optimal value=", model.ObjVal)
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    # ------------------------------------------------------------
    # 参考: networkx + matplotlib で結果を可視化
    # ------------------------------------------------------------
    try:
        import networkx as NX
        import matplotlib.pyplot as P

        P.clf()
        G = NX.Graph()

        other = [j for j in y if j not in facilities]
        customers = ["c%s" % i for i in d]

        G.add_nodes_from(facilities)
        G.add_nodes_from(other)
        G.add_nodes_from(customers)

        for i, j in edges:
            G.add_edge("c%s" % i, j)

        position = NX.drawing.layout.spring_layout(G)

        # facility: yellow, closed facility: green, customers: blue
        NX.draw(G, position, node_color="y", nodelist=facilities)
        NX.draw(G, position, node_color="g", nodelist=other)
        NX.draw(G, position, node_color="b", nodelist=customers)

        P.show()

    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")

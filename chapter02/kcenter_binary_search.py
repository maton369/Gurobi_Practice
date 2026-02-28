"""
kcenter_binary_search.py: use bisection for solving the k-center problem
（k-center を二分探索で解く）

狙い（何をしているか）
- k-center は「最悪距離（最大距離）」を最小化する min-max 問題である。
- 典型的なアプローチとして、
  「距離しきい値 theta を固定したとき、全顧客が theta 以内の施設でカバーできるか？」
  という判定問題を繰り返し解き、theta を二分探索で詰める手法がある。
- このスクリプトはその方針を採用している。

この実装の特徴
- しきい値 theta を固定すると、「距離が theta を超える割当 x[i,j] を禁止」できる。
- その状態で「未カバー顧客数（uncovered）を最小化」する MIP を解く。
- “未カバーが 0 になれる（全顧客カバー可能）” なら theta は達成可能として UB を下げる。
- そうでなければ theta は厳しすぎるとして LB を上げる。
- UB-LB <= delta になるまで繰り返す。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。

"""

from gurobipy import *


def kcover(I, J, c, k):
    """
    kcover -- k個の施設でカバーできない顧客数（未カバー数）を最小化するモデルを作る。

    Parameters:
        - I: 顧客集合
        - J: 候補施設集合
        - c[i,j]: 顧客 i を施設 j が担当するときの距離/コスト
        - k: 選ぶ施設数（ちょうど k 個）

    Returns:
        - model: 未カバー数最小化モデル
    """

    model = Model("k-center")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # y[j] ∈ {0,1} : 施設 j を選ぶなら1
    # Math: y_j \in \{0,1\}\quad(\forall j\in J)
    #
    # x[i,j] ∈ {0,1} : 顧客 i を施設 j に割り当てるなら1
    # Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
    #
    # z[i] ∈ {0,1} : 顧客 i が「未カバー（どの施設にも割当てられない）」なら1
    # Math: z_i \in \{0,1\}\quad(\forall i\in I)
    #
    # この z[i] を導入することで、「割当できない顧客が出てもモデルは可解」のままにし、
    # その代わり z[i] を目的関数で最小化する（= 未カバーを減らす）形にしている。
    z, y, x = {}, {}, {}
    for i in I:
        z[i] = model.addVar(vtype="B", name="z(%s)" % i)
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 制約
    # ------------------------------------------------------------
    for i in I:
        # (1) 「割当」か「未カバー」か、どちらか一方にする制約
        #
        # 顧客 i は「どれか1つの施設へ割当てる」か、あるいは「未カバー(z[i]=1)」のどちらか。
        # Math: \sum_{j\in J} x_{ij} + z_i = 1\quad(\forall i\in I)
        #
        # - もし割当が可能なら、ある j で x[i,j]=1 として z[i]=0 にできる。
        # - どの割当も禁止されている（後で theta により UB=0 にされる）なら、
        #   仕方なく z[i]=1 にして等式を満たす。
        model.addConstr(quicksum(x[i, j] for j in J) + z[i] == 1, "Assign(%s)" % i)

        # (2) 連結（強化）制約：選ばれていない施設には割当できない
        # Math: x_{ij} \le y_j\quad(\forall i\in I,\ \forall j\in J)
        for j in J:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    # (3) 施設数制約：ちょうど k 個を選ぶ
    # Math: \sum_{j\in J} y_j = k
    model.addConstr(quicksum(y[j] for j in J) == k, "k_center")

    # ------------------------------------------------------------
    # 目的関数：未カバー顧客数を最小化
    # Math: \min \sum_{i\in I} z_i
    #
    # 注意：
    # - k-center の「最悪距離最小化」は solve_kcenter() 側で theta を二分探索して達成する。
    # - ここでは theta を固定したときに「全員カバーできるか（zの最小値が0か）」を判定したい。
    model.setObjective(quicksum(z[i] for i in I), GRB.MINIMIZE)

    model.update()
    model.__data = x, y, z
    return model


def solve_kcenter(I, J, c, k, delta):
    """
    solve_kcenter -- 二分探索で k-center の最悪距離を最小化する。

    Parameters:
        - I: 顧客集合
        - J: 候補施設集合
        - c[i,j]: 距離/コスト
        - k: 選ぶ施設数
        - delta: 二分探索の停止許容誤差（UB-LB <= delta で終了）

    Returns:
        - facilities: 選ばれた施設のリスト
        - edges: 顧客→施設の割当（x[i,j]=1）のリスト
    """

    # ------------------------------------------------------------
    # 判定モデル（未カバー最小化モデル）を1回だけ作って使い回す
    # ------------------------------------------------------------
    model = kcover(I, J, c, k)
    x, y, z = model.__data

    facilities, edges = [], []

    # ------------------------------------------------------------
    # 二分探索の初期区間
    # ------------------------------------------------------------
    # LB: 0（距離の下限）
    # UB: 全ての顧客-施設候補の最大距離（これ以上なら必ずカバー可能）
    LB = 0.0
    UB = max(c[i, j] for (i, j) in c)

    # ------------------------------------------------------------
    # 二分探索ループ
    # ------------------------------------------------------------
    while UB - LB > delta:
        theta = (UB + LB) / 2.0

        # --------------------------------------------------------
        # しきい値 theta を固定したときに許される割当だけ残す
        # --------------------------------------------------------
        # c[i,j] > theta なら「その施設は顧客 i をカバーできない」ので x[i,j]=0 を強制したい。
        # ここでは「変数の上限 UB を 0 にする」という形で禁止している。
        #
        # Math: c_{ij} > \theta \Rightarrow x_{ij}=0
        #
        # 逆に c[i,j] <= theta なら割当可能なので UB=1 のままにする。
        #
        # 重要：
        # - 制約を追加・削除するのではなく、変数境界を更新しているのでモデル再構築が不要。
        # - ただし毎回 update() が必要になる。
        for j in J:
            for i in I:
                if c[i, j] > theta:
                    x[i, j].UB = 0.0
                else:
                    x[i, j].UB = 1.0

        model.update()

        # --------------------------------------------------------
        # MIPソルバの探索を少し軽くするためのパラメータ
        # --------------------------------------------------------
        # Cutoff は「目的値が cutoff を超える解しか見込めないなら打ち切る」ような閾値。
        # ここでは cutoff=0.1 としているので、未カバー数が 0 に近い解（理想は0）を優先して探索する意図がある。
        #
        # 注意：
        # - 本当に「全員カバー（目的=0）」が知りたいだけなら、Cutoff の設計は慎重に行うべき。
        # - 目的が整数（zの和）なので、0.1 は事実上「0以外は打ち切り」寄りの挙動になりやすい。
        model.Params.Cutoff = 0.1

        # model.Params.OutputFlag = 0  # ログを消したい場合

        # --------------------------------------------------------
        # 最適化（= この theta で未カバー最小がいくつか調べる）
        # --------------------------------------------------------
        model.optimize()

        # --------------------------------------------------------
        # 判定：theta は達成可能か？
        # --------------------------------------------------------
        # 理想的には「最適目的値が 0（未カバー0）」なら全員カバーできる → 達成可能 → UB を下げる。
        #
        # ただしこのコードは `model.status == OPTIMAL` だけで UB を更新している。
        # これは Cutoff を使って「未カバーが0にできるなら最適として返りやすい」という前提に依存している。
        #
        # より厳密に書くなら：
        # - OPTIMAL かつ sum z[i].X == 0 を確認して UB を更新
        # - そうでなければ LB を更新
        #
        # ただし元コードの意図を尊重し、ここでは「OPTIMAL なら達成可能」とみなしている。
        if model.status == GRB.Status.OPTIMAL:
            UB = theta
            facilities = [j for j in y if y[j].X > 0.5]
            edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
        else:
            LB = theta

    return facilities, edges


# ------------------------------------------------------------
# データ生成（平面上のランダム点→距離コスト）
# ------------------------------------------------------------
import math
import random


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_data(n, m, same=True):
    if same == True:
        I = range(n)
        J = range(m)
        x = [random.random() for i in range(max(m, n))]  # positions
        y = [random.random() for i in range(max(m, n))]
    else:
        I = range(n)
        J = range(n, n + m)
        x = [random.random() for i in range(n + m)]  # positions
        y = [random.random() for i in range(n + m)]
    c = {}
    for i in I:
        for j in J:
            c[i, j] = distance(x[i], y[i], x[j], y[j])

    return I, J, c, x, y


if __name__ == "__main__":
    random.seed(67)
    n = 200
    m = n
    I, J, c, x_pos, y_pos = make_data(n, m, same=True)

    k = 20
    delta = 1.0e-4

    facilities, edges = solve_kcenter(I, J, c, k, delta)

    # print は Python3 形式に統一
    print("Selected facilities:", facilities)
    print("Edges:", edges)
    print(
        "Max distance from a facility to a customer: ",
        max([c[i, j] for (i, j) in edges]),
    )

    # ------------------------------------------------------------
    # 可視化（networkx + matplotlib）
    # ------------------------------------------------------------
    try:
        import networkx as NX
        import matplotlib.pyplot as P

        P.clf()
        G = NX.Graph()

        facilities = set(facilities)
        unused = set(j for j in J if j not in facilities)
        client = set(i for i in I if i not in facilities and i not in unused)

        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        G.add_nodes_from(unused)

        for i, j in edges:
            G.add_edge(i, j)

        position = {}
        for i in range(len(x_pos)):
            position[i] = (x_pos[i], y_pos[i])

        NX.draw(G, position, with_labels=False, node_color="w", nodelist=facilities)
        NX.draw(
            G,
            position,
            with_labels=False,
            node_color="c",
            nodelist=unused,
            node_size=50,
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
# モデルまとめ（Mathover用の数式コメント）
# ------------------------------------------------------------
# Math: y_j \in \{0,1\}\quad(\forall j\in J)
# Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
# Math: z_i \in \{0,1\}\quad(\forall i\in I)
# Math: \sum_{j\in J} x_{ij} + z_i = 1\quad(\forall i\in I)
# Math: x_{ij}\le y_j\quad(\forall i\in I,\ \forall j\in J)
# Math: \sum_{j\in J} y_j = k
# Math: \min \sum_{i\in I} z_i
# Math: c_{ij}>\theta\Rightarrow x_{ij}=0

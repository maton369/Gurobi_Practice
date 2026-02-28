"""
kmedian.py: model for solving the k-median problem（k-メディアン問題）

目的（何を解くか）
- 顧客集合 I を、候補施設集合 J のうち「ちょうど k 個」選んだ施設からサービスする。
- 顧客 i を施設 j に割り当てたときのコスト c[i,j] が与えられており、
  総割当コストを最小化するように施設選択と割当を同時に決める。

k-median の直感
- 「施設を k 個だけ置くなら、どこに置くと総移動距離（総コスト）が最小になるか？」というクラスタリング問題に近い。
- k-means が “二乗距離” を使うのに対し、k-median は “距離（線形）” を使う文脈で語られることが多い。
- この実装は「候補点が離散集合 J に限られる」離散最適化版である（p-median の p=k に相当）。

数理最適化としての形式
- 施設を開くか：二値変数 y[j]
- 顧客を施設に割り当てるか：二値変数 x[i,j]
- 制約・目的は線形
- よって **0-1混合整数線形計画（0-1 MILP / MIP）** として解く。

数式コメント方針（Mathover対応）
- 数式は必ず `# Math: <LaTeX>` の1行形式にする（ホバーでレンダリングされる）
- `$$...$$` は使わない（Mathoverの既定トリガが `Math:` のため）

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *


def kmedian(I, J, c, k):
    """
    kmedian -- 顧客を k 個の施設からサービスする総コスト最小化モデルを構築する

    Parameters:
        - I: 顧客集合
        - J: 候補施設集合（顧客と同じ点集合にすることも、別集合にすることもある）
        - c[i,j]: 顧客 i を施設 j が担当するときのコスト（距離など）
        - k: 開設する施設数（ちょうど k 個）

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """

    # ------------------------------------------------------------
    # モデル作成
    # ------------------------------------------------------------
    model = Model("k-median")

    # ------------------------------------------------------------
    # 変数定義
    # ------------------------------------------------------------
    # y[j] ∈ {0,1}: 施設 j を選ぶ（開設する）なら1
    # Math: y_j \in \{0,1\}\quad(\forall j\in J)
    #
    # x[i,j] ∈ {0,1}: 顧客 i を施設 j に割り当てるなら1
    # Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
    #
    # このモデルは「割当は必ず1施設（後述のAssign制約）」なので、
    # x_{ij} は顧客 i がどの施設に属するか（クラスタラベル）を表す。
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 割当制約（各顧客は必ず1施設に割り当てられる）
    # ------------------------------------------------------------
    # 各顧客 i について、割当変数の和が 1 になるようにする。
    # これにより、顧客は「ちょうど1つの施設」を選ぶ（分割割当は不可）。
    # Math: \sum_{j\in J} x_{ij} = 1\quad(\forall i\in I)
    for i in I:
        model.addConstr(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)

        # --------------------------------------------------------
        # 強化（連結）制約：開いていない施設には割り当てられない
        # --------------------------------------------------------
        # x_{ij} <= y_j を入れることで、y_j=0 の施設には x_{ij}=0 を強制する。
        # Math: x_{ij} \le y_j\quad(\forall i\in I,\ \forall j\in J)
        #
        # これは典型的な linking constraint で、MIP探索の tightening にもなる。
        for j in J:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    # ------------------------------------------------------------
    # 施設数制約（ちょうど k 個を選ぶ）
    # ------------------------------------------------------------
    # 選ばれた施設数の合計が k になるようにする。
    # Math: \sum_{j\in J} y_j = k
    model.addConstr(quicksum(y[j] for j in J) == k, "Facilities")

    # ------------------------------------------------------------
    # 目的関数（総割当コスト最小化）
    # ------------------------------------------------------------
    # 顧客 i を施設 j に割当てるとコスト c_{ij} が発生。
    # x_{ij}=1 の組だけが効くので、総コストは Σ c_{ij} x_{ij}。
    # Math: \min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

    model.update()

    # 後段の出力処理で使いやすいように変数辞書を保存
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
    same=True なら「顧客と施設候補が同じ点集合から選ばれる」設定（I=0..n-1, J=0..m-1）
    same=False なら「顧客と施設候補が別集合」設定（I=0..n-1, J=n..n+m-1）
    """
    if same == True:
        I = range(n)
        J = range(m)
        # 同じ配列 x,y を顧客と施設候補の両方で共有する（同じ点集合上にいる）
        x = [random.random() for i in range(max(m, n))]
        y = [random.random() for i in range(max(m, n))]
    else:
        I = range(n)
        J = range(n, n + m)
        # 顧客側と施設候補側を別インデックス領域に置く
        x = [random.random() for i in range(n + m)]
        y = [random.random() for i in range(n + m)]

    # コスト行列 c[i,j] を距離で構成
    c = {}
    for i in I:
        for j in J:
            c[i, j] = distance(x[i], y[i], x[j], y[j])

    return I, J, c, x, y


if __name__ == "__main__":
    import sys

    # 乱数固定（再現性のため）
    random.seed(67)

    # 顧客数 n と施設候補数 m
    n = 200
    m = n

    # same=True なので「顧客点と施設候補点が同じ点集合」
    I, J, c, x_pos, y_pos = make_data(n, m, same=True)

    # 選ぶ施設数 k
    k = 20

    # モデル構築
    model = kmedian(I, J, c, k)

    # スレッド数を固定したい場合（実験の再現性や比較のため）
    # model.Params.Threads = 1

    # 最適化（MIP）
    model.optimize()

    EPS = 1.0e-6
    x, y = model.__data

    # 割当が立っている（x[i,j]=1）ペアを抽出
    edges = [(i, j) for (i, j) in x if x[i, j].X > EPS]

    # 選ばれた施設（y[j]=1）を抽出
    facilities = [j for j in y if y[j].X > EPS]

    # print は Python3 形式に統一
    print("Optimal value=", model.ObjVal)
    print("Selected facilities:", facilities)
    print("Edges:", edges)

    # 最大割当距離（最も遠い顧客→施設の距離）
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

        # same=True の場合、I と J が同じインデックス領域に重なる。
        # client は「顧客のうち施設でも候補でもない点」を意図しているが、
        # same=True だと I=J なので、ここは実質 “施設以外の顧客” になりやすい。
        client = set(i for i in I if i not in facilities and i not in other)

        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        G.add_nodes_from(other)

        for i, j in edges:
            G.add_edge(i, j)

        # 位置情報（座標）
        position = {}
        for i in range(len(x_pos)):
            position[i] = (x_pos[i], y_pos[i])

        # 描画色
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
# このモデルのまとめ（数式） ※ Mathover対応
# ------------------------------------------------------------
# 変数：
# Math: y_j \in \{0,1\}\quad(\forall j\in J)
# Math: x_{ij} \in \{0,1\}\quad(\forall i\in I,\ \forall j\in J)
#
# 制約：
# Math: \sum_{j\in J} x_{ij} = 1\quad(\forall i\in I)
# Math: x_{ij} \le y_j\quad(\forall i\in I,\ \forall j\in J)
# Math: \sum_{j\in J} y_j = k
#
# 目的：
# Math: \min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}
#
# 解釈：
# - ちょうど k 個の施設を選び、全顧客をどれか1つの選ばれた施設へ割り当て、
#   総コストを最小化する。

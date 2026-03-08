"""
tsp.py: solve the Traveling Salesman Problem（巡回セールスマン問題: TSP）

問題設定
- n 個の都市（顧客）をちょうど1回ずつ訪問し、出発点に戻る巡回路（ツアー）を作る。
- 総移動コストを最小化する。

この実装のアプローチ（重要）
- 最初は「次数制約だけ」を持つ assignment / 2-regular graph モデルを解く。
  これは各頂点の次数が2になるので「各頂点から2本の辺が出る」ことは保証するが、
  その結果は「複数のサイクル（subtour）」の集合になり得る（= 1本のツアーにならない）。
- そこで、解に含まれるサイクルが分裂していたら、それを禁止する「カット（subtour elimination cut）」を追加する。
- カット追加を繰り返し、サブツアーがなくなったら（連結な1つのサイクルになったら）解がTSPツアーになる。
- さらにこのコードは、最初は変数を連続（0〜1）として解き、連結になった段階で二値化して整数解に切り替える。
  つまり「カット追加による tightening → 最後に整数化」という、教育用に分かりやすい流れになっている。

数理最適化としての形式
- 辺変数 x_{ij} を持つ（無向、i<j のみ定義）
- 次数制約：各頂点の incident edge の合計が2
- 目的：総コスト最小化
- サブツアー除去制約（SEC）を逐次追加
- 最終的には x_{ij} を二値にして MIP（MILP）として解く

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。
"""

import math
import time
import random
import networkx
from gurobipy import *


def solve_tsp(V, c):
    """
    solve_tsp -- solve the traveling salesman problem
    - start with assignment model (degree-2 constraints)
    - add cuts until there are no sub-cycles (subtour elimination)

    Parameters:
        - V: nodes
        - c[i,j]: cost for traversing undirected edge (i,j) with j>i

    Returns:
        - objective value
        - list of edges used in the final tour
    """

    # ------------------------------------------------------------
    # カット追加関数（subtour elimination）
    # ------------------------------------------------------------
    def addcut(cut_edges):
        """
        addcut: 現在の解（選ばれた辺集合）が複数連結成分に分かれていたら、
        各成分 S に対して「S の内部に張れる辺の本数は |S|-1 以下」という制約を追加する。

        直感
        - もし S の中だけで |S| 本以上の辺が選ばれているなら、S の内部に閉じたサイクルができる。
        - TSPツアーは全頂点を1つのサイクルで結ぶ必要があるので、
          「部分集合 S の中だけで完結するサイクル（subtour）」を禁止したい。
        - そのための代表的な制約が subtour elimination constraint (SEC)。

        数式（SECの一形態）
        - 無向の辺変数 x_{ij}（i<j）に対し、部分集合 S ⊂ V について

        # Math: \sum_{i\in S}\sum_{j\in S,\ j>i} x_{ij} \le |S|-1

        を課すと、S の内部にサイクルを作れなくなる（少なくとも1本は外に出る必要が出る）。

        実装の手順
        1) 現在選ばれた辺 cut_edges からグラフを作る
        2) connected_components で連結成分（= サブツアー候補）を列挙
        3) 成分が1つなら（全体が連結なら）カット不要 → False
        4) 成分が複数なら各成分 S に SEC を追加 → True
        """
        G = networkx.Graph()
        G.add_edges_from(cut_edges)

        # networkx.connected_components は「連結成分の集合のイテレータ」を返す。
        # 連結成分が複数ある = サブツアーが存在する可能性が高い。
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False

        for S in Components:
            # S 内部に選ばれる辺の数を |S|-1 以下に制限
            model.addConstr(
                quicksum(x[i, j] for i in S for j in S if j > i) <= len(S) - 1
            )
            print("cut: len(%s) <= %s" % (S, len(S) - 1))
        return True

    def addcut2(cut_edges):
        """
        addcut2: 別形式のカット（カット集合形式）
        - 成分 S とその補集合 T=V\\S の間を少なくとも2本結ぶ、という制約を追加する。

        数式（一般的な cut constraint の一例）
        - 無向TSPの典型的なカットは

        # Math: \sum_{i\in S}\sum_{j\in V\setminus S} x_{ij} \ge 2

        である（S と外部は少なくとも2本で結ばれないと1つの巡回路にならない）。

        注意
        - この実装は i<j の変数しか持たないので、添字の扱いに注意が必要。
        - この関数はデバッグ用として残っているが、実際のsolveループでは addcut を使っている。
        """
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False

        for S in Components:
            T = set(V) - set(S)
            print("S:", S)
            print("T:", T)

            # 注意：ここは「i in S, j in T, j>i」だけを足しているので、
            # S-T を跨ぐ全ての辺を数え切れていない可能性がある（変数が i<j のみのため）。
            # 厳密にやるなら i<j の条件を揃えつつ (min(i,j), max(i,j)) で参照するなどが安全。
            model.addConstr(quicksum(x[i, j] for i in S for j in T if j > i) >= 2)
            print(
                "cut: %s <--> %s >= 2" % (S, T), [(i, j) for i in S for j in T if j > i]
            )
        return True

    # ------------------------------------------------------------
    # ここからメイン：モデル構築 → カット追加反復 → 最終解
    # ------------------------------------------------------------
    model = Model("tsp")

    # model.Params.OutputFlag = 0  # ログを消したい場合

    # ------------------------------------------------------------
    # 変数：無向辺の選択 x[i,j]
    # ------------------------------------------------------------
    # 無向なので i<j のみ定義する（重複を避ける）。
    #
    # 最初は連続 0..1（LP）として解き、連結化が進んだら二値（MIP）に切り替える。
    # addVar(ub=1) だけだと下限lb=0が暗黙なので 0<=x<=1 の連続変数になる。
    #
    # Math: 0 \le x_{ij} \le 1\quad(\forall i<j)
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i, j] = model.addVar(ub=1, name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 次数制約：各頂点の次数は2
    # ------------------------------------------------------------
    # 各頂点 i について、i に接続する選択辺の本数を2にする。
    #
    # 無向で i<j だけ持っているので、i に接続する辺は
    # - (j,i) 形式（j<i）の変数 x[j,i]
    # - (i,j) 形式（j>i）の変数 x[i,j]
    # の両方を足し合わせる必要がある。
    #
    # Math: \sum_{j<i} x_{ji} + \sum_{j>i} x_{ij} = 2\quad(\forall i\in V)
    for i in V:
        model.addConstr(
            quicksum(x[j, i] for j in V if j < i)
            + quicksum(x[i, j] for j in V if j > i)
            == 2,
            "Degree(%s)" % i,
        )

    # ------------------------------------------------------------
    # 目的関数：総コスト最小化
    # ------------------------------------------------------------
    # Math: \min \sum_{i<j} c_{ij}x_{ij}
    model.setObjective(
        quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i),
        GRB.MINIMIZE,
    )

    EPS = 1.0e-6

    # ------------------------------------------------------------
    # カット追加ループ
    # ------------------------------------------------------------
    # 1) 現在のモデルを解く
    # 2) x>0 の辺を取り出す
    # 3) 連結でなければ SEC を追加して再度解く
    # 4) 連結になったら、まだ連続（LP）なら二値化して最終的に整数解へ
    while True:
        model.optimize()

        # 現在の解で選ばれている辺（x_{ij} > EPS）を抽出
        edges = []
        for i, j in x:
            if x[i, j].X > EPS:
                edges.append((i, j))

        # サブツアーがあれば cut を追加し、もう一度 optimize へ
        if addcut(edges) == False:
            # 連結成分が1つ（= 全体が連結）になった
            if model.IsMIP:
                # すでに整数（xが二値）なら、これでTSPツアーが得られたとみなして終了
                break

            # まだ連続（LP）なら、この時点で x を二値化して MIP に切り替える
            # （ここから先は「連結 + 次数2」のまま、整数化により本物のツアーへ収束させる）
            #
            # Math: x_{ij}\in\{0,1\}\quad(\forall i<j)
            for i, j in x:
                x[i, j].VType = "B"
            model.update()

    return model.ObjVal, edges


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_data(n):
    """make_data: compute matrix distance based on euclidean distance"""
    V = range(1, n + 1)
    x = dict([(i, random.random()) for i in V])
    y = dict([(i, random.random()) for i in V])
    c = {}
    for i in V:
        for j in V:
            if j > i:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
    return V, c


if __name__ == "__main__":
    import sys

    # Parse argument
    if len(sys.argv) < 2:
        print("Usage: %s instance" % sys.argv[0])
        exit(1)

    # TSPLIB読み込み（外部モジュール）
    from read_tsplib import read_tsplib

    try:
        V, c, x_coord, y_coord = read_tsplib(sys.argv[1])
    except:
        print("Cannot read TSPLIB file", sys.argv[1])
        exit(1)

    obj, edges = solve_tsp(V, c)

    print()
    print("Optimal tour:", edges)
    print("Optimal cost:", obj)
    print()

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: 0 \le x_{ij} \le 1
# Math: \sum_{j<i} x_{ji} + \sum_{j>i} x_{ij} = 2
# Math: \min \sum_{i<j} c_{ij}x_{ij}
# Math: \sum_{i\in S}\sum_{j\in S,\ j>i} x_{ij} \le |S|-1
# Math: x_{ij}\in\{0,1\}

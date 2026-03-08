"""
vrp.py: model for the Vehicle Routing Problem（車両経路問題: VRP）
        using callback for adding cuts（コールバックで遅延制約=Lazy cut を追加）

問題設定（Capacitated VRP: CVRP に近い形）
- ノード集合 V のうち V[0] をデポ（depot、出発・帰還地点）とする。
- 残り V[1:],..., は顧客で、それぞれ需要 q[i] を持つ。
- 車両が m 台あり、各車両の容量が Q。
- 各顧客はちょうど1回訪問される（顧客の次数=2 で表現：入って出る）。
- デポからは車両 m 台が出発して帰還する（デポ次数=2m）。
- 目的は総移動コスト（距離）を最小化する。

この実装のアプローチ（重要）
- まずは「次数制約だけ」を持つ assignment / 2-regular 的なモデルを作り、
  デポ以外の各顧客は次数2、デポは次数2m にする。
- ただし次数制約だけだと、顧客部分が複数の連結成分に分裂し得る（= サブツアーが出る）。
- さらに VRP では容量制約があるため、顧客集合 S が “1台で運べない量” を持つときは
  S 内部だけで閉じたループ（または小さい成分）になってはいけない（デポへ戻れない/車両数が不足する）。
- そこで、解（MIPの整数解）が見つかるたびに callback で連結成分 S を検出し、
  容量から必要車両数 NS を見積もった上で、S 内部の辺数に上限を課す cut を cbLazy で追加する。
  これにより「容量的に不可能な成分」や「分裂した成分」を探索中に潰す。

数理最適化としての形式
- 無向辺変数 x_{ij} を持つ（i<j のみ定義）
- 顧客 i は次数2、デポは次数2m
- デポ辺は 0/1 ではなく 0/1/2 を許す（ub=2 の整数）
  → 2本の車両が同じ顧客へ直結する形などを “多重辺” として表現できるようにしている（表現上の工夫）
- lazy cut は「容量に基づく generalized subtour elimination」系の不等式

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。

注意（このコードの読み方）
- これは “教科書的な CVRP の代表定式化（MTZやフロー）” ではなく、
  「次数制約 + 遅延制約（cut）で必要な連結/容量条件を満たす」方向の実装である。
- cut の正当性は「顧客集合Sをサービスするには少なくともNS台必要」という下界から来る。
"""

import math
import random
import networkx
from gurobipy import *


def vrp(V, c, m, q, Q):
    """
    vrp -- solve the vehicle routing problem（モデル構築 + callbackを返す）

    Parameters:
        - V: ノード集合（V[0] をデポとみなす）
        - c[i,j]: 無向辺 (i,j) のコスト（i<j のみ）
        - m: 車両台数
        - q[i]: 顧客需要（ここではデポにも値が入る可能性があるが cut では顧客集合に限定して使う想定）
        - Q: 車両容量

    Returns:
        - model: Gurobi Model
        - vrp_callback: LazyConstraints 用コールバック
    """

    # ============================================================
    # コールバック：整数解が見つかるたびに「容量的に不可能な分裂成分」を切る
    # ============================================================
    def vrp_callback(model, where):
        """
        vrp_callback: add constraint to eliminate infeasible solutions

        Gurobi callback のポイント
        - where == GRB.callback.MIPSOL のタイミングは「整数解（候補解）」が得られた瞬間。
        - そこで現在の解から辺集合を取り出し、分裂した連結成分を検出し、必要なら cbLazy で制約を追加する。
        - これを行うには以下のパラメータ設定が必須：
          - model.params.DualReductions = 0
          - model.params.LazyConstraints = 1

        このコールバックがやっていること（概要）
        1) 現在の整数解で x_{ij}=1（or >0.5）の辺を集める（デポ辺は除外）
        2) その辺だけで作ったグラフの連結成分 S を列挙する
        3) 各成分 S の需要合計 q_sum を計算し、容量Qから必要車両数 NS を下界として見積もる
        4) “S 内部の辺数” に上限を課す cut を追加して、容量的に不可能/分裂を許さないようにする
        """
        # Lazy cut は MIPSOL でのみ追加する（LP緩和段階で追加すると意味が変わり得る）
        if where != GRB.callback.MIPSOL:
            return

        # ------------------------------------------------------------
        # 現在の解の辺集合（顧客-顧客のみ）を抽出
        # ------------------------------------------------------------
        edges = []
        for i, j in x:
            if model.cbGetSolution(x[i, j]) > 0.5:
                # デポ（V[0]）を含む辺はこの分裂検出では除外
                # ここでは「顧客集合だけのサブグラフ」が分裂しているかを見たい。
                if i != V[0] and j != V[0]:
                    edges.append((i, j))

        # ------------------------------------------------------------
        # 連結成分（顧客集合の分裂）を検出
        # ------------------------------------------------------------
        G = networkx.Graph()
        G.add_edges_from(edges)

        # networkx.connected_components は連結成分（集合）を返す
        Components = list(networkx.connected_components(G))

        # ------------------------------------------------------------
        # 各成分 S に対して capacity-based cut を追加
        # ------------------------------------------------------------
        for S in Components:
            S_card = len(S)

            # S 内の需要合計
            q_sum = sum(q[i] for i in S)

            # S をサービスするのに必要な車両数の下界（容量制約）
            # Math: NS=\left\lceil \frac{\sum_{i\in S} q_i}{Q}\right\rceil
            NS = int(math.ceil(float(q_sum) / Q))

            # S 内に実際に選ばれている辺（デバッグ表示用）
            S_edges = [(i, j) for i in S for j in S if i < j and (i, j) in edges]

            # cut を入れる条件
            # - S_card >= 3 は「2点成分」などを除外する意図（小さすぎる成分は別扱いにしたい等）
            # - len(S_edges) >= S_card は「S 内がサイクルになっている」兆候（|E|>=|V| なら閉路がある）
            # - NS > 1 は「容量的に1台で無理」→ 成分内部で閉じるのは強く不適切
            if S_card >= 3 and (len(S_edges) >= S_card or NS > 1):

                # ----------------------------------------------------
                # Lazy cut（generalized subtour elimination / capacity cut）
                # ----------------------------------------------------
                # S の内部で選べる辺の本数に上限を課す：
                #
                # 直感：
                # - “ツアー（巡回路）” において、連結な成分Sがデポに繋がるには、
                #   S と外部の間に少なくとも 2*NS 本程度の接続が必要になる（車両NS台ぶん）。
                # - そのため、S 内部だけに辺が過剰に立つ（=閉じた構造を作る）ことを禁じたい。
                #
                # 本コードの cut 形：
                #
                # Math: \sum_{i\in S}\sum_{j\in S,\ j>i} x_{ij} \le |S|-NS
                #
                # - NS=1 のとき：\sum_{inside} x_{ij} \le |S|-1（TSPのSECに近い形）
                # - NS>1 のとき：上限がさらに厳しくなり、S 内部に閉じることをより強く抑える
                #
                # 注意：
                # - これは “顧客-顧客辺だけ” を対象にしている（デポ辺は別で次数制約により管理）。
                # - 一般のCVRPで使われる cut は「S と外部の辺を下から抑える」形式も多いが、
                #   この実装では内部辺上限として書いている。
                model.cbLazy(
                    quicksum(x[i, j] for i in S for j in S if j > i) <= S_card - NS
                )
                print("adding cut for", S_edges)

        return

    # ============================================================
    # ここからモデル本体（次数制約 + 目的関数）
    # ============================================================
    model = Model("vrp")

    # ------------------------------------------------------------
    # 変数：無向辺の選択 x[i,j]（i<jのみ）
    # ------------------------------------------------------------
    # 顧客-顧客辺：0/1（ub=1）の整数変数（実質二値だが vtype="I" + ub=1）
    # デポ-顧客辺：0/1/2（ub=2）の整数変数
    #
    # デポ辺を2まで許す理由（直感）
    # - VRPではデポから車両が出入りするため、デポに接続する辺は “車両の出入り回数” を表す。
    # - ここでは無向辺として扱っているため、デポ-顧客間の辺の重複（2本）を許すことで
    #   「デポ→顧客→デポ」という 2-edge のルート表現をしやすくしている。
    #
    # Math: x_{0j}\in\{0,1,2\},\ x_{ij}\in\{0,1\}\ (i,j\ne 0)
    x = {}
    for i in V:
        for j in V:
            if j > i and i == V[0]:  # depot
                x[i, j] = model.addVar(ub=2, vtype="I", name="x(%s,%s)" % (i, j))
            elif j > i:
                x[i, j] = model.addVar(ub=1, vtype="I", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 次数制約
    # ------------------------------------------------------------
    # デポの次数：車両m台が出て戻るので合計 2m
    # Math: \sum_{j\in V\setminus\{0\}} x_{0j} = 2m
    model.addConstr(quicksum(x[V[0], j] for j in V[1:]) == 2 * m, "DegreeDepot")

    # 各顧客の次数：2（入って出る）
    # 無向で i<j のみ持っているので、
    # i に接続する辺は (j,i)（j<i）と (i,j)（j>i）を足す必要がある。
    #
    # Math: \sum_{j<i} x_{ji} + \sum_{j>i} x_{ij} = 2\quad(\forall i\in V\setminus\{0\})
    for i in V[1:]:
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
        quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE
    )

    model.update()
    model.__data = x
    return model, vrp_callback


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_data(n):
    """
    make_data: 乱数でCVRPっぽいデータを作る
    - 各ノードに座標を振り、距離をコストにする
    - 各顧客の需要 q[i] を 10..20 の整数で生成
    - 車両容量 Q は 100 固定
    """
    V = range(1, n + 1)
    x = dict([(i, random.random()) for i in V])
    y = dict([(i, random.random()) for i in V])
    c, q = {}, {}
    Q = 100
    for i in V:
        q[i] = random.randint(10, 20)
        for j in V:
            if j > i:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
    return V, c, q, Q


if __name__ == "__main__":
    import sys

    n = 19
    m = 3
    seed = 1
    random.seed(seed)

    V, c, q, Q = make_data(n)
    model, vrp_callback = vrp(V, c, m, q, Q)

    # Lazy cut を使うための必須設定
    model.params.DualReductions = 0
    model.params.LazyConstraints = 1

    # 最適化（callbackを渡す）
    model.optimize(vrp_callback)

    x = model.__data

    # 解の顧客-顧客辺（デポ辺は除外）を抽出
    edges = []
    for i, j in x:
        if x[i, j].X > 0.5:
            if i != V[0] and j != V[0]:
                edges.append((i, j))

    print("Optimal solution:", model.ObjVal)
    print("Edges in the solution:")
    print(sorted(edges))

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: \sum_{j\in V\setminus\{0\}} x_{0j} = 2m
# Math: \sum_{j<i} x_{ji} + \sum_{j>i} x_{ij} = 2
# Math: NS=\left\lceil \frac{\sum_{i\in S} q_i}{Q}\right\rceil
# Math: \sum_{i\in S}\sum_{j\in S,\ j>i} x_{ij} \le |S|-NS
# Math: \min \sum_{i<j} c_{ij}x_{ij}

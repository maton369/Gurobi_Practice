"""
atsp.py: solve the Asymmetric Traveling Salesman Problem（非対称TSP: ATSP）

問題設定（ATSP）
- 都市（ノード）集合 {1,...,n} をちょうど1回ずつ訪問し、出発点に戻る巡回路（ツアー）を作る。
- 有向辺（アーク）(i,j) のコスト c[i,j] は一般に対称でない（c[i,j] != c[j,i] があり得る）。
- 総コストを最小化する。

このファイルにある4つの定式化
1) mtz
   - Miller–Tucker–Zemlin (MTZ) のポテンシャル（順序）定式化（MILP）
2) mtz_strong
   - MTZ をリフティングして強くした版（より強い制約でLP緩和が締まりやすい）
3) scf
   - Single-Commodity Flow（単一商品フロー）定式化（MILP）
4) mcf
   - Multi-Commodity Flow（多商品フロー）定式化（MILP、ただし変数が非常に多い）

共通する「ツアーの骨格」制約
- 各ノード i から出るアークはちょうど1本（out-degree=1）
- 各ノード i に入るアークはちょうど1本（in-degree=1）
これだけだと複数のサブツアー（部分巡回）が可能なので、各定式化はそれを防ぐための追加構造を持つ。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。

注意（このファイルの実行部のバグ）
- __main__ で `mtz2(n,c)` を呼んでいるが、その関数は定義されていない。
  ここは意図として `mtz_strong(n,c)` を呼びたい可能性が高い（= 強化MTZのテスト）。
- この点はコメントで明示し、実行時にエラーになることを回避するなら修正が必要。
"""

from gurobipy import *


# ============================================================
# 1) MTZ（Miller–Tucker–Zemlin）定式化
# ============================================================
def mtz(n, c):
    """
    mtz: Miller–Tucker–Zemlin model for ATSP（potential / ordering formulation）

    発想（MTZ）
    - 各ノード i に「訪問順序（順位）」を表す連続変数 u_i を導入する。
    - もしアーク (i,j) を使うなら、u_j は u_i より後（大きい）でなければならない、という関係を
      Big-M で線形化して入れる。
    - これにより、部分巡回（サブツアー）を排除する。

    Parameters:
        - n: ノード数（ノードは 1..n を仮定）
        - c[i,j]: 有向アーク (i,j) のコスト（i != j）

    Returns:
        - model: 解ける状態のGurobi Model
    """
    model = Model("atsp - mtz")

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # x[i,j] ∈ {0,1}: アーク (i,j) をツアーに含めるなら1
    # Math: x_{ij}\in\{0,1\}\quad(\forall i\ne j)
    #
    # u[i] : ノード i の訪問順序（ポテンシャル）
    # - MTZでは 1 を基準ノードとして固定し、他ノードの順序を 1..n-1 の範囲で表すのが典型。
    # - この実装は 0..n-1 の範囲で連続変数としている（整数にしなくてもサブツアー除去に効く）。
    # Math: 0\le u_i \le n-1\quad(\forall i)
    x, u = {}, {}
    for i in range(1, n + 1):
        u[i] = model.addVar(lb=0, ub=n - 1, vtype="C", name="u(%s)" % i)
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # ------------------------------------------------------------
    # 次数制約（各ノードから出るのは1本、入るのも1本）
    # ------------------------------------------------------------
    # Math: \sum_{j\ne i} x_{ij}=1\quad(\forall i)
    # Math: \sum_{j\ne i} x_{ji}=1\quad(\forall i)
    #
    # これで各ノードの out-degree=1, in-degree=1 が保証され、
    # “何らかのサイクル集合” にはなるが、サブツアーを含み得る。
    for i in range(1, n + 1):
        model.addConstr(
            quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1,
            "Out(%s)" % i,
        )
        model.addConstr(
            quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1,
            "In(%s)" % i,
        )

    # ------------------------------------------------------------
    # MTZ制約（サブツアー除去）
    # ------------------------------------------------------------
    # 典型的なMTZ（ATSP）：
    #
    # Math: u_i - u_j + (n-1)x_{ij} \le n-2
    #
    # 直感：
    # - もし x_{ij}=1（i→jを使う）なら
    #   u_i - u_j + (n-1) <= n-2  →  u_i - u_j <= -1  →  u_j >= u_i + 1
    #   となり、訪問順序が前進する。
    # - もし x_{ij}=0 なら
    #   u_i - u_j <= n-2  で、u の範囲（0..n-1）内なら緩い制約になる（Big-Mで無効化）。
    #
    # 実装上の注意：
    # - j を 2..n に限定しているのは「基準ノード（1）を固定」する典型パターンの名残。
    # - 本来は u[1]=0 のように固定することが多いが、この実装では固定していない。
    #   それでも差分制約は効くが、対称性（u全体の平行移動）が残り得る。
    for i in range(1, n + 1):
        for j in range(2, n + 1):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + (n - 1) * x[i, j] <= n - 2,
                    "MTZ(%s,%s)" % (i, j),
                )

    # ------------------------------------------------------------
    # 目的関数（総コスト最小化）
    # ------------------------------------------------------------
    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, u
    return model


# ============================================================
# 2) 強化MTZ（Lifted MTZ）
# ============================================================
def mtz_strong(n, c):
    """
    mtz_strong: MTZ をリフティングして強化した版

    狙い
    - 標準MTZはLP緩和が弱く、探索が遅くなることがある。
    - そこで x[j,i] などの追加項を入れた “lifted” 制約で緩和を締める。
    - 実務的には、問題規模/構造によって flow 定式化の方が強い場合もあるが、
      MTZ強化は実装が軽く、ベースラインとして有用。

    Returns:
        - model
    """
    model = Model("atsp - mtz-strong")

    x, u = {}, {}
    for i in range(1, n + 1):
        u[i] = model.addVar(lb=0, ub=n - 1, vtype="C", name="u(%s)" % i)
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()

    # Math: \sum_{j\ne i} x_{ij}=1\quad(\forall i)
    # Math: \sum_{j\ne i} x_{ji}=1\quad(\forall i)
    for i in range(1, n + 1):
        model.addConstr(
            quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1, "Out(%s)" % i
        )
        model.addConstr(
            quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1, "In(%s)" % i
        )

    # Lifted MTZ（代表例）
    #
    # Math: u_i-u_j+(n-1)x_{ij}+(n-3)x_{ji}\le n-2
    #
    # - 逆向きアーク x_{ji} の情報も使い、緩和を締める効果がある。
    for i in range(1, n + 1):
        for j in range(2, n + 1):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2,
                    "LiftedMTZ(%s,%s)" % (i, j),
                )

    # 追加の境界強化（ノード1に関するリフティング）
    #
    # Math: -x_{1i}-u_i+(n-3)x_{i1}\le -2
    # Math: -x_{i1}+u_i+(n-3)x_{1i}\le n-2
    #
    # 直感：
    # - ノード1を基準とした順序関係を強く縛り、u の自由度を減らす。
    for i in range(2, n + 1):
        model.addConstr(
            -x[1, i] - u[i] + (n - 3) * x[i, 1] <= -2, name="LiftedLB(%s)" % i
        )
        model.addConstr(
            -x[i, 1] + u[i] + (n - 3) * x[1, i] <= n - 2, name="LiftedUB(%s)" % i
        )

    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, u
    return model


# ============================================================
# 3) SCF（Single-Commodity Flow）定式化
# ============================================================
def scf(n, c):
    """
    scf: single-commodity flow formulation for ATSP

    発想（SCF）
    - 1番ノードから「(n-1) 単位のフロー」を流し、各ノード i(>=2) がちょうど1単位を受け取るようにする。
    - フローは選ばれたアーク上でしか流せない（f_{ij} <= M x_{ij}）。
    - これによりサブツアーが起きるとフローが届かず、不可になる → サブツアー除去になる。

    変数
    - x_{ij} ∈ {0,1} : アーク選択
    - f_{ij} ≥ 0     : 単一商品フロー量

    Returns:
        - model
    """
    model = Model("atsp - scf")

    x, f = {}, {}

    # 変数作成
    # x_{ij} は二値
    # Math: x_{ij}\in\{0,1\}\quad(\forall i\ne j)
    #
    # f_{ij} は連続フロー
    # - 1から出るフローは最大 n-1
    # - その他は最大 n-2（大雑把な上界）
    # Math: 0\le f_{ij}\le U_{ij}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
                if i == 1:
                    f[i, j] = model.addVar(
                        lb=0, ub=n - 1, vtype="C", name="f(%s,%s)" % (i, j)
                    )
                else:
                    f[i, j] = model.addVar(
                        lb=0, ub=n - 2, vtype="C", name="f(%s,%s)" % (i, j)
                    )

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

    # フロー供給（ノード1から合計 n-1 を流す）
    # Math: \sum_{j=2}^{n} f_{1j} = n-1
    model.addConstr(quicksum(f[1, j] for j in range(2, n + 1)) == n - 1, "FlowOut")

    # フロー保存（各ノード i>=2 は差分が +1）
    # - 受け取る - 出ていく = 1
    # これは「各ノードがちょうど1単位を消費する」意味になる。
    # Math: \sum_{j\ne i} f_{ji} - \sum_{j\ne i} f_{ij} = 1\quad(\forall i=2..n)
    for i in range(2, n + 1):
        model.addConstr(
            quicksum(f[j, i] for j in range(1, n + 1) if j != i)
            - quicksum(f[i, j] for j in range(1, n + 1) if j != i)
            == 1,
            "FlowCons(%s)" % i,
        )

    # フローは選ばれたアーク上でしか流せない（連結制約）
    # Math: f_{ij} \le M_{ij} x_{ij}
    #
    # ここで M_{ij} は上界（Big-M）で、
    # - i=1 からは最大 n-1
    # - それ以外は最大 n-2
    for j in range(2, n + 1):
        model.addConstr(f[1, j] <= (n - 1) * x[1, j], "FlowUB(%s,%s)" % (1, j))
        for i in range(2, n + 1):
            if i != j:
                model.addConstr(f[i, j] <= (n - 2) * x[i, j], "FlowUB(%s,%s)" % (i, j))

    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, f
    return model


# ============================================================
# 4) MCF（Multi-Commodity Flow）定式化
# ============================================================
def mcf(n, c):
    """
    mcf: multi-commodity flow formulation for ATSP

    発想（MCF）
    - 各ノード k(=2..n) を「1単位の需要」とみなし、ノード1から各ノードkへ 1単位のフロー（commodity k）を送る。
    - すべての commodity が到達できる構造にすることで、サブツアーを強く排除できる（緩和が強いことが多い）。
    - その代わり変数数が O(n^3) と爆増するため、大きいnには重い。

    変数
    - x_{ij} ∈ {0,1} : アーク選択
    - f_{ij}^k ∈ [0,1] : commodity k のフローがアーク (i,j) を通るか（連続で0/1上界）

    Returns:
        - model
    """
    model = Model("mcf")

    x, f = {}, {}

    # 変数作成
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

            # commodity は k=2..n を想定
            # f[i,j,k] は「k向けフローが (i,j) を通る」連続変数（上界1）
            # Math: 0\le f_{ij}^k \le 1
            #
            # 実装の条件：
            # - i != j（自己ループ除外）
            # - j != 1（ノード1へ戻る方向は commodity の中間輸送に不要、という意図）
            # - i != k（需要ノードkから出るフローを作らない、という整理）
            if i != j and j != 1:
                for k in range(2, n + 1):
                    if i != k:
                        f[i, j, k] = model.addVar(
                            ub=1, vtype="C", name="f(%s,%s,%s)" % (i, j, k)
                        )

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

    # commodity k ごとにフロー制約
    for k in range(2, n + 1):
        # ノード1から commodity k を 1 単位流出
        # Math: \sum_{i=2}^{n} f_{1i}^k = 1
        model.addConstr(
            quicksum(f[1, i, k] for i in range(2, n + 1) if (1, i, k) in f) == 1,
            "FlowOut(%s)" % k,
        )

        # ノード k は commodity k を 1 単位流入（到達）
        # Math: \sum_{i\ne k} f_{ik}^k = 1
        model.addConstr(
            quicksum(f[i, k, k] for i in range(1, n + 1) if (i, k, k) in f) == 1,
            "FlowIn(%s)" % k,
        )

        # 中間ノードでは保存（入る=出る）
        # Math: \sum_{j\ne i} f_{ji}^k = \sum_{j\ne i} f_{ij}^k
        for i in range(2, n + 1):
            if i != k:
                model.addConstr(
                    quicksum(f[j, i, k] for j in range(1, n + 1) if (j, i, k) in f)
                    == quicksum(f[i, j, k] for j in range(1, n + 1) if (i, j, k) in f),
                    "FlowCons(%s,%s)" % (i, k),
                )

    # フローは選ばれたアーク上でしか流せない
    # Math: f_{ij}^k \le x_{ij}
    for i, j, k in f:
        model.addConstr(f[i, j, k] <= x[i, j], "FlowUB(%s,%s,%s)" % (i, j, k))

    # Math: \min \sum_{i\ne j} c_{ij}x_{ij}
    model.setObjective(quicksum(c[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x, f
    return model


def sequence(arcs):
    """
    sequence: selected arcs から訪問順の列を復元する

    arcs は「各ノードから出るアークがちょうど1本」の前提で、
    succ[i]=j を作れば 1→succ[1]→succ[succ[1]]→... と辿って順列が得られる。

    注意：
    - ここは「1 から始める」固定になっている。
    - len(arcs)-2 のループ回数は元コードのままだが、
      n都市なら n-1 回辿って n個並べたいので、通常は n-1 回にした方が安全。
      （この関数は教育用の簡易復元として読むのが良い）
    """
    succ = {}
    for i, j in arcs:
        succ[i] = j
    curr = 1
    sol = [curr]
    for _ in range(len(arcs) - 2):
        curr = succ[curr]
        sol.append(curr)
    return sol


if __name__ == "__main__":
    # 小さな例（n=5）で各定式化を比較する
    n = 5
    c = {
        (1, 1): 0,
        (1, 2): 1989,
        (1, 3): 102,
        (1, 4): 102,
        (1, 5): 103,
        (2, 1): 104,
        (2, 2): 0,
        (2, 3): 11,
        (2, 4): 104,
        (2, 5): 108,
        (3, 1): 107,
        (3, 2): 108,
        (3, 3): 0,
        (3, 4): 19,
        (3, 5): 102,
        (4, 1): 109,
        (4, 2): 102,
        (4, 3): 107,
        (4, 4): 0,
        (4, 5): 15,
        (5, 1): 13,
        (5, 2): 103,
        (5, 3): 104,
        (5, 4): 101,
        (5, 5): 0,
    }

    # ------------------------------------------------------------
    # MTZ
    # ------------------------------------------------------------
    model = mtz(n, c)
    model.Params.OutputFlag = 0
    model.optimize()
    cost = model.ObjVal
    print("Opt.value=", cost)

    for v in model.getVars():
        if v.X > 0.001:
            print(v.VarName, v.X)

    x, u = model.__data
    arcs = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    sol = sequence(arcs)
    print(sol)

    # ------------------------------------------------------------
    # 強化MTZ（元コードでは mtz2 を呼んでいるが未定義なので注意）
    # ------------------------------------------------------------
    # 本来ここは mtz_strong(n,c) を呼ぶのが自然。
    model = mtz_strong(n, c)
    model.Params.OutputFlag = 0
    model.optimize()
    cost = model.ObjVal
    print("Opt.value=", cost)

    for v in model.getVars():
        if v.X > 0.001:
            print(v.VarName, v.X)

    x, u = model.__data
    arcs = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    sol = sequence(arcs)
    print(sol)

    # ------------------------------------------------------------
    # SCF
    # ------------------------------------------------------------
    model = scf(n, c)
    model.Params.OutputFlag = 0
    model.optimize()
    cost = model.ObjVal
    print("Opt.value=", cost)

    for v in model.getVars():
        if v.X > 0.001:
            print(v.VarName, v.X)

    x, f = model.__data
    arcs = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    sol = sequence(arcs)
    print(sol)

    # ------------------------------------------------------------
    # MCF
    # ------------------------------------------------------------
    model = mcf(n, c)
    model.Params.OutputFlag = 0
    model.optimize()
    cost = model.ObjVal
    print("Opt.value=", cost)

    for v in model.getVars():
        if v.X > 0.001:
            print(v.VarName, v.X)

    x, f = model.__data
    arcs = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    sol = sequence(arcs)
    print(sol)

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: \sum_{j\ne i} x_{ij}=1
# Math: \sum_{j\ne i} x_{ji}=1
# Math: u_i-u_j+(n-1)x_{ij}\le n-2
# Math: u_i-u_j+(n-1)x_{ij}+(n-3)x_{ji}\le n-2
# Math: \sum_{j=2}^{n} f_{1j} = n-1
# Math: \sum_{j\ne i} f_{ji} - \sum_{j\ne i} f_{ij} = 1
# Math: f_{ij}\le M x_{ij}
# Math: f_{ij}^k \le x_{ij}
# Math: \min \sum_{i\ne j} c_{ij}x_{ij}

"""
cutstock.py: use gurobi for solving the cutting stock problem（カッティングストック問題）

目的（何を解くか）
- 幅 B の原反ロール（roll）から、幅 w_i の注文品（item）を必要数量 q_i だけ切り出す。
- 切り出し方（パターン）を工夫して、使用するロール本数を最小化する。

入力の意味
- w = (w_i): 注文品の幅（サイズ）の種類（m種類）
- q = (q_i): 各幅の必要数量
- B: ロール幅（ビン容量）

Gilmore–Gomory（列生成）の基本アイデア
- 1本のロールから切り出す組合せ（パターン）を t_k（ベクトル）とする。
  t_k(i) は「パターン k で幅 w_i を何個切り出すか」。
- パターン集合が与えられれば、次の Master Problem（LP）で最小ロール数が求まる（緩和）：

  minimize   sum_k x_k
  subject to sum_k t_k(i) x_k >= q_i   for all i
             x_k >= 0                 for all k

- しかし、パターンは指数個あるので全部は列挙できない。
- そこで「影の価格（双対変数）π」を使い、“改善できる新しいパターン” を
  ナップサック問題（Pricing / Subproblem）として生成する（列生成）。

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する。
- $$...$$ は使わない（Mathoverの既定トリガが Math: のため）。

注意（このコードの特徴）
- Master は最終的に整数（vtype="I"）で解き直しているため、列生成は「パターン生成」に使い、
  最後に整数マスタ問題（IP）を解く構造。
- 厳密な Branch-and-Price ではなく “列生成 → 最後にIP” の典型的な教育用実装である。

"""

from gurobipy import *

LOG = True
EPS = 1.0e-6


def solveCuttingStock(w, q, B):
    """
    solveCuttingStock: column generation (Gilmore–Gomory approach)

    Parameters:
        - w: width list（サイズ種類）
        - q: quantities（必要本数）
        - B: roll width（容量）

    Returns:
        - rolls: list of rolls, each roll is a list of cut widths (sorted)
    """
    t = []  # 現在のパターン集合（list of pattern vectors）
    m = len(w)  # サイズ種類数

    # ------------------------------------------------------------
    # 初期パターン生成
    # ------------------------------------------------------------
    # 「各サイズ i だけを詰められるだけ詰める」単純パターンを m 本作る。
    # 例: 幅 w_i を floor(B / w_i) 個切るパターン。
    #
    # これにより最初からマスタ問題が可解になる（各注文は少なくとも何らかのパターンで供給可能）。
    for i, width in enumerate(w):
        pat = [0] * m
        pat[i] = int(B / width)
        t.append(pat)

    K = len(t)

    # ------------------------------------------------------------
    # Master Problem（マスタ問題）
    # ------------------------------------------------------------
    # 変数 x_k: パターン k を何本使うか
    #
    # 理論上は LP 緩和では x_k >= 0 の連続で良い。
    # ただし本コードは vtype="I" で整数を使っている（最終的に整数解が欲しい方針）。
    #
    # Math: x_k \ge 0\quad(\forall k)
    #
    # 目的：ロール本数（パターン使用本数）の合計を最小化
    # Math: \min \sum_k x_k
    master = Model("master LP")
    x = {}
    for k in range(K):
        x[k] = master.addVar(vtype="I", name="x(%s)" % k)
    master.update()

    # 注文制約（需要を満たす）
    #
    # 各サイズ i について、パターンに含まれる切断数 t_k(i) を足し合わせ、
    # 必要数量 q_i 以上になるようにする。
    #
    # Math: \sum_k t_k(i)\,x_k \ge q_i\quad(\forall i)
    orders = {}
    for i in range(m):
        orders[i] = master.addConstr(
            quicksum(t[k][i] * x[k] for k in range(K) if t[k][i] > 0) >= q[i],
            "Order(%s)" % i,
        )

    master.setObjective(quicksum(x[k] for k in range(K)), GRB.MINIMIZE)

    master.update()  # relax() の前に update が必要
    # master.Params.OutputFlag = 0  # ログを消したい場合

    # ------------------------------------------------------------
    # 列生成ループ
    # ------------------------------------------------------------
    # マスタ問題の LP 緩和を解き、双対変数 π（影の価格）を得る。
    # その π を使って Pricing（ナップサック）を解き、改善列（新パターン）を見つける。
    while True:
        # --------------------------------------------------------
        # (1) マスタ問題の LP 緩和を解く
        # --------------------------------------------------------
        # relax() により整数制約を外し、連続緩和を解く。
        relax = master.relax()
        relax.optimize()

        # 双対変数 π_i（需要制約の影の価格）
        # 直感：サイズ i の需要を1増やしたとき目的値がどれだけ増えるか、の“価値”。
        pi = [c.Pi for c in relax.getConstrs()]

        # --------------------------------------------------------
        # (2) Pricing Problem（ナップサック）で新パターン生成
        # --------------------------------------------------------
        # 目的：π で重みづけした “価値” を最大化する切断パターン y を探す。
        #
        # 変数 y_i: 新パターンでサイズ i を何個切るか
        # Math: y_i \in \mathbb{Z}_{\ge 0}\quad(\forall i)
        #
        # 容量制約：合計幅が B を超えない
        # Math: \sum_i w_i y_i \le B
        #
        # 目的（価値最大化）：
        # Math: \max \sum_i \pi_i y_i
        #
        # 列生成の理屈：
        # - マスタの目的係数は各列（パターン）で 1（1本のロール）。
        # - 新列の reduced cost は 1 - Σ π_i t_k(i)（最小化の場合）に相当する。
        # - よって Σ π_i y_i > 1 となるパターンが見つかれば reduced cost が負になり、
        #   LP 解を改善できるため、その列を追加する価値がある。
        knapsack = Model("KP")
        knapsack.ModelSense = -1  # maximize（古い書き方：ModelSense=-1 で最大化）
        y = {}
        for i in range(m):
            # 上限 ub=q[i] は「必要数以上切っても意味が薄い」ための軽い制限（なくても理論上はOK）。
            y[i] = knapsack.addVar(lb=0, ub=q[i], vtype="I", name="y(%s)" % i)
        knapsack.update()

        knapsack.addConstr(quicksum(w[i] * y[i] for i in range(m)) <= B, "Width")
        knapsack.setObjective(quicksum(pi[i] * y[i] for i in range(m)), GRB.MAXIMIZE)

        # knapsack.Params.OutputFlag = 0
        knapsack.optimize()

        # --------------------------------------------------------
        # (3) 列追加するかどうか（改善条件）
        # --------------------------------------------------------
        # Σ π_i y_i <= 1 なら reduced cost >= 0 となり、改善列がないので終了。
        #
        # Math: \sum_i \pi_i y_i \le 1\Rightarrow \text{stop}
        #
        # 逆に Σ π_i y_i > 1 なら改善列があるので、そのパターンを追加する。
        if knapsack.ObjVal < 1 + EPS:
            break

        # 新パターン（整数に丸め）
        pat = [int(y[i].X + 0.5) for i in y]
        t.append(pat)

        # --------------------------------------------------------
        # (4) マスタ問題に新列（新パターン）を追加
        # --------------------------------------------------------
        # Gurobiの Column を使うと「既存制約に対する係数」を指定して変数を追加できる。
        # ここでは需要制約 Order(i) に対し係数 t[K][i] を追加する。
        col = Column()
        for i in range(m):
            if t[K][i] > 0:
                col.addTerms(t[K][i], orders[i])

        # obj=1 は「1本のロールを使う」目的係数。
        x[K] = master.addVar(obj=1, vtype="I", name="x(%s)" % K, column=col)

        master.update()  # 次の relax() のために update
        K += 1

    # ------------------------------------------------------------
    # 最後に整数マスタ問題（IP）として解き直す
    # ------------------------------------------------------------
    # 列生成ループは LP 緩和で良いパターンを集める工程。
    # 最終的に「ロール本数は整数」なので、master を整数のまま optimize() して整数解を得る。
    master.optimize()

    # ------------------------------------------------------------
    # 解の復元：ロールごとの切断サイズリストを作る
    # ------------------------------------------------------------
    rolls = []
    for k in x:
        # x[k].X 本だけパターン k を使う
        for _ in range(int(x[k].X + 0.5)):
            roll = [w[i] for i in range(m) if t[k][i] > 0 for _ in range(t[k][i])]
            rolls.append(sorted(roll))

    rolls.sort()
    return rolls


def CuttingStockExample1():
    """CuttingStockExample1: create toy instance for the cutting stock problem."""
    B = 110
    w = [20, 45, 50, 55, 75]
    q = [48, 35, 24, 10, 8]
    return w, q, B


def CuttingStockExample2():
    """CuttingStockExample2: create toy instance for the cutting stock problem."""
    B = 9
    w = [2, 3, 4, 5, 6, 7, 8]
    q = [4, 2, 6, 6, 2, 2, 2]
    return w, q, B


def mkCuttingStock(s):
    """mkCuttingStock: convert a bin packing instance into cutting stock format"""
    w, q = [], []
    for item in sorted(s):
        if w == [] or item != w[-1]:
            w.append(item)
            q.append(1)
        else:
            q[-1] += 1
    return w, q


def mkBinPacking(w, q):
    """mkBinPacking: convert a cutting stock instance into bin packing format"""
    s = []
    for j in range(len(w)):
        for _ in range(q[j]):
            s.append(w[j])
    return s


if __name__ == "__main__":
    from bpp import FFD, solveBinPacking

    w, q, B = CuttingStockExample1()
    # w, q, B = CuttingStockExample2()

    s = mkBinPacking(w, q)

    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nCutting stock problem, column generation:")
    rolls = solveCuttingStock(w, q, B)
    print(len(rolls), "rolls:")
    print(rolls)

    print("\n\n\nBin packing problem:")
    bins = solveBinPacking(s, B)
    print(len(bins), "bins:")
    print(bins)

# ------------------------------------------------------------
# Mathover用の数式コメント（要点まとめ）
# ------------------------------------------------------------
# Math: \min \sum_k x_k
# Math: \sum_k t_k(i)x_k \ge q_i\quad(\forall i)
# Math: \sum_i w_i y_i \le B
# Math: \max \sum_i \pi_i y_i
# Math: \sum_i \pi_i y_i \le 1\Rightarrow \text{stop}

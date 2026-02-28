"""
bpp.py: use gurobi for solving the bin packing problem（ビンパッキング問題: BPP）

目的（何を解くか）
- アイテム i（サイズ s[i]）を、容量 B のビン（箱）に詰める。
- 各ビンの合計サイズが B を超えないようにしつつ、使うビン数を最小化する。

入力の意味
- s = (s_i): アイテムのサイズ（幅・重量など）
- B: ビン容量

この実装の構成
1) FFD（First Fit Decreasing）で上界 U（使うビン数の上限）を作る
2) Martello and Toth (1990) の 0-1 整数計画（IP）定式化で最適解を求める
3) オプションで tie-breaking 制約や SOS 制約による改善を示唆（コードではコメントアウト）

数理最適化としての形式
- 変数が二値（0/1）で、制約・目的は線形
- よって **0-1混合整数線形計画（0-1 MIP / MILP）** である

数式コメント方針（Mathover対応）
- ソースコード内の数式コメントは `# Math: <LaTeX>` の1行形式に統一する
- レビュー本文ではLaTeXを使うが、ここはソースコードなので Mathover 形式を採用する

"""

from gurobipy import *


def FFD(s, B):
    """
    First Fit Decreasing (FFD) heuristics for the Bin Packing Problem.

    アルゴリズム（FFD）の概要
    - アイテムを大きい順に並べる（Decreasing）
    - 各アイテムを「入る最初のビン」に入れる（First Fit）
    - どのビンにも入らなければ新しいビンを開く

    Parameters:
        - s: item sizes（アイテムサイズのリスト）
        - B: bin capacity（ビン容量）

    Returns:
        - sol: list of bins, each bin is a list of item sizes in that bin
    """
    remain = [B]  # 各ビンの残容量
    sol = [[]]  # 各ビンに入っているアイテム（サイズ）一覧

    # 大きい順に処理するのが FFD の肝（小さい順だとビンが増えやすい）
    for item in sorted(s, reverse=True):
        # 既存のビンを先頭から順に見て、入る最初のビンに入れる
        for j, free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else:
            # どのビンにも入らないなら新しいビンを作る
            sol.append([item])
            remain.append(B - item)

    return sol


def bpp(s, B):
    """
    bpp: Martello and Toth's model to solve the bin packing problem（IP定式化）

    Parameters:
        - s: item sizes
        - B: bin capacity

    Returns:
        - model: 目的関数・制約まで設定済みのGurobi Model
    """
    n = len(s)

    # ------------------------------------------------------------
    # 上界 U を FFD で得る
    # ------------------------------------------------------------
    # U は「使うビン数」の上限で、モデル内でビン候補を 0..U-1 用意するために使う。
    # FFD は最適ではないが高速で、現実的な上界をくれる。
    U = len(FFD(s, B))

    model = Model("bpp")
    # model.setParam("MIPFocus", 1)  # 探索方針調整（必要なら）

    # ------------------------------------------------------------
    # 変数
    # ------------------------------------------------------------
    # x[i,j] ∈ {0,1}: アイテム i をビン j に入れるなら1
    # Math: x_{ij} \in \{0,1\}\quad(\forall i\in\{0,\dots,n-1\},\ \forall j\in\{0,\dots,U-1\})
    #
    # y[j] ∈ {0,1}: ビン j を使うなら1（少なくとも1つ入れるなら開く）
    # Math: y_j \in \{0,1\}\quad(\forall j\in\{0,\dots,U-1\})
    #
    # x と y を分ける典型理由：
    # - x だけだと「ビンを使ったか」を目的に書きにくい
    # - y を導入して「容量制約を y でON/OFF」し、目的で Σ y を最小化するのが定石
    x, y = {}, {}
    for i in range(n):
        for j in range(U):
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    for j in range(U):
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)

    model.update()

    # ------------------------------------------------------------
    # 制約（1）割当制約：各アイテムは必ずどれか1つのビンへ
    # ------------------------------------------------------------
    # Math: \sum_{j=0}^{U-1} x_{ij} = 1\quad(\forall i\in\{0,\dots,n-1\})
    #
    # これにより「各アイテムはちょうど1つのビン」に入る。
    # 分割（分割投入）を許さない、純粋なビンパッキングの条件である。
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(U)) == 1, "Assign(%s)" % i)

    # ------------------------------------------------------------
    # 制約（2）容量制約：使うビンなら容量 B 以内
    # ------------------------------------------------------------
    # Math: \sum_{i=0}^{n-1} s_i x_{ij} \le B\,y_j\quad(\forall j\in\{0,\dots,U-1\})
    #
    # - y_j = 0 のとき右辺0なので、x_{ij} はすべて0を強制される（実質そのビンは使えない）
    # - y_j = 1 のとき通常の容量制約
    #
    # これは Big-M 連結（M=B）で、ビン使用のON/OFFを表現している。
    for j in range(U):
        model.addConstr(
            quicksum(s[i] * x[i, j] for i in range(n)) <= B * y[j], "Capac(%s)" % j
        )

    # ------------------------------------------------------------
    # 制約（3）強化（連結）制約：閉じたビンには入れられない
    # ------------------------------------------------------------
    # Math: x_{ij}\le y_j\quad(\forall i,\forall j)
    #
    # 容量制約だけでも y_j=0 なら合計が0なので間接的に効くが、
    # x_{ij} を個別に縛ることで探索が締まり、MIPが速くなることがある。
    for j in range(U):
        for i in range(n):
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    # ------------------------------------------------------------
    # 追加の改善案（コメントアウトされている部分）
    # ------------------------------------------------------------
    # 1) tie breaking（対称性の破壊）
    # - ビンはラベルが違うだけで同等（対称性が強い）なので、探索が遅くなりやすい。
    # - y[j] >= y[j+1] のように「前のビンから順に使う」形を強制すると、同型解を減らせる。
    #
    # Math: y_j \ge y_{j+1}\quad(\forall j\in\{0,\dots,U-2\})
    #
    # 2) SOS（Special Ordered Set）制約
    # - 各 i について x[i,0..U-1] のうち1つだけが1になることを SOS1 で与える。
    # - 割当制約と同義だが、ソルバがより効率よく扱える場合がある。
    #
    # Math: \text{SOS1}(x_{i0},x_{i1},\dots,x_{i,U-1})
    #
    # 注意：
    # - Gurobi では addSOS(1, [...]) のように追加する。
    # - ただし割当制約（等式）と併用すると冗長になるので、どちらを採用するかは実験で決める。
    #
    ##    # tie breaking constraints
    ##    for j in range(U-1):
    ##        model.addConstr(y[j] >= y[j+1), "TieBrk(%s)"%j)
    ##
    ##    # SOS constraints
    ##    for i in range(n):
    ##        model.addSOS(1,[x[i,j] for j in range(U)])

    # ------------------------------------------------------------
    # 目的関数：使うビン数の最小化
    # ------------------------------------------------------------
    # Math: \min \sum_{j=0}^{U-1} y_j
    model.setObjective(quicksum(y[j] for j in range(U)), GRB.MINIMIZE)

    model.update()
    model.__data = x, y
    return model


def solveBinPacking(s, B):
    """
    solveBinPacking: IPモデルでビンパッキング問題を解く。

    Parameters:
        - s: item sizes
        - B: bin capacity

    Returns:
        - bins: 各ビンに入ったアイテムサイズのリスト（空ビンは除去）
    """
    n = len(s)
    U = len(FFD(s, B))

    # モデル構築→最適化
    model = bpp(s, B)
    x, y = model.__data
    model.optimize()

    # 解の復元：x[i,j]=1 ならアイテム i をビン j に入れる
    bins = [[] for _ in range(U)]
    for i, j in x:
        if x[i, j].X > 0.5:
            bins[j].append(s[i])

    # 空ビンを削除し、見やすいように整形
    for _ in range(bins.count([])):
        bins.remove([])
    for b in bins:
        b.sort()
    bins.sort()

    return bins


import random


def DiscreteUniform(n=10, LB=1, UB=99, B=100):
    """
    DiscreteUniform: ランダムなBPPインスタンスを作る（サイズは整数一様乱数）

    注意：
    - 引数 B があるが、関数内で B=100 に上書きしている（元コードの挙動）。
      本当に引数 B を使いたいなら上書きを削除する方が自然。
    """
    B = 100
    s = [0] * n
    for i in range(n):
        s[i] = random.randint(LB, UB)
    return s, B


if __name__ == "__main__":
    random.seed(256)
    s, B = DiscreteUniform()

    # print は Python3 形式に統一
    print("items:", s)
    print("bin size:", B)

    # FFD の解（上界・ヒューリスティック解）
    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    # MIP 最適解
    print("\n\n\nBin packing problem:")
    bins = solveBinPacking(s, B)
    print(len(bins), "bins:")
    print(bins)

# ------------------------------------------------------------
# モデルまとめ（Mathover用）
# ------------------------------------------------------------
# Math: x_{ij} \in \{0,1\}
# Math: y_j \in \{0,1\}
# Math: \sum_{j=0}^{U-1} x_{ij} = 1
# Math: \sum_{i=0}^{n-1} s_i x_{ij} \le B\,y_j
# Math: x_{ij}\le y_j
# Math: \min \sum_{j=0}^{U-1} y_j

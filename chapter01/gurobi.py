from gurobipy import *

model = Model("lol")
x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(name="x2")
x3 = model.addVar(ub=30, name="x3")
model.update()

model.addConstr(2 * x1 + x2 + x3 <= 60)
model.addConstr(x1 + 2 * x2 + x3 <= 60)

model.setObjective(15 * x1 + 18 * x2 + 30 * x3, GRB.MAXIMIZE)

model.optimize()

for v in model.getVars():
    print(f"{v.varName} = {v.x}")

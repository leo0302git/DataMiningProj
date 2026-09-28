# 回归 RRL 规则图

Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。

L1R0: Diameter > 0.305 AND Shell weight > 0.16 AND Shucked weight <= 0.4445; Support=0.332735
L1R1: Shell weight <= 0.23325; Support=0.500000
L1R2: TRUE; Support=1.000000
L1R3: TRUE; Support=1.000000
L1R4: TRUE; Support=1.000000
L1R5: TRUE; Support=1.000000
L1R6: TRUE; Support=1.000000
L1R7: Shell weight <= 0.37; Support=0.836026
L1R8: Shucked weight > 0.2345 AND Shell weight <= 0.23325; Support=0.172053
L1R9: Height > 0.125 AND Shucked weight <= 0.33375; Support=0.147816
L1R10: TRUE; Support=1.000000
L1R11: Shell weight > 0.23325; Support=0.500000
L1R12: TRUE; Support=1.000000
L1R13: Shucked weight > 0.33375 AND Height <= 0.175; Support=0.352184
L1R14: Height > 0.1; Support=0.810293
L1R15: Sex=I AND Height <= 0.14; Support=0.280072
L1R16: Shell weight > 0.293; Support=0.333034
L1R17: FALSE; Support=0.000000
L1R18: Shucked weight > 0.57558332 OR Shell weight <= 0.37; Support=0.948534
L1R19: FALSE; Support=0.000000
L1R20: FALSE; Support=0.000000
L1R21: Shucked weight > 0.57558332 OR Shell weight <= 0.16; Support=0.500898
L1R22: FALSE; Support=0.000000
L1R23: Shucked weight > 0.4445 OR Shell weight <= 0.293; Support=0.933273
L1R24: FALSE; Support=0.000000
L1R25: Length > 0.4 OR Height > 0.1; Support=0.846499
L1R26: FALSE; Support=0.000000
L1R27: Shell weight > 0.293; Support=0.333034
L1R28: FALSE; Support=0.000000
L1R29: FALSE; Support=0.000000
L1R30: FALSE; Support=0.000000
L1R31: FALSE; Support=0.000000

Rings = 9.86780453 + sum(weight_j * rule_j)

L1R0: 0.949832797 Rings
L1R1: -0.196444184 Rings
L1R2: 0.242546171 Rings
L1R3: 0.193144202 Rings
L1R4: 0.186862156 Rings
L1R5: 0.131601259 Rings
L1R6: 0.207740471 Rings
L1R7: -1.15019882 Rings
L1R8: -0.903517067 Rings
L1R9: 0.837798297 Rings
L1R10: 0.220744163 Rings
L1R11: 0.378164083 Rings
L1R12: 0.200772077 Rings
L1R13: -0.782796025 Rings
L1R14: 0.564926207 Rings
L1R15: -1.31084454 Rings
L1R16: 0.705412686 Rings
L1R17: -4.62067098e-39 Rings
L1R18: -0.808320403 Rings
L1R19: 1.18884606e-38 Rings
L1R20: 1.57671301e-40 Rings
L1R21: -0.805398643 Rings
L1R22: -1.6409205e-42 Rings
L1R23: -0.81084913 Rings
L1R24: 4.66857998e-40 Rings
L1R25: 1.19855094 Rings
L1R26: -1.75083835e-39 Rings
L1R27: 0.433091998 Rings
L1R28: 1.24184471e-40 Rings
L1R29: -5.22511967e-40 Rings
L1R30: -1.07988264e-40 Rings
L1R31: -3.04971591e-40 Rings

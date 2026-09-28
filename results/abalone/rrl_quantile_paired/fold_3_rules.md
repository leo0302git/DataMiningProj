# 回归 RRL 规则图

Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。

L1R0: Diameter > 0.375 AND Shell weight > 0.23325; Support=0.497008
L1R1: TRUE; Support=1.000000
L1R2: Height <= 0.1 AND Height <= 0.125; Support=0.189707
L1R3: Shell weight <= 0.37; Support=0.836026
L1R4: Shell weight > 0.16; Support=0.665769
L1R5: Height > 0.175 AND Shucked weight <= 0.57558332; Support=0.057151
L1R6: Shell weight <= 0.37; Support=0.836026
L1R7: TRUE; Support=1.000000
L1R8: Sex=I AND Height <= 0.125; Support=0.236984
L1R9: Height > 0.1 AND Length <= 0.545 AND Shucked weight <= 0.2345; Support=0.148713
L1R10: Whole weight > 1.2975; Support=0.166367
L1R11: TRUE; Support=1.000000
L1R12: TRUE; Support=1.000000
L1R13: Sex=I; Support=0.321364
L1R14: TRUE; Support=1.000000
L1R15: Height > 0.16 AND Shucked weight <= 0.57558332; Support=0.148115
L1R16: FALSE; Support=0.000000
L1R17: Shell weight > 0.293; Support=0.333034
L1R18: FALSE; Support=0.000000
L1R19: FALSE; Support=0.000000
L1R20: Sex=I OR Shucked weight > 0.33375 OR Height <= 0.125 OR Shell weight <= 0.09; Support=0.898265
L1R21: FALSE; Support=0.000000
L1R22: FALSE; Support=0.000000
L1R23: FALSE; Support=0.000000
L1R24: Shucked weight > 0.57558332 OR Shell weight <= 0.293; Support=0.823758
L1R25: FALSE; Support=0.000000
L1R26: Shucked weight > 0.4445; Support=0.333034
L1R27: FALSE; Support=0.000000
L1R28: FALSE; Support=0.000000
L1R29: FALSE; Support=0.000000
L1R30: Shucked weight > 0.4445 OR Diameter <= 0.305; Support=0.507780
L1R31: FALSE; Support=0.000000

Rings = 10.8006287 + sum(weight_j * rule_j)

L1R0: 1.54001331 Rings
L1R1: 0.313637763 Rings
L1R2: -0.733485341 Rings
L1R3: -0.103783228 Rings
L1R4: 0.541716218 Rings
L1R5: 0.648087442 Rings
L1R6: -1.53220713 Rings
L1R7: 0.0721162781 Rings
L1R8: -0.809524238 Rings
L1R9: 0.786737621 Rings
L1R10: 0.321677148 Rings
L1R11: 0.0935558975 Rings
L1R12: 0.22277382 Rings
L1R13: -0.255631059 Rings
L1R14: 0.192269981 Rings
L1R15: 0.376560658 Rings
L1R16: -3.1546031e-40 Rings
L1R17: 0.54381609 Rings
L1R18: 0.00658998638 Rings
L1R19: -5.0599206e-40 Rings
L1R20: -0.664524078 Rings
L1R21: 2.22873718e-40 Rings
L1R22: -0.0680157617 Rings
L1R23: -3.44223363e-40 Rings
L1R24: -0.76426059 Rings
L1R25: -3.07405647e-40 Rings
L1R26: -0.204623982 Rings
L1R27: -5.2240687e-40 Rings
L1R28: 9.36137439e-40 Rings
L1R29: 7.00556606e-39 Rings
L1R30: -1.02166605 Rings
L1R31: 0.111081131 Rings

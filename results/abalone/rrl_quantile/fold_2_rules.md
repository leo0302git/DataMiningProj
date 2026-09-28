# 回归 RRL 规则图

Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。

L1R0: TRUE; Support=1.000000
L1R1: Shell weight <= 0.093999993; Support=0.166966
L1R2: Shell weight > 0.295; Support=0.324656
L1R3: Height > 0.16 AND Shucked weight <= 0.57550001; Support=0.145721
L1R4: TRUE; Support=1.000000
L1R5: Sex=I AND Height <= 0.14 AND Shell weight <= 0.295; Support=0.271694
L1R6: Height > 0.16; Support=0.285757
L1R7: Diameter > 0.31 AND Shell weight > 0.093999993 AND Length <= 0.545 AND Shucked weight <= 0.23633333; Support=0.153501
L1R8: TRUE; Support=1.000000
L1R9: Shucked weight <= 0.57550001; Support=0.833932
L1R10: TRUE; Support=1.000000
L1R11: Whole weight > 1.0316667 AND Viscera weight > 0.12 AND Shell weight > 0.3705; Support=0.163974
L1R12: Shucked weight <= 0.3345; Support=0.500299
L1R13: TRUE; Support=1.000000
L1R14: TRUE; Support=1.000000
L1R15: Height <= 0.1 AND Shell weight <= 0.23125; Support=0.180730
L1R16: FALSE; Support=0.000000
L1R17: Shucked weight > 0.44166667 OR Whole weight <= 0.7955 OR Shell weight <= 0.23125; Support=0.854877
L1R18: FALSE; Support=0.000000
L1R19: FALSE; Support=0.000000
L1R20: FALSE; Support=0.000000
L1R21: FALSE; Support=0.000000
L1R22: FALSE; Support=0.000000
L1R23: FALSE; Support=0.000000
L1R24: Diameter > 0.425 OR Height <= 0.125; Support=0.835428
L1R25: FALSE; Support=0.000000
L1R26: FALSE; Support=0.000000
L1R27: Shell weight > 0.3705; Support=0.166367
L1R28: Height <= 0.1 OR Shell weight <= 0.165; Support=0.341113
L1R29: FALSE; Support=0.000000
L1R30: Shell weight <= 0.295; Support=0.675344
L1R31: FALSE; Support=0.000000

Rings = 10.3164949 + sum(weight_j * rule_j)

L1R0: 0.442958504 Rings
L1R1: -0.766893923 Rings
L1R2: 0.291324973 Rings
L1R3: 0.425138801 Rings
L1R4: 0.218134016 Rings
L1R5: -1.20278811 Rings
L1R6: 0.512391627 Rings
L1R7: 1.12265372 Rings
L1R8: 0.382851273 Rings
L1R9: 0.21729888 Rings
L1R10: 0.0702352747 Rings
L1R11: 0.745596409 Rings
L1R12: 1.07036817 Rings
L1R13: 0.313009858 Rings
L1R14: 0.490499645 Rings
L1R15: -0.293719679 Rings
L1R16: 2.87346059e-40 Rings
L1R17: -1.58430743 Rings
L1R18: -4.97224135e-40 Rings
L1R19: 7.85207926e-39 Rings
L1R20: -3.41098467e-40 Rings
L1R21: -2.12123096e-39 Rings
L1R22: -1.03630716e-38 Rings
L1R23: 4.46953954e-40 Rings
L1R24: -0.605723262 Rings
L1R25: 0.0400754884 Rings
L1R26: -0.0679400042 Rings
L1R27: 1.14183831 Rings
L1R28: -1.68248761 Rings
L1R29: 5.58675277e-40 Rings
L1R30: -0.811456084 Rings
L1R31: -3.08065658e-40 Rings

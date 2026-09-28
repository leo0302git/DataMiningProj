# 回归 RRL 规则图

Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。

L1R0: Shell weight > 0.23 AND Shucked weight <= 0.57400001; Support=0.332834
L1R1: TRUE; Support=1.000000
L1R2: Shell weight > 0.091833335 AND Shucked weight <= 0.3345; Support=0.334331
L1R3: TRUE; Support=1.000000
L1R4: Shucked weight > 0.2355 AND Shell weight <= 0.29; Support=0.336426
L1R5: TRUE; Support=1.000000
L1R6: Shucked weight <= 0.57400001; Support=0.833283
L1R7: TRUE; Support=1.000000
L1R8: Height > 0.175; Support=0.150853
L1R9: Length > 0.40000001; Support=0.832984
L1R10: Diameter > 0.305 AND Shell weight > 0.23 AND Shucked weight <= 0.57400001; Support=0.332535
L1R11: Height <= 0.125 AND Shell weight <= 0.16216667 AND Shell weight <= 0.367; Support=0.304699
L1R12: Shell weight <= 0.367; Support=0.833583
L1R13: Diameter > 0.375 AND Shell weight > 0.29 AND Shucked weight <= 0.44233333; Support=0.063155
L1R14: Shell weight > 0.367; Support=0.166417
L1R15: TRUE; Support=1.000000
L1R16: FALSE; Support=0.000000
L1R17: FALSE; Support=0.000000
L1R18: Height <= 0.1; Support=0.188866
L1R19: FALSE; Support=0.000000
L1R20: Shucked weight > 0.44233333 OR Diameter <= 0.375 OR Shell weight <= 0.16216667; Support=0.697695
L1R21: Height > 0.1; Support=0.811134
L1R22: Sex=I OR Shucked weight > 0.3345 OR Whole weight <= 0.55549999; Support=0.882670
L1R23: FALSE; Support=0.000000
L1R24: FALSE; Support=0.000000
L1R25: FALSE; Support=0.000000
L1R26: FALSE; Support=0.000000
L1R27: FALSE; Support=0.000000
L1R28: Height > 0.155; Support=0.332834
L1R29: FALSE; Support=0.000000
L1R30: FALSE; Support=0.000000
L1R31: FALSE; Support=0.000000

Rings = 9.89603329 + sum(weight_j * rule_j)

L1R0: 0.251400352 Rings
L1R1: -0.123592384 Rings
L1R2: 0.114983268 Rings
L1R3: 0.124285333 Rings
L1R4: -1.02361774 Rings
L1R5: -0.00683253864 Rings
L1R6: 0.256522596 Rings
L1R7: -0.0611678623 Rings
L1R8: 0.582038224 Rings
L1R9: 0.142254025 Rings
L1R10: 0.870996416 Rings
L1R11: -1.09929168 Rings
L1R12: -0.2972458 Rings
L1R13: 1.13995159 Rings
L1R14: 1.76286387 Rings
L1R15: 0.202958882 Rings
L1R16: 4.75176105e-40 Rings
L1R17: -7.394652e-41 Rings
L1R18: -0.541910887 Rings
L1R19: 2.40486639e-40 Rings
L1R20: -0.913040042 Rings
L1R21: 1.13970089 Rings
L1R22: -1.36823463 Rings
L1R23: -1.40129846e-40 Rings
L1R24: 2.71598267e-40 Rings
L1R25: -3.59172415e-40 Rings
L1R26: -2.95680114e-38 Rings
L1R27: -5.39561566e-40 Rings
L1R28: 0.554974377 Rings
L1R29: -1.62054983e-39 Rings
L1R30: -1.73284568e-40 Rings
L1R31: -3.34408668e-40 Rings

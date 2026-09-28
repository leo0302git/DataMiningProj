# 回归 RRL 规则图

Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。

L1R0: Shell weight > 0.23125 AND Shell weight > 0.3705 AND Shucked weight <= 0.57550001; Support=0.054758
L1R1: TRUE; Support=1.000000
L1R2: TRUE; Support=1.000000
L1R3: Diameter > 0.31 AND Diameter <= 0.425 AND Shucked weight <= 0.44166667; Support=0.333333
L1R4: TRUE; Support=1.000000
L1R5: Shucked weight > 0.3345 AND Height <= 0.175 AND Whole weight <= 1.3; Support=0.294434
L1R6: TRUE; Support=1.000000
L1R7: TRUE; Support=1.000000
L1R8: Shell weight > 0.295; Support=0.324656
L1R9: TRUE; Support=1.000000
L1R10: Height <= 0.1 AND Shell weight <= 0.093999993; Support=0.153202
L1R11: Shell weight <= 0.295; Support=0.675344
L1R12: TRUE; Support=1.000000
L1R13: Sex=I AND Height <= 0.16; Support=0.304907
L1R14: TRUE; Support=1.000000
L1R15: Shell weight <= 0.3705; Support=0.833633
L1R16: FALSE; Support=0.000000
L1R17: FALSE; Support=0.000000
L1R18: FALSE; Support=0.000000
L1R19: Shucked weight > 0.57550001 OR Shell weight <= 0.165; Support=0.502992
L1R20: Shell weight <= 0.165; Support=0.336924
L1R21: FALSE; Support=0.000000
L1R22: Shell weight > 0.3705; Support=0.166367
L1R23: FALSE; Support=0.000000
L1R24: FALSE; Support=0.000000
L1R25: Diameter > 0.465 OR Height > 0.16 OR Shell weight > 0.295 OR Shucked weight <= 0.23633333; Support=0.731598
L1R26: FALSE; Support=0.000000
L1R27: FALSE; Support=0.000000
L1R28: FALSE; Support=0.000000
L1R29: Diameter > 0.31; Support=0.825254
L1R30: FALSE; Support=0.000000
L1R31: Shucked weight > 0.44166667 OR Diameter <= 0.375 OR Shell weight <= 0.23125; Support=0.824955

Rings = 10.8397608 + sum(weight_j * rule_j)

L1R0: 0.667468309 Rings
L1R1: 0.216314122 Rings
L1R2: 0.292449206 Rings
L1R3: 0.705269635 Rings
L1R4: 0.407508343 Rings
L1R5: -0.870545328 Rings
L1R6: 0.16217503 Rings
L1R7: 0.430445164 Rings
L1R8: 0.391059577 Rings
L1R9: -0.0213762708 Rings
L1R10: -1.62903059 Rings
L1R11: -0.279676795 Rings
L1R12: 0.444971591 Rings
L1R13: -1.22405612 Rings
L1R14: 0.336314738 Rings
L1R15: -1.15759063 Rings
L1R16: -3.79751884e-43 Rings
L1R17: 0.0686696991 Rings
L1R18: 0.0789177194 Rings
L1R19: -1.04930902 Rings
L1R20: -0.782476366 Rings
L1R21: -3.91082783e-40 Rings
L1R22: 0.184514776 Rings
L1R23: 2.88637356e-39 Rings
L1R24: 0.071692504 Rings
L1R25: 0.971181035 Rings
L1R26: 5.52098983e-40 Rings
L1R27: -3.28416015e-39 Rings
L1R28: -4.54993204e-40 Rings
L1R29: 0.163421869 Rings
L1R30: -0.0558624864 Rings
L1R31: -1.6985364 Rings

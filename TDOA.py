import numpy as np
from scipy.optimize import least_squares
import math

# 铝的材料参数
E = 6.9e10 # 杨氏模量（单位：Pa）
miu = 0.35
G = E/2/(1+miu) # 剪切模量（单位：Pa）
rho = 2700.0 # 密度（单位：kg/m³）

# 计算纵波和横波的传播速度
v_longitudinal = math.sqrt(E / rho/(1-miu))
v_shear = math.sqrt(G / rho)
v_lamb = math.sqrt(E / (2 * rho * (1 + miu)))

# 打印结果
print("纵波传播速度：", v_longitudinal, "m/s")
print("横波传播速度：", v_shear, "m/s")

def trilateration(t1, t2, t3, pos1, pos2, pos3):
    v = v_shear  # 介质为铝，声速为5092 m/s
    v = v*100
    d1 = v * t1
    d2 = v * t2
    d3 = v * t3

    def equations(p):
        x, y = p[0], p[1]
        eq1 = math.sqrt((x - pos1[0]) ** 2 + (y - pos1[1]) ** 2) - math.sqrt((x - pos2[0]) ** 2 + (y - pos2[1]) ** 2) - d1
        eq2 = math.sqrt((x - pos2[0]) ** 2 + (y - pos2[1]) ** 2) - math.sqrt((x - pos3[0]) ** 2 + (y - pos3[1]) ** 2) - d2
        eq3 = math.sqrt((x - pos3[0]) ** 2 + (y - pos3[1]) ** 2) - math.sqrt((x - pos1[0]) ** 2 + (y - pos1[1]) ** 2) - d3
        return [eq1, eq2, eq3]

    solution = least_squares(equations, [0, 0])
    x, y = solution.x[0], solution.x[1]

    return x, y
f1 = open(r".\estimated.txt", 'w') #保存文件

fs=3*10**6
f = open(r".\x5.txt", 'r', encoding='utf-8') #载入时差数据
lines = f.readlines()
delta_t1 = []
delta_t2 = []
delta_t3 = []

for i in range(0, len(lines)):
    a = lines[i]
    b = a.split(sep=None, maxsplit=-1)
    delta_t1.append(np.float32(b[0]))
    delta_t2.append(np.float32(b[1]))
    delta_t3.append(np.float32(b[2]))

t1 = [x / (3*10**6) for x in delta_t1]  # 探头1与探头2的时间差（秒）
t2 = [x / (3*10**6) for x in delta_t2]  # 探头2与探头3的时间差（秒）
t3 = [x / (3*10**6) for x in delta_t3]  # 探头3与探头1的时间差（秒）

pos1 = [-2, -2]  # 探头1的位置坐标（x，y），单位厘米
pos2 = [22, 22]  # 探头2的位置坐标（x，y），单位厘米
pos3 = [22, -2]  # 探头3的位置坐标（x，y），单位厘米
m=0
m11=0
m22=0
m2=0
s=0
for i in range(0, len(lines)):
    # 调用函数进行定位
    estimated_x, estimated_y = trilateration(t1[i], t2[i], t3[i], pos1, pos2, pos3)
    j=i+1

    k = j // 15  ##################
    qq = j % 15
    if qq == 0:
        qq = 15
        k = k - 1
    pp=k//5
    qq=k%5
    if pp==0:
        yy=0
    if pp == 1:
        yy = 5
    if pp==2:
        yy=10
    if pp==3:
        yy=15
    if pp==4:
        yy=20
    if qq==0:
        xx=0
    if qq == 1:
        xx = 5
    if qq==2:
        xx=10
    if qq==3:
        xx=15
    if qq==4:
        xx=20
    if abs(estimated_x)<1000 and abs(estimated_y)<1000:
        m = m + abs(xx - estimated_x) ** 2 + abs(yy - estimated_y) ** 2
        m11 = m11 + abs(xx*10 - estimated_x*10) ** 2
        m22 = m22 + abs(yy*10 - estimated_y*10) ** 2
        m2 = m2 + np.sqrt(abs(xx - estimated_x) ** 2 + abs(yy - estimated_y) ** 2)/5
        s=s+1

    # 打印预测的目标位置坐标
    print("预测的目标位置坐标：", estimated_x, estimated_y)
    f1.write("{:.1f} {:.1f} {:.1f} {:.1f}\n".format(estimated_x*10, estimated_y*10, xx*10, yy*10))

#误差计算
print(m/s*10)
m1=np.sqrt(m11/s)+np.sqrt(m22/s)
print(m1)
print(m2/s)


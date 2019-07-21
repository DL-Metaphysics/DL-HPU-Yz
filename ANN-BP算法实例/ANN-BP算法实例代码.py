import numpy as np
import matplotlib.pyplot as plt
import math
a=np.array([0.05,0.1])           #a1,a2的输入值
weight1=np.array([[0.15,0.25],[0.2,0.3]])   #a1对b1,b2的权重，a2对b1，b2的权重
weight2=np.array([[0.4,0.5],[0.45,0.55]])     #b1对c1,c2的权重，b2对c1，c2的权重
target=np.array([0.01,0.99])
d1=0.35   #输入层的偏置（1）的权重
d2=0.6    #隐藏层的偏置（1）的权重
β=0.5    #学习效率

#一：前向传播

#计算输入层到隐藏层的输入值，得矩阵netb1,netb2
netb=np.dot(a,weight1)+d1

#计算隐藏层的输出值,得到矩阵outb1,outb2
m=[]
for i in range(len(netb)):
    outb=1.0 / (1.0 + math.exp(-netb[i]))
    m.append(outb)
m=np.array(m)

#计算隐藏层到输出层的输入值，得矩阵netc1,netc2
netc=np.dot(m,weight2)+d2

#计算隐藏层的输出值,得到矩阵outc1,outc2
n=[]
for i in range(len(netc)):
    outc=1.0 / (1.0 + math.exp(-netc[i]))
    n.append(outc)
n=np.array(n)

#二：反向传播
count=0 #计数
e=0     #误差  
E=[]    #统计误差
#梯度下降
while True:
  count+=1
  
  #总误差对w1-w4的偏导
  pd1=(-(target[0]-n[0])*n[0]*(1-n[0])*weight2[0][0]-(target[1]-n[1])*n[1]*(1-n[1])*weight2[0][1])*m[0]*(1-m[0])*a[0]
  pd2=(-(target[0]-n[0])*n[0]*(1-n[0])*weight2[0][0]-(target[1]-n[1])*n[1]*(1-n[1])*weight2[0][1])*m[0]*(1-m[0])*a[1]
  pd3=(-(target[0]-n[0])*n[0]*(1-n[0])*weight2[1][0]-(target[1]-n[1])*n[1]*(1-n[1])*weight2[0][1])*m[0]*(1-m[0])*a[0]                          
  pd4=(-(target[0]-n[0])*n[0]*(1-n[0])*weight2[1][1]-(target[1]-n[1])*n[1]*(1-n[1])*weight2[0][1])*m[0]*(1-m[0])*a[1]
  weight1[0][0]=weight1[0][0]-β*pd1
  weight1[1][0]=weight1[1][0]-β*pd2
  weight1[0][1]=weight1[0][1]-β*pd3
  weight1[1][1]=weight1[1][1]-β*pd4

  #总误差对w5-w8的偏导
  pd5=-(target[0]-n[0])*n[0]*(1-n[0])*m[0]
  pd6=-(target[0]-n[0])*n[0]*(1-n[0])*m[1]
  pd7=-(target[1]-n[1])*n[1]*(1-n[1])*m[0]
  pd8=-(target[1]-n[1])*n[1]*(1-n[1])*m[1]
  weight2[0][0]=weight2[0][0]-β*pd5
  weight2[1][0]=weight2[1][0]-β*pd6
  weight2[0][1]=weight2[0][1]-β*pd7
  weight2[1][1]=weight2[1][1]-β*pd8
  
  netb=np.dot(a,weight1)+d1
  m=[]
  for i in range(len(netb)):
    outb=1.0 / (1.0 + math.exp(-netb[i]))
    m.append(outb)
  m=np.array(m)
  netc=np.dot(m,weight2)+d2
  n=[]
  for i in range(len(netc)):
    outc=1.0 / (1.0 + math.exp(-netc[i]))
    n.append(outc)
  n=np.array(n)
  
  #计算总误差
  for j in range(len(n)):
    e += (target[j]-n[j])**2/2
  E.append(e)
  #判断
  if e<0.0000001:
    break
  else:
      e=0
print(count)
print(e)
print(n)
plt.plot(range(len(E)),E,label='error')
plt.legend() 
plt.xlabel('time')
plt.ylabel('error')
plt.show()

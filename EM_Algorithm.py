#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2022/4/20
# @file EM_Algorithm.py
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


isdebug = True
class EM_Algorithm():

    def __init__(self):
        def trainingData():
            # 生成二维高斯模型的数据
            s1 = 0.2
            s2 = 0.5
            p = 0.3
            q = 0.6
            r = 0.9
            num=self.num
            A = np.zeros(num)
            count=0
            for i in range(1, num + 1):
                if np.random.random(1) < s1:
                    if(np.random.random(1)<p):
                        A[i - 1] = 1
                        count+=1
                    else:
                        A[i - 1] = -1
                elif np.random.random(1)<s1+s2:
                    if (np.random.random(1) < q):
                        A[i - 1] = 1
                        count += 1
                    else:
                        A[i - 1] = -1
                else:
                    if (np.random.random(1) < r):
                        A[i - 1] = 1
                        count += 1
                    else:
                        A[i - 1] = -1

            if isdebug:
                np.save('./resultData'+str(s1)+str(s2)+str(p)+str(q)+str(r)+'.'+str(num)+'.npy',A)
                print(A)
                print(np.size(A))
                print(count)
                plt.hist(A)
                plt.show()
            return count

        self.s1 = 0.3
        self.s2 = 0.4
        self.p = 0.4
        self.q = 0.5
        self.r = 0.8
        self.num=96000
        self.iter_num=1000
        self.Epsilon=0.00000000001
        self.A = np.load('./resultData0.20.50.30.60.9.96000.npy')
        self.count = self.A.tolist().count(1)

        #用于生成数据
        # self.count=trainingData()


    def Update(self):
        def eUpdate():
            us1=list()
            us2=list()
            s1temp=0
            s2temp=0
            for i in range(self.num):
                if self.A[i]==1:
                    temp=(self.s1*self.p+self.s2*self.q+(1-self.s1-self.s2)*self.r)
                    us1.append(self.s1*self.p/temp)
                    s1temp+=self.s1*self.p/temp
                    us2.append(self.s2*self.q/temp)
                    s2temp += self.s2*self.q/temp
                else:
                    temp=(self.s1*(1-self.p)+self.s2*(1-self.q)+(1-self.s1-self.s2)*(1-self.r))
                    us1.append(self.s1 * (1 - self.p) / temp)
                    us2.append(self.s2 * (1 - self.q) / temp)
            return us1,us2,s1temp,s2temp
        def mUpdate(us1,us2,s1temp,s2temp):
            temp1=sum(us1)
            temp2=sum(us2)
            self.s1=temp1/self.num
            self.s2=temp2/self.num
            self.p=s1temp/temp1
            self.q=s2temp/temp2
            self.r=(self.count-s1temp-s2temp)/(self.num-temp1-temp2)

        iteration=0
        for i in range(self.iter_num):
            iteration=iteration+1
            Old_p = copy.deepcopy(self.p)
            Old_q = copy.deepcopy(self.q)
            us1,us2,s1temp,s2temp=eUpdate()
            mUpdate(us1,us2,s1temp,s2temp)
            if (abs(Old_p - self.p)+abs(Old_q - self.q)) < self.Epsilon:
                break
        print('迭代次数:')
        print(iteration)
        print('选中A硬币概率：',self.s1)
        print('选中B硬币概率：',self.s2)
        print('选中C硬币概率：',1-self.s1-self.s2)

        print('A硬币投掷为正的概率：',self.p)
        print('B硬币投掷为正的概率：',self.q)
        print('C硬币投掷为正的概率：',self.r)


if __name__ == '__main__':
    # run(6,40,20,2,1000,1000,0.0001)
    # plt.hist(X[0,:],50)
    # plt.show()
    em = EM_Algorithm()
    s1=0.3
    s2=0.3
    p=0.5
    q = 0.5
    r = 0.5
    em.Update()



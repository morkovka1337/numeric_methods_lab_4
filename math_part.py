import math
import pylab
import numpy as np
from numpy import float64
from matplotlib import mlab
from matplotlib.figure import Figure
from Form_for_4_lab import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt




class mathpart(Ui_MainWindow):

 

    def build_test_1(self, n):

        def phi(self, i):
            nonlocal h
            return (h**2)*(12*(i**2)+1)/3 + 10*i*h + 6


        def acc_sol(self, x):
            return 2*x**2 + 5*x + 7

        h = 1/n
        v = np.zeros(n+1)
        u = np.zeros(n+1)
        for i in range(0, n+1):
            u[i] = acc_sol(self, i * h)
        
        
        b = np.zeros(n+1)
        syst = np.zeros((n+1, n+1))
        b[0] = 7
        for i in range(1, n):
            #b[i] = -phi(self, i)
            b[i] = -4*((i*h)**2) - 10*i*h - 6 
        b[n] = 14

        for i in range(1, n):
            syst[i][i-1] = 2/(h**2)
            syst[i][i] = -(4/(h**2) + 2)
            syst[i][i+1] = 2/(h**2)
        
        syst[0][0] = 1
        syst[n][n] = 1

        v = np.linalg.solve(syst, b)

        
        self.tableWidget.setRowCount(n+1)
        for i in range(0, n+1):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i*h)))
            self.tableWidget.setItem(i, 2, QtWidgets.QTableWidgetItem(str(u[i])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(v[i])))
            self.tableWidget.setItem(i, 5, QtWidgets.QTableWidgetItem(str(u[i] - v[i])))
        self.label_2.setText(QtCore.QCoreApplication.translate("MainWindow", 
                    "Результаты расчета: \n Для решения тестовой задачи \n использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение точного и приближенного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(u-v))) + ", значение которой равно:"
                    + str(max(abs(u-v)))))

        plt.subplot(111)
        plt.plot(u)
        plt.ylabel("Температура")
        plt.plot(v)
        plt.plot(u-v)
        plt.legend(("Точное решение", "Численное решение", "Разность точного и численного"))
        plt.show()

    def build_test_2(self, n):
        ksi = np.pi/4
        def TDMASolve(a, c, b, d):
            nonlocal n
            alpha = np.zeros(n)
            beta = np.zeros(n)
            x = np.zeros(n+1)
            x[0] = d[0] #x[0] = kapa1*x[1] + mu1, mu1 = d[0]
            alpha[0] = 0 # alpha[0] = kapa1, kapa1 в этой задаче всегда 0
            beta[0] = d[0] # beta[0] = mu[1]
            for i in range(0, n-1):
                alpha[i+1] = b[i]/(c[i]-alpha[i]*a[i])
                # phi[i] = d[i+1]
                beta[i+1] = (d[i+1] * beta[i] * a[i])/(c[i] - alpha[i] * a[i])
                # alpha.append(b[i-1]/(c[i-1]-alpha[i-1]*a[i-1]))
                # beta.append((d[i] + beta[i-1]*a[i-1])/c[i-1]-alpha[i-1]*a[i-1])
            
            plt.subplot(211)
            plt.plot(beta)
            x[-1] = d[-1]
            for i in range (n-1,  0, -1):
                x[i] = alpha[i] * x[i+1] + beta[i]
            return x

        def a_func(x):
            nonlocal h
            if x <= ksi:
                return 1
            elif x-h >= ksi:
                return 2
            elif x-h<ksi and x > ksi:
                return h/((ksi-x)/2 + h)
        def d_func(x):
            nonlocal h
            if x+h/2 <= ksi:
                return ksi + 1
            elif x-h/2 < ksi and x+h/2 > ksi:
                return (ksi+1)*(ksi-x+h/2)/h + np.pi**2*(x+h/2-ksi)/(2*h)
            elif x-h/2 >= ksi:
                return np.pi**2/2
            
        def f(x):
            nonlocal h
            if x+h/2 <= ksi:
                return 1
            elif x-h/2 < ksi and x+h/2 > ksi:
                return (ksi-x+h/2)/h + np.sqrt(2)*(x+h/2-ksi)/(2*h)
            elif x-h/2 >= ksi:
                return np.sqrt(2)/2

            
        # def intu(x):
        #     return 2*x**3/3 + 5*x**2/2 + 7*x

        # def du(x):
        #     return 4*x + 5

        h = 1/n
        v = np.zeros(n+1)
        u = np.zeros(n+1)
        d = np.zeros(n+1)
        d[0] = 1
        for i in range(1, n):
            d[i] = -f(i*h)
            #d[i] = -(q(i*h)/h)*(intu(i*h+0.5*h) - intu(i*h-0.5*h)) + (k(i*h)/h) *(du(i*h+0.5*h) - du(i*h-0.5*h))
        d[n] = 0

        # syst = np.zeros((n+1, n+1))
        # syst[0][0] = 1
        C = np.zeros(n-1)
        A = np.zeros(n-1)
        B = np.zeros(n-1)
        for i in range(1, n): #последний индекс - n-1
            xi = (i)*h
            xi1 = (i+1)*h
            A[i-1] = a_func(xi)/h**2
            C[i-1] = -(a_func(xi)/(h**2)+d_func(xi)+a_func(xi1)/(h**2))
            B[i-1] = a_func(xi1)/h**2
        # syst[n][n] = 0

        # v = np.linalg.solve(syst, d)
        
        # for i in range(0, n):
        #     a[i] = k(i*h)/(h**2)
        #     c[i+1] = -(k((i+1)*h) + k(i*h))/(h**2) + q(i*h)
        #     b[i] = k((i+1)*h)/(h**2)
        # c[0] = 1
        # c[n] = 1

        v = TDMASolve(A, C, B, d)
        
        plt.subplot(212)
        #plt.plot(u)
        plt.ylabel("Температура")
        plt.plot(v)
        #plt.plot(u-v)
        plt.legend("Численное решение")
        plt.show()

        

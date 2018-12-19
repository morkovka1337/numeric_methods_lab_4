import math
import numpy as np
from numpy import float64
from matplotlib.figure import Figure
from Form_for_4_lab import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt




class mathpart(Ui_MainWindow):

    def build_test_1(self, n):
        def TDMASolve(a, c, b, d):
            n = len(d)-1
            alpha = np.zeros(n)
            beta = np.zeros(n)
            x = np.zeros(n+1)
            alpha[0] = 0 # alpha[0] = kapa1, kapa1 в этой задаче всегда 0
            beta[0] = d[0] # beta[0] = mu[1]
            for i in range(0, n-1):
                alpha[i+1] = b[i]/(c[i]-alpha[i]*a[i])
                beta[i+1] = (d[i+1] + beta[i] * a[i])/(c[i] - alpha[i] * a[i])

            x[-1] = d[-1]
            for i in range (n-1, 0, -1):
                x[i] = alpha[i] * x[i+1] + beta[i]
            x[0] = alpha[0] * x[1] + beta[0]
            return x
            
        def acc_sol(x):
            return x**2

        h = 1/n
        v = np.zeros(n+1)
        u = [acc_sol(i * h) for i in range(0, n+1)]
        
        
        d = np.zeros(n+1)
        d[0] = 0
        for i in range(1, n):
            d[i] = 2*(i*h)**2-2
        d[n] = 1
        
        C = np.zeros(n-1)
        A = np.zeros(n-1)
        B = np.zeros(n-1)
        for i in range(1, n): #последний индекс - n-1
            A[i-1] = 1/(h**2)
            C[i-1] = (2/(h**2) + 2)
            B[i-1] = 1/(h**2)

        v = TDMASolve(A, C, B, d)

        
        self.tableWidget.setRowCount(n+1)
        for i in range(0, n+1):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i*h)))
            self.tableWidget.setItem(i, 2, QtWidgets.QTableWidgetItem(str(u[i])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(v[i])))
            self.tableWidget.setItem(i, 5, QtWidgets.QTableWidgetItem(str(u[i] - v[i])))

        plt.subplot(111)
        plt.plot(u)
        plt.ylabel("Температура")
        plt.plot(v)
        plt.legend(("Точное решение", "Численное решение прогонкой"))
        plt.show()
        plt.subplot(111)
        plt.plot(abs(u-v))
        plt.legend(("Разность точного и численного"))
        plt.show()
        QMessageBox.about(self, "Справка", 
            "Результаты расчета: \n Для решения тестовой задачи использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение точного и приближенного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(u-v))*h) + ", значение которой равно:"
                    + str(max(abs(u-v))))



    def build_test_2(self, n):
        ksi = 0.125

        def TDMASolve(a, c, b, d):
            n = len(d)-1
            alpha = np.zeros(n, float64)
            beta = np.zeros(n, float64)
            x = np.zeros(n+1, float64)
            alpha[0] = 0 # alpha[0] = kapa1, kapa1 в этой задаче всегда 0
            beta[0] = d[0] # beta[0] = mu[1]
            for i in range(0, n-1):
                alpha[i+1] = b[i]/(c[i]-alpha[i]*a[i])
                beta[i+1] = (d[i+1] + beta[i] * a[i])/(c[i] - alpha[i] * a[i])

            x[-1] = d[-1]
            for i in range (n-1, 0, -1):
                x[i] = alpha[i] * x[i+1] + beta[i]
            x[0] = alpha[0] * x[1] + beta[0]
            return x

        def a_func(x):
            nonlocal h
            if x <= ksi:
                return ksi+1
            elif x-h >= ksi:
                return 1
            elif x-h<ksi and x > ksi:
                return h/((ksi-x+h)/(ksi+1) + (x-ksi))
        def d_func(x):
            nonlocal h
            if x+h/2 <= ksi:
                return np.exp(-ksi)
            elif x-h/2 < ksi and x+h/2 > ksi:
                return ((ksi-x+h/2)*np.exp(-ksi) + (x+h/2-ksi)*np.exp(-(ksi**2)))/h
            elif x-h/2 >= ksi:
                return np.exp(-(ksi**2))
            
        def f(x):
            nonlocal h
            if x+h/2 <= ksi:
                return np.cos(ksi)
            elif x-h/2 < ksi and x+h/2 > ksi:
                return ((ksi-x+h/2)*(np.cos(ksi)) + (x+h/2-ksi))/h
            elif x-h/2 >= ksi:
                return 1

        def acc_sol(x):
            
            c1=0.10915303818319690987; c2=-1.2334602899406248433; c3= 0.15793898573540121610
            c4=-1.1914718806035373699
            if x < ksi:
                return (c1*np.exp(np.sqrt(np.exp(-ksi)/(ksi+1))*x) + 
                    c2*np.exp(-np.sqrt(np.exp(-ksi)/(ksi+1))*x) + 
                        np.cos(ksi)/np.exp(-ksi))
            else:
                return (c3*np.exp(np.sqrt(np.exp(-(ksi**2)))*x) + 
                    c4*np.exp(-np.sqrt(np.exp(-(ksi**2)))*x) +
                        1/(np.exp(-(ksi**2))))
            # if x < ksi:
            #     return float64(c1*np.exp(np.sqrt(np.pi/4+1)*x) + c2*np.exp(-np.sqrt(np.pi/4+1)*x) + 1/(np.pi/4 + 1))
            # else:
            #     return float64(c3*np.exp(np.pi*x/4) + c4 * np.exp(-np.pi*x/4) + 4*np.sqrt(2)/(np.pi*np.pi))
    
        h = float64(1/n)
        v = np.zeros(n+1, float64)
        u = np.zeros(n+1, float64)
        d = np.zeros(n+1, float64)
        d[0] = 0
        for i in range(1, n):
            d[i] = f(i*h)
        d[n] = 1

        C = np.zeros(n-1, float64)
        A = np.zeros(n-1, float64)
        B = np.zeros(n-1, float64)
        for i in range(1, n): #последний индекс - n-1
            xi =float64((i)*h)
            xi1 = float64((i+1)*h)
            A[i-1] = a_func(xi)/h**2
            C[i-1] = (a_func(xi)/(h**2)+d_func(xi)+a_func(xi1)/(h**2))
            B[i-1] = a_func(xi1)/h**2

        
        
        v = TDMASolve(A, C, B, d)

        u = [float64(acc_sol(float64(i*h))) for i in range(0, n+1)]
        y = u-v

        self.tableWidget.setRowCount(n+1)
        for i in range(0, n+1):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i*h)))
            self.tableWidget.setItem(i, 2, QtWidgets.QTableWidgetItem(str(u[i])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(v[i])))
            self.tableWidget.setItem(i, 5, QtWidgets.QTableWidgetItem(str(u[i] - v[i])))
        plt.subplot(111)
        plt.plot(u)
        plt.ylabel("Температура")
        plt.plot(v)
        plt.legend(("Точное решение", "Численное решение прогонкой"))
        plt.show()
        plt.subplot(111)
        plt.plot(abs(y))
        plt.legend(("Разность точного и численного"))
        plt.show()
        QMessageBox.about(self, "Справка", 
            "Результаты расчета: \n Для решения тестовой задачи использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение точного и приближенного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(u-v))*h) + ", значение которой равно:"
                    + str(max(abs(u-v))))




    def build_test_3(self, n):
        ksi = 0.125
        def TDMASolve(a, c, b, d):
            n = len(d)-1
            alpha = np.zeros(n)
            beta = np.zeros(n)
            x = np.zeros(n+1, float64)
            alpha[0] = 0 # alpha[0] = kapa1, kapa1 в этой задаче всегда 0
            beta[0] = d[0] # beta[0] = mu[1]
            for i in range(0, n-1):
                alpha[i+1] = b[i]/(c[i]-alpha[i]*a[i])
                beta[i+1] = (d[i+1] + beta[i] * a[i])/(c[i] - alpha[i] * a[i])

            x[-1] = d[-1]
            for i in range (n-1, 0, -1):
                x[i] = alpha[i] * x[i+1] + beta[i]
            x[0] = alpha[0] * x[1] + beta[0]
            return x

        def a_func(x):
            nonlocal h
            if x <= ksi:
                return x-h/2 + 1
           
            elif x-h < ksi and x > ksi:
                # left_mean = (ksi+x-h)/2
                # left_int = (ksi-(x-h))/(left_mean+1)
                # # right_mean = (x+ksi)/2
                # right_int = (x-ksi)
                # full_int = left_int + right_int
                return h / ((ksi - (x - h)) / ((x - h + ksi)/2 + 1) +
                     (x - ksi)) 
            
            elif x-h >= ksi:
                return 1

        def d_func(x):
            nonlocal h
            if x+h/2 <= ksi:
                return np.exp(-x)

            elif x-h/2 < ksi and x+h/2 > ksi:
                # left_mean = (ksi+(x-h/2))/2
                # left_int = (ksi-(x-h/2))*(left_mean+1)
                # right_mean = (x+h/2+ksi)/2
                # right_int = (x+h/2-ksi) * 2*right_mean**2
                # return (left_int + right_int)/h
                return ((ksi - (x - h / 2)) * np.exp(-(x + h / 2 - ksi)/2) +
                     (x + h / 2 - ksi) * np.exp(-(x + h / 2 - ksi)/2)) / h   
            
            elif (x-h/2) >= ksi:
                return np.exp(-x**2)       
        def f(x):
            nonlocal h
            if x+h/2 <= ksi:
                return np.cos(x)
            elif x-h/2 < ksi and x+h/2 > ksi:
                # left_mean = (ksi+(x-h/2))/2
                # left_int = (ksi-(x-h/2)) * np.sin(2*left_mean)
                # right_mean = (x+h/2+ksi)/2
                # right_int = (x+h/2-ksi)*np.sin(right_mean)
                # return (left_int + right_int)/h
                return ((ksi - (x - h / 2)) * (np.cos((x + h / 2 - ksi)/2)) + 
                        (x + h / 2 - ksi) ) / h

            elif x-h/2 >= ksi:
                return 1
            

        h = float64(1/n)
        v = np.zeros(n+1, float64)
        d = np.zeros(n+1, float64)
        d[0] = 0
        for i in range(1, n):
            d[i] = f(i*h)
        d[n] = 1

        C = np.zeros(n-1, float64)
        A = np.zeros(n-1, float64)
        B = np.zeros(n-1, float64)
        for i in range(1, n): #последний индекс - n-1
            xi =float64((i)*h)
            xi1 = float64((i+1)*h)
            A[i-1] = a_func(xi)/(h**2)
            C[i-1] = (a_func(xi)+a_func(xi1))/(h**2)+d_func(xi)
            B[i-1] = a_func(xi1)/(h**2)

        v = TDMASolve(A, C, B, d)

        n_new = n* 2
        h = float64(1/n_new)
        d = np.zeros(n_new+1, float64)
        d[0] = 0
        for i in range(1, n_new):
            d[i] = f(i*h)
        d[n_new] = 1

        C = np.zeros(n_new-1, float64)
        A = np.zeros(n_new-1, float64)
        B = np.zeros(n_new-1, float64)
        for i in range(1, n_new): #последний индекс - n-1
            xi =float64((i-1)*h)
            xi1 = float64((i)*h)
            A[i-1] = a_func(xi)/h**2
            C[i-1] = (a_func(xi)/(h**2)+d_func(xi)+a_func(xi1)/(h**2))
            B[i-1] = a_func(xi1)/h**2
        
        v2 = TDMASolve(A, C, B, d)
        
        v2 = v2[::2]
        y = v2-v
        h*=2
        self.tableWidget.setRowCount(n+1)
        for i in range(0, n+1):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i*h)))
            self.tableWidget.setItem(i, 4, QtWidgets.QTableWidgetItem(str(v2[i])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(v[i])))
            self.tableWidget.setItem(i, 6, QtWidgets.QTableWidgetItem(str(v2[i] - v[i])))
            
        plt.subplot(111)
        plt.plot(v2)
        plt.ylabel("Температура")
        plt.plot(v)
        plt.legend(("Численное с половинным шагом", "Численное решение"))
        plt.show()
        plt.subplot(111)
        plt.plot(abs(y))
        plt.legend(("Разность точного и численного"))
        plt.show()
        QMessageBox.about(self, "Справка", 
            "Результаты расчета: \n Для решения тестовой задачи использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение численного с двойным шагом и численного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(y))*h) + ", значение которой равно:"
                    + str(max(abs(y))))
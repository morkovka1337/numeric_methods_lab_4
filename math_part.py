import math
import numpy as np
from numpy import float64
from matplotlib.figure import Figure
from Form_for_4_lab import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
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
            return 2*x**2 + 5*x + 7

        h = 1/n
        v = np.zeros(n+1)
        u = [acc_sol(i * h) for i in range(0, n+1)]
        
        
        b = np.zeros(n+1)
        #syst = np.zeros((n+1, n+1))
        C = np.zeros(n-1)
        A = np.zeros(n-1)
        B = np.zeros(n-1)
        for i in range(1, n): #последний индекс - n-1
            A[i-1] = 2/(h**2)
            C[i-1] = (4/(h**2) + 2)
            B[i-1] = 2/(h**2)
        b[0] = 7
        for i in range(1, n):
            b[i] = 4*((i*h)**2) + 10*i*h + 6 
        b[n] = 14



        v = TDMASolve(A, C, B, b)

        
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
                    "решений наблюдается в точке x=" + str(np.argmax(abs(u-v))*h) + ", значение которой равно:"
                    + str(max(abs(u-v)))))

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

    def build_test_2(self, n):
        ksi = np.pi/4
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

        def a_func(x):
            nonlocal h
            if x <= ksi:
                return 1
            elif x-h >= ksi:
                return 2
            elif x-h<ksi and x > ksi:
                return h/((ksi-x+h) + (x-ksi)/2)
        def d_func(x):
            nonlocal h
            if x+h/2 <= ksi:
                return ksi + 1
            elif x-h/2 < ksi and x+h/2 > ksi:
                return (ksi+1)*(ksi-x+h/2)/h + np.pi**2*(x+h/2-ksi)/(8*h)
            elif x-h/2 >= ksi:
                return np.pi**2/8
            
        def f(x):
            nonlocal h
            if x+h/2 <= ksi:
                return 1
            elif x-h/2 < ksi and x+h/2 > ksi:
                return (ksi-x+h/2)/h + np.sqrt(2)*(x+h/2-ksi)/(2*h)
            elif x-h/2 >= ksi:
                return np.sqrt(2)/2

        def acc_sol(x):
            
            c1 = -0.23104783711550236358  # -0.231048
            c2 = 0.67094868360394499529 # 0.670949
            c3 = -0.32356455123943433927 # -0.323565
            c4 = 0.29940138539753108349 # 0.299401
            # if x < 0.125:
            #     return c1*np.exp(np.sqrt(1.125/np.exp(-0.125))*x) + c2*np.exp(-np.sqrt(1.125/np.exp((-0.125)))*x) + np.cos(0.125)/np.exp(-0.125)
            # else:
            #     return c3*np.exp(np.sqrt(np.exp(-(0.125**2)))*x) + c4*np.exp(-np.sqrt(np.exp(-(0.125**2)))*x) +1/(np.exp(-(0.125**2)))
            if x < ksi:
                return float64(c1*np.exp(np.sqrt(np.pi/4+1)*x) + c2*np.exp(-np.sqrt(np.pi/4+1)*x) + 1/(np.pi/4 + 1))
            else:
                return float64(c3*np.exp(np.pi*x/4) + c4 * np.exp(-np.pi*x/4) + 4*np.sqrt(2)/(np.pi*np.pi))
    
        h = float64(1/n)
        v = np.zeros(n+1)
        u = np.zeros(n+1)
        d = np.zeros(n+1)
        d[0] = 1
        for i in range(1, n):
            d[i] = f(i*h)
        d[n] = 0

        C = np.zeros(n-1)
        A = np.zeros(n-1)
        B = np.zeros(n-1)
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
        self.label_2.setText(QtCore.QCoreApplication.translate("MainWindow", 
                    "Результаты расчета: \n Для решения тестовой задачи \n использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение точного и приближенного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(y))*h) + ", значение которой равно:"
                    + str(max(abs(y)))))

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

    def build_test_3(self, n):
        ksi = np.pi/4
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
                return np.sqrt(2) * np.cos(x-h/2)
           
            elif x-h < ksi and x > ksi:
                # left_mean = (ksi+x-h)/2
                # left_int = (ksi-(x-h))/(np.sqrt(2)*np.cos(left_mean))
                # # right_mean = (x+ksi)/2
                # right_int = (x-ksi)/2
                # full_int = left_int + right_int
                return h / ((ksi - (x - h)) / (np.sqrt(2) * np.cos((x - h + ksi)/2)) +
                     (x - ksi) / 2) 
            
            elif x-h >= ksi:
                return 2

        def d_func(x):
            nonlocal h
            if x+h/2 <= ksi:
                return x + 1 

            elif x-h/2 < ksi and x+h/2 > ksi:
                # left_mean = (ksi+(x-h/2))/2
                # left_int = (ksi-(x-h/2))*(left_mean+1)
                # right_mean = (x+h/2+ksi)/2
                # right_int = (x+h/2-ksi) * 2*right_mean**2
                # return (left_int + right_int)/h
                return ((ksi - (x - h / 2)) * ((x + h / 2 - ksi)/2 + 1) +
                     (x + h / 2 - ksi) * (2*((x + h / 2 - ksi)/2**2))) / h   
            
            elif (x-h/2) >= ksi:
                return (x**2)*2        
        def f(x):
            nonlocal h
            if x+h/2 <= ksi:
                return np.sin(2*x)
            elif x-h/2 < ksi and x+h/2 > ksi:
                # left_mean = (ksi+(x-h/2))/2
                # left_int = (ksi-(x-h/2)) * np.sin(2*left_mean)
                # right_mean = (x+h/2+ksi)/2
                # right_int = (x+h/2-ksi)*np.sin(right_mean)
                # return (left_int + right_int)/h
                return ((ksi - (x - h / 2)) * (np.sin(2*(x + h / 2 - ksi)/2)) + 
                        (x + h / 2 - ksi) * (np.sin((x + h / 2 - ksi)/2))) / h

            elif x-h/2 >= ksi:
                return np.sin(x)
            

        h = float64(1/n)
        v = np.zeros(n+1, float64)
        d = np.zeros(n+1, float64)
        d[0] = 1
        for i in range(1, n):
            d[i] = f(i*h)
        d[n] = 0

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
        d[0] = 1
        for i in range(1, n_new):
            d[i] = f(i*h)
        d[n_new] = 0

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
        h *= 2
        self.tableWidget.setRowCount(n+1)
        for i in range(0, n+1):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i*h)))
            self.tableWidget.setItem(i, 4, QtWidgets.QTableWidgetItem(str(v2[i])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(v[i])))
            self.tableWidget.setItem(i, 6, QtWidgets.QTableWidgetItem(str(v2[i] - v[i])))
        self.label_2.setText(QtCore.QCoreApplication.translate("MainWindow", 
                    "Результаты расчета: \n Для решения тестовой задачи \n использована" 
                    + " сетка с числом разбиений по x: \n n = " + str(n) + 
                    "\n Максимальное отклонение численного с двойным шагом и численного \n"+ 
                    "решений наблюдается в точке x=" + str(np.argmax(abs(y))*h) + ", значение которой равно:"
                    + str(max(abs(y)))))

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

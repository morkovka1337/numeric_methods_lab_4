# -*- coding: utf-8 -*-
import sys
import math
import math_part
# Импортируем наш интерфейс из файла
from Form_for_4_lab import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from MyMplCanc import MtMplCanv
from numpy import float64
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtGui, QtCore
from MyMplCanc import MtMplCanv
from MyMplCanc import MtMplCanv2
from matplotlib.figure import Figure
class MyWin(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, *args, **kwargs):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.MyFunction)
    def MyFunction(self):
        n = int(self.textEdit.toPlainText())
        if self.comboBox.currentText() == "Тестовая задача 1":
            math_part.mathpart.build_test_1(self, n)
        elif self.comboBox.currentText() == "Тестовая задача 2":
            math_part.mathpart.build_test_2(self, n)
        elif self.comboBox.currentText() == "Основная задача":
            math_part.mathpart.build_test_3(self, n)
            
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
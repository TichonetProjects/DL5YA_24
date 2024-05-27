import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QDialogButtonBox


class MyMainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("PyQt5 class")
        self.setGeometry(300,400,400,200)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Hi there",self)
        self.label.setGeometry(50,20,100,10)
        self.button = QtWidgets.QPushButton("OK",self)
        self.button.setGeometry(10,20,20,10)
        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.label.setText("You pressed the button")
        self.label.adjustSize()






app = QApplication(sys.argv)
win = MyMainWindow()
win.show()
sys.exit(app.exec_())




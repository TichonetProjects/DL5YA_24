import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QDialogButtonBox
import MyMainWindow


def button_clicked():
    msg = QMessageBox()
    msg.setWindowTitle("My first popup")
    msg.setText("Hi There")
    msg.exec_()

app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(150,200,300,40)
button = QtWidgets.QPushButton("OK",win)
button.clicked.connect(button_clicked)
win.show()
sys.exit(app.exec_())



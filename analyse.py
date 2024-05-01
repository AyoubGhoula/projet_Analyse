import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QMessageBox, QFileDialog, QWidget, QVBoxLayout , QMainWindow
from PyQt5.uic import loadUi



class Analyse_Windows(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("analyse.ui", self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Analyse_Windows()
    window.show()
    sys.exit(app.exec_())
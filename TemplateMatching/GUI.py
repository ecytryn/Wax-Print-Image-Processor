import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QTextEdit, QVBoxLayout
from PyQt6.QtGui import QIcon



class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Show me them pearly whites!")
        self.resize(300, 200)

        # container for layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        #widgets
        self.inputField = QLineEdit()
        button = QPushButton("Ok!", clicked=self.sayHello)
        self.output = QTextEdit()

        layout.addWidget(self.inputField)
        layout.addWidget(button)
        layout.addWidget(self.output)

    def sayHello(self):
        inputText = self.inputField.text()
        self.output.setText(f"Hello {inputText}")

app = QApplication(sys.argv)
app.setStyleSheet('''
    QWidget {
        font-size: 25px
    }
''') # css!
window = MyApp()
window.show()

app.exec()
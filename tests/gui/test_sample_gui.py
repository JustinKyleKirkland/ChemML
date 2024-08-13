import unittest

from PyQt5.QtCore import QPoint, Qt, QUrl
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from gui.sample_gui import CSVInteractiveApp


class TestCSVInteractiveApp(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        self.window = CSVInteractiveApp()

    def tearDown(self):
        self.window.close()

    def test_load_csv(self):
        # Simulate clicking the load button
        QTest.mouseClick(self.window.load_button, Qt.LeftButton)
        # Assert that the file dialog is opened
        self.assertTrue(self.window.isVisible())

    def test_display_csv_image(self):
        # Simulate loading a CSV file
        self.window.display_csv_image("test.csv")
        # Assert that the table widget is populated with data
        self.assertEqual(self.window.table_widget.rowCount(), 3)
        self.assertEqual(self.window.table_widget.columnCount(), 2)

    def test_drag_and_drop_csv(self):
        # Simulate dragging and dropping a CSV file
        event = QTest.QDropEvent(
            QPoint(0, 0),
            Qt.CopyAction,
            Qt.DropAction(Qt.CopyAction | Qt.MoveAction),
            QUrl.fromLocalFile("test.csv"),
            None,
        )
        self.window.dropEvent(event)
        # Assert that the CSV file is displayed in the table widget
        self.assertEqual(self.window.table_widget.rowCount(), 3)
        self.assertEqual(self.window.table_widget.columnCount(), 2)

    def test_close_event(self):
        # Simulate closing the application
        self.window.closeEvent(None)
        # Assert that the application is closed
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()

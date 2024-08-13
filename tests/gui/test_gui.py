import logging
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from PyQt5.QtWidgets import QApplication

from gui.gui import CSVInteractiveApp


class TestCSVInteractiveApp(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        self.window = CSVInteractiveApp()

    def tearDown(self):
        self.app.quit()

    @patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName")
    def test_load_csv(self, mock_getOpenFileName):
        mock_getOpenFileName.return_value = ("test.csv", "")

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {
                    "Column1": [1, 2, 3],
                    "Column2": [4, 5, 6],
                }
            )

            self.window.load_csv()

            self.assertEqual(self.window.table_widget.rowCount(), 3)
            self.assertEqual(self.window.table_widget.columnCount(), 2)
            self.assertEqual(
                self.window.table_widget.horizontalHeaderItem(0).text(), "Column1"
            )
            self.assertEqual(self.window.table_widget.item(0, 0).text(), "1")
            self.assertEqual(self.window.table_widget.item(1, 1).text(), "5")

    @patch("pandas.read_csv")
    def test_display_csv_image(self, mock_read_csv):
        # Mock pandas read_csv to return a sample DataFrame
        mock_read_csv.return_value = pd.DataFrame(
            {"Column1": [1, 2], "Column2": [3, 4]}
        )

        # Call display_csv_image directly
        self.window.display_csv_image("test.csv")

        # Check if the data was displayed correctly
        self.assertEqual(self.window.table_widget.rowCount(), 2)
        self.assertEqual(self.window.table_widget.columnCount(), 2)
        self.assertEqual(
            self.window.table_widget.horizontalHeaderItem(0).text(), "Column1"
        )
        self.assertEqual(self.window.table_widget.item(1, 1).text(), "4")

    @patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName")
    def test_load_csv_no_file(self, mock_getOpenFileName):
        # Mock the file dialog to return no file
        mock_getOpenFileName.return_value = ("", "")

        with patch("pandas.read_csv"):
            with self.assertLogs(logging.getLogger(), level="INFO") as log:
                self.window.load_csv()
                # Check if the log contains the expected message
                self.assertIn("INFO:root:No file selected", log.output)

    def test_dragEnterEvent_accept(self):
        event = MagicMock()
        event.mimeData().hasUrls.return_value = True
        self.window.dragEnterEvent(event)
        event.accept.assert_called_once()

    def test_dragEnterEvent_ignore(self):
        event = MagicMock()
        event.mimeData().hasUrls.return_value = False
        self.window.dragEnterEvent(event)
        event.ignore.assert_called_once()

    @patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName")
    def test_dropEvent(self, mock_getOpenFileName):
        event = MagicMock()
        mime_data = MagicMock()
        mime_data.hasUrls.return_value = True
        mime_data.urls.return_value = [
            MagicMock(toLocalFile=MagicMock(return_value="test.csv"))
        ]
        event.mimeData.return_value = mime_data

        # Mock pandas read_csv
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {"Column1": [1, 2], "Column2": [3, 4]}
            )

            self.window.dropEvent(event)

            # Check if the data was displayed correctly
            self.assertEqual(self.window.table_widget.rowCount(), 2)
            self.assertEqual(self.window.table_widget.columnCount(), 2)

    @patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName")
    def test_closeEvent(self, mock_getOpenFileName):
        event = MagicMock()
        mime_data = MagicMock()
        mime_data.hasUrls.return_value = True
        mime_data.urls.return_value = [
            MagicMock(toLocalFile=MagicMock(return_value="test.csv"))
        ]
        event.mimeData.return_value = mime_data

        # Mock pandas read_csv
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {"Column1": [1, 2], "Column2": [3, 4]}
            )

            self.window.dropEvent(event)

            # Check if the data was displayed correctly
            self.assertEqual(self.window.table_widget.rowCount(), 2)
            self.assertEqual(self.window.table_widget.columnCount(), 2)


if __name__ == "__main__":
    unittest.main()

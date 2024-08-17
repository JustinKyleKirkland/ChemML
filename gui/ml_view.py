# ml_view.py

from PyQt5.QtWidgets import (
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)


class MLView(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Machine Learning View")

        # Layout setup
        self.layout = QVBoxLayout()

        # Create a label for the machine learning methods
        ml_methods_label = QLabel("Select Machine Learning Methods:")
        self.layout.addWidget(ml_methods_label)

        # Create a QListWidget for ML methods with checkboxes
        self.ml_methods_list = QListWidget()
        self.ml_methods_list.setSelectionMode(QListWidget.MultiSelection)
        self.ml_methods_list.addItems(
            [
                "Linear Regression",
                "Logistic Regression",
                "Decision Trees",
                "Random Forest",
                "Support Vector Machines",
                "Neural Networks",
            ]
        )

        self.layout.addWidget(self.ml_methods_list)

        # Add the layout to the widget
        self.setLayout(self.layout)

    def get_selected_models(self):
        """
        Returns a list of selected machine learning models.

        Returns:
            List[str]: A list of selected ML methods.
        """
        selected_items = self.ml_methods_list.selectedItems()
        return [item.text() for item in selected_items]

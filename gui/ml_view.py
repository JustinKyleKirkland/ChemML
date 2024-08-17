# ml_view.py

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class MLView(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Machine Learning View")

        # Main layout setup
        self.main_layout = QVBoxLayout()

        # Create a horizontal layout for the two label/list box columns
        self.h_layout = QHBoxLayout()

        # Create the first column with label and list widget
        self.left_column_layout = QVBoxLayout()
        self.ml_methods_label = QLabel("Select Machine Learning Methods:")
        self.left_column_layout.addWidget(self.ml_methods_label)

        self.available_methods_list = QListWidget()
        self.available_methods_list.setSelectionMode(QListWidget.MultiSelection)
        self.available_methods_list.addItems(
            [
                "Linear Regression",
                "Logistic Regression",
                "Decision Trees",
                "Random Forest",
                "Support Vector Machines",
                "Neural Networks",
            ]
        )
        self.left_column_layout.addWidget(self.available_methods_list)

        # Create buttons for moving items
        self.move_right_button = QPushButton("→")
        self.move_left_button = QPushButton("←")
        self.move_right_button.clicked.connect(self.move_to_selected)
        self.move_left_button.clicked.connect(self.move_to_available)

        # Add buttons to a vertical layout
        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.move_right_button)
        self.button_layout.addWidget(self.move_left_button)

        # Create the second column with label and list widget
        self.right_column_layout = QVBoxLayout()
        selected_methods_label = QLabel("Selected ML Methods")
        self.right_column_layout.addWidget(selected_methods_label)

        self.selected_methods_list = QListWidget()
        self.selected_methods_list.setSelectionMode(QListWidget.MultiSelection)
        self.right_column_layout.addWidget(self.selected_methods_list)

        # Add columns and buttons to the horizontal layout
        self.h_layout.addLayout(self.left_column_layout)
        self.h_layout.addLayout(self.button_layout)
        self.h_layout.addLayout(self.right_column_layout)

        # Add the horizontal layout to the main layout
        self.main_layout.addLayout(self.h_layout)

        # Set the main layout for the widget
        self.setLayout(self.main_layout)

        # Set equal sizes for list widgets
        self.available_methods_list.setMinimumWidth(200)
        self.selected_methods_list.setMinimumWidth(200)

    def move_to_selected(self) -> None:
        """
        Move selected items from the available methods list to the selected methods list.
        """
        selected_items = self.available_methods_list.selectedItems()
        for item in selected_items:
            self.selected_methods_list.addItem(item.text())
            self.available_methods_list.takeItem(self.available_methods_list.row(item))

    def move_to_available(self) -> None:
        """
        Move selected items from the selected methods list back to the available methods list.
        """
        selected_items = self.selected_methods_list.selectedItems()
        for item in selected_items:
            self.available_methods_list.addItem(item.text())
            self.selected_methods_list.takeItem(self.selected_methods_list.row(item))

    def get_selected_models(self):
        """
        Returns a list of selected machine learning models.

        Returns:
            List[str]: A list of selected ML methods.
        """
        selected_items = self.selected_methods_list.selectedItems()
        return [item.text() for item in selected_items]

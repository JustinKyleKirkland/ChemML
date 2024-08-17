import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MLView(QWidget):
    # Define a signal to indicate data readiness
    data_ready_signal = pyqtSignal(pd.DataFrame)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Machine Learning View")

        self.main_layout = QVBoxLayout()
        self.setup_ml_methods_section()
        self.setup_target_selection_section()
        self.setup_feature_columns_section()

        self.setLayout(self.main_layout)
        self.initialize_widgets()
        self.previous_target_column = None

        self.data_ready_signal.connect(self.update_column_selection)
        self.target_combo.currentIndexChanged.connect(self.update_feature_columns)

    def setup_ml_methods_section(self) -> None:
        """Sets up the section for machine learning methods."""
        self.h_layout_ml_methods = QHBoxLayout()

        # First column with label and list widget
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

        # Buttons for moving items
        self.move_right_button = QPushButton("→")
        self.move_left_button = QPushButton("←")
        self.move_right_button.clicked.connect(self.move_to_selected)
        self.move_left_button.clicked.connect(self.move_to_available)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.move_right_button)
        self.button_layout.addWidget(self.move_left_button)

        # Second column with label and list widget
        self.right_column_layout = QVBoxLayout()
        self.selected_methods_label = QLabel("Selected ML Methods")
        self.right_column_layout.addWidget(self.selected_methods_label)

        self.selected_methods_list = QListWidget()
        self.selected_methods_list.setSelectionMode(QListWidget.MultiSelection)
        self.right_column_layout.addWidget(self.selected_methods_list)

        # Combine columns and buttons
        self.h_layout_ml_methods.addLayout(self.left_column_layout)
        self.h_layout_ml_methods.addLayout(self.button_layout)
        self.h_layout_ml_methods.addLayout(self.right_column_layout)

        self.main_layout.addLayout(self.h_layout_ml_methods)

    def setup_target_selection_section(self) -> None:
        """Sets up the section for selecting the target column."""
        self.h_layout_target = QHBoxLayout()

        self.form_layout = QFormLayout()
        self.target_label = QLabel("Target Column:")
        self.target_combo = QComboBox()
        self.target_combo.addItem("Select Target Column")

        self.form_layout.addRow(self.target_label, self.target_combo)
        self.h_layout_target.addLayout(self.form_layout)

        self.main_layout.addLayout(self.h_layout_target)

    def setup_feature_columns_section(self) -> None:
        """Sets up the section for feature columns."""
        self.h_layout_features = QHBoxLayout()
        self.feature_label = QLabel("Feature Columns:")
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_feature)

        self.h_layout_features.addWidget(self.feature_label)
        self.h_layout_features.addWidget(self.feature_list)
        self.h_layout_features.addWidget(self.remove_button)

        self.main_layout.addLayout(self.h_layout_features)

    def initialize_widgets(self) -> None:
        """Initializes widget properties."""
        for widget in [
            self.available_methods_list,
            self.selected_methods_list,
            self.feature_list,
        ]:
            widget.setMinimumWidth(200)

    def set_column_names(self, columns: list) -> None:
        """
        Set the columns available for selection in the target combo box.

        Parameters:
            columns (list of str): The column names from the loaded CSV.
        """
        self.target_combo.clear()
        self.target_combo.addItem("Select Target Column")
        self.target_combo.addItems(columns)

    def move_to_selected(self) -> None:
        """Move selected items from the available methods list to the selected methods list."""
        selected_items = self.available_methods_list.selectedItems()
        for item in selected_items:
            self.selected_methods_list.addItem(item.text())
            self.available_methods_list.takeItem(self.available_methods_list.row(item))

    def move_to_available(self) -> None:
        """Move selected items from the selected methods list back to the available methods list."""
        selected_items = self.selected_methods_list.selectedItems()
        for item in selected_items:
            self.available_methods_list.addItem(item.text())
            self.selected_methods_list.takeItem(self.selected_methods_list.row(item))

    def get_selected_models(self) -> list:
        """
        Returns a list of selected machine learning models.

        Returns:
            List[str]: A list of selected ML methods.
        """
        return [item.text() for item in self.selected_methods_list.findItems("*", 0)]

    def get_target_column(self) -> str:
        """
        Returns the selected target column.

        Returns:
            str: The selected target column.
        """
        return self.target_combo.currentText()

    def get_feature_columns(self) -> list:
        """
        Returns the selected feature columns.

        Returns:
            List[str]: A list of selected feature columns.
        """
        return [item.text() for item in self.feature_list.findItems("*", 0)]

    def set_data_ready_signal(self, receiver) -> None:
        """
        Connects the data_ready_signal to a receiver function.

        Parameters:
            receiver (function): The function to be called when data is ready.
        """
        self.data_ready_signal.connect(receiver)

    def remove_feature(self) -> None:
        """Remove selected feature columns from the list."""
        selected_items = self.feature_list.selectedItems()
        for item in selected_items:
            self.feature_list.takeItem(self.feature_list.row(item))

    def update_feature_columns(self) -> None:
        """Updates the feature columns based on the selected target column."""
        current_target = self.get_target_column()

        if current_target and current_target != "Select Target Column":
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                if item.text() == current_target:
                    self.feature_list.takeItem(i)
                    break

            if self.previous_target_column and self.previous_target_column != "Select Target Column":
                self.feature_list.addItem(self.previous_target_column)

            self.previous_target_column = current_target
        else:
            self.feature_list.clear()
            self.previous_target_column = None

    def update_column_selection(self, df: pd.DataFrame) -> None:
        """
        Updates the target column dropdown and feature column list based on the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the new data.
        """
        columns = df.columns.tolist()
        self.set_column_names(columns)

        self.feature_list.clear()
        self.feature_list.addItems(columns)

        if self.previous_target_column in columns:
            self.feature_list.addItem(self.previous_target_column)
        elif self.previous_target_column == "Select Target Column":
            self.previous_target_column = None

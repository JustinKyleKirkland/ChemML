import logging

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ml_backend.ml_backend import download_results_as_json, run_ml_methods


class MLView(QWidget):
    # Define a signal to indicate data readiness
    data_ready_signal = pyqtSignal(pd.DataFrame)

    def __init__(self) -> None:
        super().__init__()
        self.logger: logging.Logger = logging.getLogger("MLView")
        self.df = pd.DataFrame()

        self.setWindowTitle("Machine Learning View")

        self.main_layout = QVBoxLayout()
        self.setup_ml_methods_section()
        self.setup_target_selection_section()
        self.setup_feature_columns_section()
        self.setup_run_ml_button()  # Add the run ML button section at the bottom

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
                "Ridge Regression",
                "Lasso Regression",
                "ElasticNet Regression",
                "Decision Trees",
                "Random Forest",
                "Support Vector Machines",
                "Neural Networks",
                "Gradient Boosting",
                "AdaBoost",
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

        # First column with label and list widget
        self.left_column_layout = QVBoxLayout()
        self.feature_label = QLabel("Available Feature Columns:")
        self.left_column_layout.addWidget(self.feature_label)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        self.left_column_layout.addWidget(self.feature_list)

        # Buttons for moving items
        self.move_right_button = QPushButton("→")
        self.move_left_button = QPushButton("←")
        self.move_right_button.clicked.connect(self.move_feature_to_selected)
        self.move_left_button.clicked.connect(self.move_feature_to_available)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.move_right_button)
        self.button_layout.addWidget(self.move_left_button)

        # Second column with label and list widget
        self.right_column_layout = QVBoxLayout()
        self.selected_features_label = QLabel("Selected Feature Columns")
        self.right_column_layout.addWidget(self.selected_features_label)

        self.selected_feature_list = QListWidget()
        self.selected_feature_list.setSelectionMode(QListWidget.MultiSelection)
        self.right_column_layout.addWidget(self.selected_feature_list)

        # Combine columns and buttons
        self.h_layout_features.addLayout(self.left_column_layout)
        self.h_layout_features.addLayout(self.button_layout)
        self.h_layout_features.addLayout(self.right_column_layout)

        self.main_layout.addLayout(self.h_layout_features)

    def setup_run_ml_button(self) -> None:
        """Sets up the button to run machine learning methods."""
        self.run_ml_button = QPushButton("Run ML Methods")
        self.run_ml_button.clicked.connect(self.run_ml_methods_clicked)

        # Style the button
        self.run_ml_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.run_ml_button.setFixedHeight(40)

        # Add the button to the main layout
        self.main_layout.addWidget(self.run_ml_button)

        self.plot_results_button = QPushButton("Plot Results")
        self.plot_results_button.clicked.connect(self.plot_results_clicked)
        self.plot_results_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.plot_results_button.setFixedHeight(40)

        self.main_layout.addWidget(self.plot_results_button)

    def initialize_widgets(self) -> None:
        """Initializes widget properties."""
        for widget in [
            self.available_methods_list,
            self.selected_methods_list,
            self.feature_list,
            self.selected_feature_list,
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

    def move_feature_to_selected(self) -> None:
        """Move selected features from the available features list to the selected features list."""
        selected_items = self.feature_list.selectedItems()
        for item in selected_items:
            self.selected_feature_list.addItem(item.text())
            self.feature_list.takeItem(self.feature_list.row(item))

    def move_feature_to_available(self) -> None:
        """Move selected features from the selected features list back to the available features list."""
        selected_items = self.selected_feature_list.selectedItems()
        for item in selected_items:
            self.feature_list.addItem(item.text())
            self.selected_feature_list.takeItem(self.selected_feature_list.row(item))

    def get_selected_models(self) -> list:
        """
        Returns a list of selected machine learning models.

        Returns:
            List[str]: A list of selected ML methods.
        """
        selected_models = []
        for i in range(self.selected_methods_list.count()):
            item = self.selected_methods_list.item(i)
            selected_models.append(item.text())
        return selected_models

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
        feature_columns = []
        for i in range(self.selected_feature_list.count()):
            item = self.selected_feature_list.item(i)
            feature_columns.append(item.text())
        return feature_columns

    def set_data_ready_signal(self, receiver) -> None:
        """
        Connects the data_ready_signal to a receiver function.

        Parameters:
            receiver (function): The function to be called when data is ready.
        """
        self.data_ready_signal.connect(receiver)

    def remove_feature(self) -> None:
        """Remove selected feature columns from the list."""
        selected_items = self.selected_feature_list.selectedItems()
        for item in selected_items:
            self.selected_feature_list.takeItem(self.selected_feature_list.row(item))

    def update_feature_columns(self) -> None:
        """Updates the feature columns based on the selected target column."""
        current_target = self.get_target_column()

        # Update the feature columns list
        self.feature_list.clear()
        columns = self.df.columns.tolist()  # Get all column names from the DataFrame

        # Remove the selected target column from the list if it exists
        if current_target and current_target != "Select Target Column":
            columns.remove(current_target)

        self.feature_list.addItems(columns)

        # Ensure the previous target column is correctly managed
        if self.previous_target_column and self.previous_target_column != "Select Target Column":
            if self.previous_target_column in columns:
                self.feature_list.addItem(self.previous_target_column)

        self.previous_target_column = current_target

    def update_column_selection(self, df: pd.DataFrame) -> None:
        """
        Updates the target column dropdown and feature column list based on the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the new data.
        """
        columns = df.columns.tolist()
        self.set_column_names(columns)

        # Update feature columns list
        self.feature_list.clear()
        self.feature_list.addItems(columns)

        # Update selected features list
        self.selected_feature_list.clear()
        if self.previous_target_column in columns:
            self.selected_feature_list.addItem(self.previous_target_column)
        elif self.previous_target_column == "Select Target Column":
            self.previous_target_column = None

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Sets the DataFrame for the ML view.

        Args:
            df (pd.DataFrame): The DataFrame to be used.
        """
        self.df = df
        self.update_column_selection(df)

    def run_ml_methods_clicked(self) -> None:
        target_column = self.get_target_column()
        feature_columns = self.get_feature_columns()
        selected_models = self.get_selected_models()

        self.logger.info("Running ML methods...")
        self.logger.info(f"Target Column: {target_column}")
        self.logger.info(f"Feature Columns: {feature_columns}")
        self.logger.info(f"Selected Models: {selected_models}")

        if not target_column or target_column == "Select Target Column":
            self.logger.warning("No target column selected.")
            QMessageBox.warning(self, "Warning", "Please select a target column.")
            return

        if not feature_columns:
            self.logger.warning("No feature columns selected.")
            QMessageBox.warning(self, "Warning", "Please select feature columns.")
            return

        if not selected_models:
            self.logger.warning("No ML methods selected.")
            QMessageBox.warning(self, "Warning", "Please select machine learning methods.")
            return

        if self.df.empty:
            self.logger.warning("No data loaded.")
            QMessageBox.warning(self, "Warning", "Please load a CSV file.")
            return

        try:
            # Call the ML methods in the backend
            results = run_ml_methods(self.df, target_column, feature_columns, selected_models)

            self.logger.info("ML methods executed successfully.")

            # Find the best method
            best_method = max(results.keys(), key=lambda m: results[m].get("cv_mean_score", 0))
            best_result = results[best_method]

            # Prepare the result string for the best method
            result_str = (
                f"Best Method: {best_method}\n"
                f"  CV Mean Score: {best_result['cv_mean_score']:.4f}\n"
                f"  CV Std Score: {best_result['cv_std_score']:.4f}\n"
                f"  Train MSE: {best_result.get('train_mse', 'N/A'):.4f}\n"
                f"  Test MSE: {best_result.get('test_mse', 'N/A'):.4f}\n"
                f"  Train R2: {best_result.get('train_r2', 'N/A'):.4f}\n"
                f"  Test R2: {best_result.get('test_r2', 'N/A'):.4f}\n"
                f"  Best Hyperparameters: {best_result.get('best_hyperparameters', 'N/A')}\n"
            )

            # Show the result of the best method
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Best ML Method Result")
            msg_box.setText(result_str)

            close_button = msg_box.addButton("Close", QMessageBox.RejectRole)
            save_button = msg_box.addButton("Save All Results", QMessageBox.ActionRole)
            msg_box.exec_()

            if msg_box.clickedButton() == save_button:
                filename, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "JSON Files (*.json)")
                if filename:
                    download_results_as_json(results, filename)
                    self.logger.info(f"Results saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error running ML methods: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error running ML methods: {e}")

    def plot_results_clicked(self) -> None:
        """Opens a dialog with a checklist to select ML methods for plotting."""
        selected_methods = self.get_selected_models()
        if not selected_methods:
            QMessageBox.warning(self, "Warning", "No ML methods available for plotting.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Method to Plot")

        layout = QVBoxLayout()
        checkboxes = {}
        for method in selected_methods:
            checkbox = QCheckBox(method)
            layout.addWidget(checkbox)
            checkboxes[method] = checkbox

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(
            lambda: self.plot_selected_methods([m for m, cb in checkboxes.items() if cb.isChecked()])
        )
        button_box.rejected.connect(dialog.reject)

        layout.addWidget(button_box)
        dialog.setLayout(layout)
        dialog.exec_()

    def plot_selected_methods(self, methods: list) -> None:
        """Plots the target vs. predicted values for the selected methods."""
        if not methods:
            QMessageBox.warning(self, "Warning", "No methods selected for plotting.")
            return

        target_column = self.get_target_column()
        if not target_column:
            QMessageBox.warning(self, "Warning", "No target column selected.")
            return

        results = run_ml_methods(self.df, target_column, self.get_feature_columns(), methods)

        for method in methods:
            result = results.get(method, {})
            test_predictions = result.get("test_predictions", [])
            test_actuals = result.get("y_test", [])

            if len(test_predictions) != len(test_actuals):
                QMessageBox.warning(self, "Warning", f"Prediction length mismatch for {method}.")
                continue

            if not test_predictions:
                QMessageBox.warning(self, "Warning", f"No predictions available for {method}.")
                continue

            plt.figure()
            plt.scatter(test_actuals, test_predictions, alpha=0.7)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{method} - Actual vs Predicted")
            plt.grid(True)
            plt.show()

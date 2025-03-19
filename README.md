# ChemML

## Description

ChemML is a powerful Python-based application that provides a graphical user interface (GUI) for chemical data analysis and machine learning. It enables users to upload CSV files containing chemical data, perform data preprocessing, and apply various machine learning models for predictive analysis.

## Features

- **Interactive CSV Data Management**

  - Upload and view CSV files
  - Data preprocessing capabilities
  - Column selection and manipulation
  - Data visualization tools

- **Machine Learning Capabilities**

  - Multiple model selection options
  - Advanced model configuration
  - Training and evaluation tools
  - Performance metrics visualization

- **Data Visualization**
  - Interactive plotting capabilities
  - Multiple chart types
  - Customizable visualization options

## Project Structure

```
ChemML/
├── gui/                    # GUI components
│   ├── csv_view.py        # CSV data management interface
│   ├── ml_view.py         # Basic ML model interface
│   ├── ml_advanced_view.py # Advanced ML configuration
│   ├── plot_view.py       # Data visualization
│   └── gui.py             # Main GUI application
├── ml_backend/            # Machine learning backend
│   ├── ml_backend.py      # Core ML functionality
│   └── model_configs.py   # ML model configurations
├── utils/                 # Utility functions
├── tests/                 # Test suite
└── data_sets/            # Sample datasets
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JustinKyleKirkland/ChemML.git
   ```
2. Navigate to the project directory
   ```bash
   cd ChemML
   ```
3. Create virtual environment
   ```bash
   python -m venv .venv
   ```
4. Activate virtual environment
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```
5. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Activate your virtual environment if not already activated
2. Run the application:
   ```bash
   python main.py
   ```
3. Use the GUI to:
   - Upload your CSV file
   - Preprocess your data
   - Select and configure ML models
   - Train and evaluate models
   - Visualize results

## Dependencies

Key dependencies include:

- PyQt5 for the GUI
- pandas for data manipulation
- scikit-learn for machine learning
- matplotlib and seaborn for visualization
- RDKit for chemical data processing
- NumPy for numerical computations

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Guidelines for contributing to the project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the GNU Public License - see the [LICENSE](LICENSE) file for details.

## Author

Justin Kyle Kirkland

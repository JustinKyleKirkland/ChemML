# ChemML

## Description

ChemML is a powerful Python-based application that provides a modern, user-friendly graphical interface for chemical data analysis and machine learning. It enables researchers and chemists to upload CSV files containing chemical data, perform data preprocessing, visualize relationships, and apply various machine learning models for predictive analysis.

## Features

- **Interactive CSV Data Management**
  - Upload, view, and export CSV files
  - Advanced data filtering and transformations
  - Specialized handling of chemical SMILES notation
  - One-hot encoding and data imputation capabilities
  - Undo/redo functionality for data operations

- **Chemistry-Specific Functionality**
  - Automatic SMILES structure detection and rendering (hover to preview a molecule)
  - Convert to canonical SMILES representation
  - Calculate basic molecular descriptors (MW, LogP, TPSA, H-donors/acceptors)
  - Pick from ~200 RDKit descriptors via a searchable dialog
  - Generate molecular fingerprints (Morgan/ECFP4, MACCS)

- **Advanced Data Visualization**
  - Interactive scatter plots with customizable markers
  - Trend line fitting with R² calculation
  - Customizable axes and plot styling options
  - Interactive zooming and plot manipulation

- **Machine Learning Capabilities**
  - 12 regression models (Linear/Ridge/Lasso/Elastic Net, Decision Tree, Random
    Forest, Gradient Boosting, AdaBoost, SVR, Neural Network, KNN, Gaussian Process)
  - Feature selection and optional standardization
  - Grid-search hyperparameter tuning with cross-validation
  - Configurable test split, CV folds and random seed via the **Settings** tab
  - Actual-vs-predicted plots and metrics (R², MSE) with JSON export

## Project Structure

```
ChemML/
├── chemml/               # Main package
│   ├── core/             # Core functionality
│   │   ├── chem/         # Chemistry-specific utilities
│   │   ├── data/         # Data handling utilities
│   │   └── models/       # ML model implementations
│   ├── ui/               # User interface components
│   │   ├── assets/       # UI assets (colors, icons, styles)
│   │   ├── controllers/  # Business logic controllers
│   │   ├── views/        # UI views (main, CSV, ML, plot)
│   │   └── widgets/      # Reusable UI components
│   └── utils/            # Utility functions
├── data_sets/            # Sample datasets
├── tests/                # Test suite
│   ├── gui/              # GUI component tests
│   └── utils/            # Utility function tests
└── logs/                 # Application logs
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JustinKKirkland/ChemML.git
   ```

2. Navigate to the project directory:
   ```bash
   cd ChemML
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install ChemML as an editable package (reads dependencies from
   `pyproject.toml` and adds a `chemml` console command):
   ```bash
   pip install -e .
   ```

## Usage

1. Activate your virtual environment if not already activated.

2. Run the application:
   ```bash
   python main.py
   # or, if installed with `pip install -e .`
   chemml
   ```

3. Use the intuitive tabbed interface to:
   - Load and preprocess your chemical data
   - Visualize relationships between variables
   - Configure and train machine learning models
   - Evaluate model performance

### Data Tab
- Use the toolbar to load CSV files containing chemical data
- Apply filters and transformations to prepare your data
- Right-click on SMILES columns for specialized chemical operations

### Visualize Tab
- Select X and Y variables to plot
- Configure marker styles, trend lines, and axes
- Analyze relationships with statistical measures

### ML Models Tab
- Select target variables and features
- Choose and configure appropriate ML models
- Train models and evaluate performance

### Settings Tab
- Configure the test-set size, cross-validation folds and random seed
- Toggle feature standardization
- Changes apply to the next training run

## Development

Install the development dependencies and run the test suite:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest              # run all tests
ruff check .        # lint
ruff format .       # format (tabs, 120 cols)
```

The project follows an MVC layout: pure logic lives in `chemml/core` and the
`*Controller` classes, while `chemml/ui/views` and `chemml/ui/widgets` contain
only the Qt presentation layer. Tests mirror this split under `tests/`.

## Dependencies

Key dependencies include:

- PyQt5 for the GUI interface
- pandas / NumPy for data manipulation
- scikit-learn for machine learning algorithms
- matplotlib / seaborn for data visualization
- RDKit for chemical data processing and visualization

Runtime dependencies are listed in `requirements.txt` (and `pyproject.toml`);
development tools are in `requirements-dev.txt`.

## Contributing

Guidelines for contributing to the project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## License

This project is licensed under the GNU Public License - see the [LICENSE](LICENSE) file for details.

## Version History

- **1.1.0** - Maintenance and feature update
  - Configurable ML settings tab (test split, CV folds, seed, scaling)
  - Searchable custom RDKit descriptor selection
  - Fixed descriptor/fingerprint alignment after filtering
  - Repaired and expanded the test suite; single-source logging; cleaner packaging

- **1.0.0** (April 2025) - Initial release with modernized UI
  - Enhanced CSV data handling
  - Interactive visualization capabilities
  - Specialized chemistry functionality
  - Integrated machine learning workflow

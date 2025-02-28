import pytest

from gui.ml_advanced_view import MLAdvancedView, ModelParamGroup, ParamInputWidget


@pytest.fixture
def test_model_config():
	"""Create a simple test model configuration."""
	return {
		"Test Model": {
			"param1": [1, 2, 3],
			"param2": [0.1, 0.2, 0.3],
		}
	}


@pytest.fixture
def ml_advanced_view(qtbot, test_model_config):
	"""Create the advanced ML settings view for testing."""
	view = MLAdvancedView()
	qtbot.addWidget(view)

	# Add a test model group
	group = ModelParamGroup("Test Model", test_model_config["Test Model"])
	group.params_changed.connect(view._handle_settings_change)
	view.model_groups["Test Model"] = group
	view.tab_widget.addTab(group, "Test")

	return view


@pytest.fixture
def param_input_widget(qtbot):
	"""Create a parameter input widget for testing."""
	param_name = "max_depth"
	param_config = [10, 20, 30]  # List of possible values
	widget = ParamInputWidget(param_name, param_config)
	qtbot.addWidget(widget)
	return widget


@pytest.fixture
def model_param_group(qtbot):
	"""Create a model parameter group for testing."""
	model_name = "Random Forest Regression"
	param_config = {
		"max_depth": [10, 20, 30],
		"min_samples_split": [2, 5, 10],
	}
	group = ModelParamGroup(model_name, param_config)
	qtbot.addWidget(group)
	return group


class TestParamInputWidget:
	def test_initialization(self, param_input_widget):
		"""Test initial state of parameter input widget."""
		assert param_input_widget.param_name == "max_depth"
		assert param_input_widget.type_combo.count() == 3
		assert param_input_widget.type_combo.currentText() == "Single Value"
		assert param_input_widget.single_value_widget is not None
		assert len(param_input_widget.range_widgets) > 0
		assert param_input_widget.stack.currentIndex() == 0  # Single value widget should be visible initially

	def test_value_change_signal(self, param_input_widget, qtbot):
		"""Test that value changes emit the correct signal."""
		signal_received = False
		param_name_received = None
		value_received = None

		def handle_param_change(name, value):
			nonlocal signal_received, param_name_received, value_received
			signal_received = True
			param_name_received = name
			value_received = value

		param_input_widget.param_changed.connect(handle_param_change)

		# Test single value change
		if hasattr(param_input_widget.single_value_widget, "setValue"):
			param_input_widget.single_value_widget.setValue(20)
			assert signal_received
			assert param_name_received == "max_depth"
			assert value_received == 20

		# Test range value change
		param_input_widget.type_combo.setCurrentText("Grid Search")
		signal_received = False
		if len(param_input_widget.range_widgets) >= 3:
			param_input_widget.range_widgets[0].setValue(10)  # Start
			param_input_widget.range_widgets[1].setValue(30)  # Stop
			param_input_widget.range_widgets[2].setValue(10)  # Step
			assert signal_received
			assert isinstance(value_received, list)
			assert len(value_received) > 0

	def test_type_change(self, param_input_widget, qtbot):
		"""Test changing the input type."""
		# Test initial state
		assert param_input_widget.stack.currentIndex() == 0  # Single value widget should be visible

		# Test Grid Search
		param_input_widget.type_combo.setCurrentText("Grid Search")
		assert param_input_widget.stack.currentIndex() == 1  # Range widgets should be visible

		# Test back to Single Value
		param_input_widget.type_combo.setCurrentText("Single Value")
		assert param_input_widget.stack.currentIndex() == 0  # Single value widget should be visible again


class TestModelParamGroup:
	def test_initialization(self, model_param_group):
		"""Test initial state of model parameter group."""
		assert model_param_group.model_name == "Random Forest Regression"
		assert model_param_group.current_params == {}

	def test_param_change_signal(self, model_param_group, qtbot):
		"""Test that parameter changes emit the correct signal."""
		signal_received = False
		model_name_received = None
		params_received = None

		def handle_params_change(name, params):
			nonlocal signal_received, model_name_received, params_received
			signal_received = True
			model_name_received = name
			params_received = params

		model_param_group.params_changed.connect(handle_params_change)

		# Find a parameter widget and change its value
		param_widgets = model_param_group.findChildren(ParamInputWidget)
		assert len(param_widgets) > 0

		first_param = param_widgets[0]
		if hasattr(first_param.single_value_widget, "setValue"):
			first_param.single_value_widget.setValue(20)

		assert signal_received
		assert model_name_received == "Random Forest Regression"
		assert isinstance(params_received, dict)


class TestMLAdvancedView:
	def test_initialization(self, ml_advanced_view):
		"""Test initial state of the advanced view."""
		assert ml_advanced_view.tab_widget.count() == 5  # General, Linear, Tree-Based, Other, and Test tabs
		assert "general" in ml_advanced_view.current_settings
		assert ml_advanced_view.current_settings["general"]["cv_folds"] == 5
		assert ml_advanced_view.current_settings["general"]["test_size"] == 0.2
		assert ml_advanced_view.current_settings["general"]["random_state"] == 42
		assert ml_advanced_view.current_settings["general"]["n_jobs"] == -1

	def test_general_settings_change(self, ml_advanced_view, qtbot):
		"""Test that general settings changes emit the correct signal."""
		signal_received = False
		settings_received = None

		def handle_settings_change(settings):
			nonlocal signal_received, settings_received
			signal_received = True
			settings_received = settings

		ml_advanced_view.settings_changed.connect(handle_settings_change)

		# Test changing CV folds
		with qtbot.waitSignal(ml_advanced_view.settings_changed, timeout=1000):
			ml_advanced_view.cv_folds.setValue(10)

		assert signal_received
		assert settings_received["general"]["cv_folds"] == 10

		# Test changing test size
		signal_received = False
		with qtbot.waitSignal(ml_advanced_view.settings_changed, timeout=1000):
			ml_advanced_view.test_size.setValue(0.3)

		assert signal_received
		assert settings_received["general"]["test_size"] == 0.3

		# Test changing random state
		signal_received = False
		with qtbot.waitSignal(ml_advanced_view.settings_changed, timeout=1000):
			ml_advanced_view.random_state.setValue(123)

		assert signal_received
		assert settings_received["general"]["random_state"] == 123

		# Test changing n_jobs
		signal_received = False
		with qtbot.waitSignal(ml_advanced_view.settings_changed, timeout=1000):
			ml_advanced_view.n_jobs.setValue(4)

		assert signal_received
		assert settings_received["general"]["n_jobs"] == 4

	def test_general_settings_ranges(self, ml_advanced_view):
		"""Test that general settings have correct ranges."""
		assert ml_advanced_view.cv_folds.minimum() == 2
		assert ml_advanced_view.cv_folds.maximum() == 20

		assert ml_advanced_view.test_size.minimum() == 0.1
		assert ml_advanced_view.test_size.maximum() == 0.5
		assert ml_advanced_view.test_size.decimals() == 2

		assert ml_advanced_view.random_state.minimum() == -1
		assert ml_advanced_view.random_state.maximum() == 999999

		assert ml_advanced_view.n_jobs.minimum() == -1
		assert ml_advanced_view.n_jobs.maximum() == 32

	def test_settings_change_signal(self, ml_advanced_view, qtbot):
		"""Test that settings changes emit the correct signal."""
		signal_received = False
		settings_received = None

		def handle_settings_change(settings):
			nonlocal signal_received, settings_received
			signal_received = True
			settings_received = settings

		ml_advanced_view.settings_changed.connect(handle_settings_change)

		# Get the test model group
		test_group = ml_advanced_view.model_groups["Test Model"]
		param_widgets = test_group.findChildren(ParamInputWidget)
		assert len(param_widgets) > 0

		first_param = param_widgets[0]
		if hasattr(first_param.single_value_widget, "setValue"):
			with qtbot.waitSignal(ml_advanced_view.settings_changed, timeout=1000):
				first_param.single_value_widget.setValue(2)

		assert signal_received
		assert isinstance(settings_received, dict)
		assert "Test Model" in settings_received
		assert isinstance(settings_received["Test Model"], dict)

	def test_get_current_settings(self, ml_advanced_view):
		"""Test retrieving current settings."""
		settings = ml_advanced_view.get_current_settings()
		assert isinstance(settings, dict)

	def test_tab_organization(self, ml_advanced_view):
		"""Test that models are organized in correct tabs."""
		# Check General Settings tab
		general_tab = ml_advanced_view.tab_widget.widget(0)
		assert general_tab is not None
		assert "General Settings" in ml_advanced_view.tab_widget.tabText(0)

		# Check Linear Models tab
		linear_tab = ml_advanced_view.tab_widget.widget(1)
		assert linear_tab is not None
		assert "Linear Models" in ml_advanced_view.tab_widget.tabText(1)

		# Check Tree-Based Models tab
		tree_tab = ml_advanced_view.tab_widget.widget(2)
		assert tree_tab is not None
		assert "Tree-Based Models" in ml_advanced_view.tab_widget.tabText(2)

		# Check Other Models tab
		other_tab = ml_advanced_view.tab_widget.widget(3)
		assert other_tab is not None
		assert "Other Models" in ml_advanced_view.tab_widget.tabText(3)

		# Check Test tab
		test_tab = ml_advanced_view.tab_widget.widget(4)
		assert test_tab is not None
		assert "Test" in ml_advanced_view.tab_widget.tabText(4)

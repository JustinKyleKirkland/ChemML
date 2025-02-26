import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Create a Qt Application that persists for the entire test session"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        # Disable effects and animations to reduce test flakiness
        app.setAttribute(Qt.AA_DisableHighDpiScaling)
        app.setStyle("fusion")
    yield app
    app.processEvents()


@pytest.fixture
def app(qapp, request):
    """Fixture that provides the Qt application instance to individual tests"""

    def cleanup():
        qapp.processEvents()

    request.addfinalizer(cleanup)
    return qapp


@pytest.fixture(autouse=True)
def cleanup_widgets(request, qapp):
    """Cleanup any widgets after each test"""
    widgets = []

    def track_widget(widget):
        widgets.append(widget)

    def cleanup():
        for widget in widgets:
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        qapp.processEvents()

    request.addfinalizer(cleanup)
    return track_widget

import pytest
from utils.exceptions import RiserMLException, ConfigurationError

def test_exception_inheritance():
    err = ConfigurationError("Test error")
    assert isinstance(err, RiserMLException)
    assert isinstance(err, Exception)
    assert str(err) == "Test error"
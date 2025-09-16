import os
import pytest

@pytest.fixture(autouse=True)
def change_test_dir(request):
    old_dir = os.getcwd()
    test_file_dir = os.path.dirname(os.path.abspath(request.fspath))
    os.chdir(test_file_dir)
    yield
    os.chdir(old_dir)
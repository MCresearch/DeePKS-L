import pytest
import torch

# Set a fixed random seed for reproducibility in tests
@pytest.fixture(autouse=True, scope="module") # module for each file
def set_seed():
    torch.manual_seed(12345678)
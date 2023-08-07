from networks.loading import *
from networks.loading2 import *

def test_load_yml():

    model_dir = "networks/test/test_model"
    model = ModelLoader(model_dir, load_state_dict=False, mode="yaml")

    assert model[1].layers[0].in_features == 2
    assert torch.equal(model[0].x_mean, torch.tensor([0.321, 0.42]))

    model = load_model(model_dir, load_state_dict=False, mode="yaml")

    assert model[1].layers[0].in_features == 2
    assert torch.equal(model[0].x_mean, torch.tensor([0.321, 0.42]))

    return

def test_load_json():

    model_dir = "networks/test/test_model"
    model = ModelLoader(model_dir, load_state_dict=False, mode="json")

    assert model[1].layers[0].in_features == 2
    assert torch.equal(model[0].x_mean, torch.tensor([0.321, 0.42]))

    model = load_model(model_dir, load_state_dict=False, mode="json")

    assert model[1].layers[0].in_features == 2
    assert torch.equal(model[0].x_mean, torch.tensor([0.321, 0.42]))

    return

def test_load_state_dict():
    model_dir = "networks/test/test_model"
    model: nn.Module = ModelLoader(model_dir, load_state_dict=True, mode="yaml")

    # model[1].layers[0].weight.data = torch.zeros_like(model[1].layers[0].weight) + 2.0
    # torch.save(model.state_dict(), f"{model_dir}/state_dict.pt")

    assert torch.equal(model[1].layers[0].weight, torch.zeros_like(model[1].layers[0].weight) + 2.0)

    return

if __name__ == "__main__":
    test_load_yml()
    test_load_json()
    test_load_state_dict()

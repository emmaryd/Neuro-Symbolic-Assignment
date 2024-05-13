from json import dumps
from pathlib import Path
import os
import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network

from utils import MNISTOperator, get_trainer

FILE_PATH = Path(__file__).resolve().parent


def save_networks(name_model, model, name_network, network):
    model.save_state(FILE_PATH / "snapshot" / f"{name_model}.pth")

    saved_networks_dir = "networks"
    if not os.path.exists(FILE_PATH / saved_networks_dir):
        os.makedirs(saved_networks_dir)
    torch.save(network, FILE_PATH / saved_networks_dir / f"{name_network}.pth")


def modulo(x):
    return x[0] % (x[1] + 1)


def modulus(n: int, dataset: str, seed=None):
    """Returns a dataset for binary subtraction"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="modulus" if n == 1 else "multi_modulus",
        operator=modulo,
        size=n,
        seed=seed,
    )


def main(N, batch_size, n_epochs, pretrained_mnist=None):
    name_model = f"modulus_{N}"

    train_set = modulus(N, "train")
    test_set = modulus(N, "test")

    if pretrained_mnist is None:
        name_network = "mnist_modulus"
        network = MNIST_Net()
    else:
        name_network = pretrained_mnist
        network = torch.load(FILE_PATH / "networks" / f"{name_network}.pth")

    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    model = Model("models/modulus.pl", [net])
    model.set_engine(ExactEngine(model), cache=True)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    trainer = get_trainer(model)

    if pretrained_mnist is None:
        trainer.train(loader, n_epochs, log_iter=100, profile=0)
        save_networks(name_model, model, name_network, network)

    trainer.logger.comment(dumps(model.get_hyperparameters()))
    trainer.logger.comment(
        "Accuracy {}".format(
            get_confusion_matrix(model, test_set, verbose=1).accuracy()
        )
    )
    trainer.logger.write_to_file(str(FILE_PATH / "log" / name_model / name_network))


if __name__ == "__main__":
    N = 2
    batch_size = 2
    n_epochs = 1

    if os.path.exists(FILE_PATH / "networks" / "mnist_modulus.pth"):
        pretrained_mnist = "mnist_modulus"
    else:
        pretrained_mnist = None

    main(
        N,
        batch_size,
        n_epochs,
        pretrained_mnist=pretrained_mnist,
    )

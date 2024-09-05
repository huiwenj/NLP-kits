import torch
from torch import nn

from app.models.classify import all_letters, RNN, device, train_one_epoch, load_labeled_file


class ClassifyService:
    def __init__(self):
        pass

    def classify(self, image):
        pass

    def train(self):
        word_train_data = load_labeled_file("data/complex_words_training.txt")

        test_model = RNN(input_size=len(all_letters), hidden_size=10).to(device)
        test_model.train()

        before = list(test_model.parameters())[-1].clone()
        output, loss, line, category = train_one_epoch(test_model, nn.NLLLoss(),
                                                     torch.optim.SGD(test_model.parameters(), lr=0.2),
                                                     word_train_data[0], word_train_data[1])
        after = list(test_model.parameters())[-1].clone()

        return [before.detach().cpu().numpy(), after.detach().cpu().numpy()]



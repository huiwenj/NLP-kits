import os

import numpy as np
import torch
from torch import nn

from app.models.classify import all_letters, RNN, device, load_labeled_file, run, calculateAccuracy, line_to_tensor, \
    read_data, run_multi, languages, RnnMulti, calculate_accuracy_multi, replace_nan_with_none
from app.models.response import R


# This is a service class that contains the logic for training and predicting the model
class WordClassifyService:
    def train(self):
        word_train_data = load_labeled_file("data/classify/word/complex_words_training.txt")
        word_val_data = load_labeled_file("data/classify/word/complex_words_development.txt")

        word_train_losses, word_val_losses, model_path = run(train_data=word_train_data,
                                                             val_data=word_val_data,
                                                             hidden_size=50,
                                                             n_epochs=10000,
                                                             learning_rate=0.005,
                                                             loss_func=nn.NLLLoss(),
                                                             print_every=5000,
                                                             plot_every=250,
                                                             model_name="word_RNN"
                                                             )

        test_model = RNN(input_size=len(all_letters), hidden_size=50).to(device)
        test_model.load_state_dict(torch.load(model_path))
        val_acc = calculateAccuracy(test_model, word_val_data[0], word_val_data[1])

        return {
            "train_losses": word_train_losses,
            "val_losses": word_val_losses,
            "val_acc": val_acc
        }

    def predict(self, word):
        """
        Predict the word is simple or hard
        Args:
            word: str

        Returns: 1 is complex, 0 is simple

        """
        if not os.path.exists("./.models/word_RNN"):
            return R.bad_request("Model not found, please train the model first")

        try:
            model = RNN(input_size=len(all_letters), hidden_size=50).to(device)
            model.load_state_dict(torch.load("./.models/word_RNN"))
            model.eval()
        except Exception as e:
            return R.error(f"Error when loading model: {str(e)}")

        try:
            with torch.no_grad():

                line_tensor = line_to_tensor(word)
                hidden = model.init_hidden().to(device)

                for i in range(line_tensor.size()[0]):
                    output, hidden = model(line_tensor[i], hidden)

                log_probabilities = output.cpu().data
                log_prob0, log_prob1 = log_probabilities[0]

                prediction = int(log_prob0 < log_prob1)
                return R.success(prediction)
        except Exception as e:
            return R.error(f"Error when predicting: {str(e)}")


class CityClassifyService:
    def train(self):
        city_train_data_raw = read_data("data/classify/city/", train=True)
        city_val_data_raw = read_data("data/classify/city/", train=False)

        X_city, y_city_str = city_train_data_raw
        X_val_city, y_val_city_str = city_val_data_raw

        y_city = np.array([languages.index(country) for country in y_city_str])
        y_val_city = np.array([languages.index(country) for country in y_val_city_str])

        city_train_data = X_city, y_city
        city_val_data = X_val_city, y_val_city

        city_all_losses, city_val_losses, model_path = run_multi(train_data=city_train_data,
                                                     val_data=city_val_data,
                                                     hidden_size=100,
                                                     n_epochs=5000,
                                                     learning_rate=0.002,
                                                     loss_func=nn.NLLLoss(),
                                                     print_every=500,
                                                     plot_every=100,
                                                     model_name="city_RNN"
                                                     )

        test_model_multi = RnnMulti(input_size=len(all_letters), hidden_size=100, output_size=len(languages)).to(device)
        test_model_multi.load_state_dict(torch.load(model_path))

        test_model_multi.eval()

        val_acc = calculate_accuracy_multi(test_model_multi, city_val_data_raw[0], y_val_city)

        return {
            "train_losses": replace_nan_with_none(city_all_losses),
            "val_losses": replace_nan_with_none(city_val_losses),
            "val_acc": val_acc
        }

    def predict(self, city):
        if not os.path.exists("./.models/city_RNN"):
            return R.bad_request("Model not found, please train the model first")


        try:
            model = RnnMulti(input_size=len(all_letters), hidden_size=100, output_size=len(languages)).to(
                device)
            model.load_state_dict(torch.load("./.models/city_RNN"))

        except Exception as e:
            return R.error(f"Error when loading model: {str(e)}")

        try:
            with torch.no_grad():
                model.eval()
                line_tensor = line_to_tensor(city)
                hidden = model.init_hidden().to(device)

                for i in range(line_tensor.size()[0]):
                    output, hidden = model(line_tensor[i], hidden)

                log_probabilities = output.cpu().data
                prediction = torch.argmax(log_probabilities).item()
                return R.success(languages[prediction])
        except Exception as e:
            return R.error(f"Error when predicting: {str(e)}")

    def list_city(self):
        try:
            with open("data/classify/city/cities_test.txt") as f:
                countries = f.readlines()
                countries = [country.strip() for country in countries]
                return R.success(countries)
        except Exception as e:
            return R.error(f"Error when getting city list, {str(e)}")






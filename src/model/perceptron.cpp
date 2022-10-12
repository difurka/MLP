#include <sstream>
#include <fstream>
#include "perceptron.h"

namespace s21 {

Perceptron::Perceptron() {
}

Perceptron::Perceptron(int number_of_hidden_layers, TypeOfPerceptron type)
  : number_of_hidden_layers_(number_of_hidden_layers), type_of_perceptron_(type) {
  LayerSizeInitialize(number_of_hidden_layers);
}

Perceptron::~Perceptron() {
}

void Perceptron::LayerSizeInitialize(int number_of_hidden_layers) {
    if (number_of_hidden_layers == 2) {
    SetLayersSize({kInput, kHidden1, kHidden2, kOutput});
  } else if (number_of_hidden_layers == 3) {
    SetLayersSize({kInput, kHidden1, kHidden2, kHidden3, kOutput});
  } else if (number_of_hidden_layers == 4) {
    SetLayersSize({kInput, kHidden1, kHidden2, kHidden3, kHidden4, kOutput});
  } else if (number_of_hidden_layers == 5) {
    SetLayersSize({kInput, kHidden1, kHidden2, kHidden3, kHidden4, kHidden5, kOutput});
  }
}

std::vector<double>& Perceptron::GetResultCosts() {
  return result_costs_;
}

const TypeOfPerceptron& Perceptron::GetTypeOfPerceptron() const {
  return type_of_perceptron_;
}

int Perceptron::GetNumberOfHiddenLayers() {
  return number_of_hidden_layers_;
}

void Perceptron::CleanResultCosts() {
  result_costs_ = {};
}

void Perceptron::SetLayersSize(const std::vector<int> &count_of_neurons_in_layers) {
  layers_size_ = count_of_neurons_in_layers;
}

std::vector<int>& Perceptron::GetLayersSize() {
  return layers_size_;
}

}  // namespace s21

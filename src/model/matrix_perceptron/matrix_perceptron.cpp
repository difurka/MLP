#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include "matrix_perceptron.h"

namespace s21 {

/* -------------------------------------------------------------------------- */
/*                            layer                                           */
/* -------------------------------------------------------------------------- */

MatrixPerceptron::Layer::Layer(int count_of_neurons, int count_of_inputs)
    : count_of_neurons_(count_of_neurons), count_of_inputs_(count_of_inputs),
    weights_(Matrix(count_of_neurons, count_of_inputs)),
    biases_(Matrix(count_of_neurons, 1)),
    outs_(Matrix(count_of_neurons, 1)),
    delta_weights_(Matrix(count_of_neurons, count_of_inputs)),
    delta_biases_(Matrix(count_of_neurons, 1)) {}

  void MatrixPerceptron::Layer::SetLayerWeights(const Matrix& weights) {
    weights_ = weights;
  }

  void MatrixPerceptron::Layer::SetLayerBiases(const Matrix& biases) {
    biases_ = biases;
  }

  void MatrixPerceptron::Layer::SetDeltaWeights(const Matrix& delta_weights) {
    delta_weights_ = delta_weights;
  }

  void MatrixPerceptron::Layer::SetDeltaBiases(const Matrix& delta_biases) {
    delta_biases_ = delta_biases;
  }

  void MatrixPerceptron::Layer::SetOuts(const Matrix& outs) {
    outs_ = outs;
  }

  Matrix& MatrixPerceptron::Layer::GetLayerWeights() {
    return weights_;
  }

  Matrix& MatrixPerceptron::Layer::GetLayerBiases() {
    return biases_;
  }

  Matrix& MatrixPerceptron::Layer::GetDeltaWeights() {
    return delta_weights_;
  }

  Matrix& MatrixPerceptron::Layer::GetDeltaBiases() {
    return delta_biases_;
  }

  Matrix& MatrixPerceptron::Layer::GetLayerOuts() {
    return outs_;
  }

  int MatrixPerceptron::Layer::GetCountOfInputs() const {
    return count_of_inputs_;
  }

  int MatrixPerceptron::Layer::GetCountOfNeurons() const {
    return count_of_neurons_;
  }

  void MatrixPerceptron::Layer::CalculateLinearSumma(const Matrix& inputs) {
    linear_summa_ = weights_ * inputs + biases_;
  }
  void MatrixPerceptron::Layer::CalculateLayerOuts(const Matrix& inputs) {
    CalculateLinearSumma(inputs);
    for (int i = 0; i < count_of_neurons_; ++i) {
      outs_(i, 0) = (1.0 / (1.0 + exp(-linear_summa_(i, 0))));
    }
  }

/* -------------------------------------------------------------------------- */
/*                            network                                         */
/* -------------------------------------------------------------------------- */

MatrixPerceptron::MatrixPerceptron(int number_of_hidden_layers)
  : Perceptron(number_of_hidden_layers, kMatrix) {
    SetNeuronsInLayers();
}

MatrixPerceptron::MatrixPerceptron(const std::vector<int>& count_of_neurons_in_layers) {
  SetLayersSize(count_of_neurons_in_layers);
  SetNeuronsInLayers();
}

MatrixPerceptron::MatrixPerceptron(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases) {
  CreatePerceptronFromMatrix(weights, biases);
}

MatrixPerceptron::~MatrixPerceptron() {}

void MatrixPerceptron::CreatePerceptronFromMatrix(const std::vector<Matrix>& weights,
                                            const std::vector<Matrix>& biases) {
  if (weights.size() != biases.size()) throw std::out_of_range("invalid size of weights and biases");
  int number_of_last_ = weights.size();
  layers_size_.resize(number_of_last_ + 1);
  for (int i = 0; i < number_of_last_; ++i)
    layers_size_[i] = weights[i].GetColumns();
  layers_size_[number_of_last_] = weights[number_of_last_ - 1].GetRows();
  layers_.resize(number_of_last_+ 1);
  layers_[0] = Layer(layers_size_[0], 1);
  for (int i = 1; i < number_of_last_ + 1; ++i) {
    layers_[i] = Layer(layers_size_[i], layers_size_[i - 1]);
    layers_[i].SetLayerWeights(weights[i - 1]);
    layers_[i].SetLayerBiases(biases[i - 1]);
  }
}

void MatrixPerceptron::SetNeuronsInLayers() {
  layers_.resize(layers_size_.size());
  layers_[0] = Layer(layers_size_[0], 1);
  for (int i = 1; i < CountOfLayers(); ++i) {
    int rows = layers_size_[i];
    int cols = layers_size_[i - 1];
    layers_[i] = Layer(rows, cols);
  }
}

void MatrixPerceptron::SetRandomWeights() {
    for (int i = 1; i < CountOfLayers(); ++i) {
    int rows = layers_size_[i];
    int cols = layers_size_[i - 1];
    Matrix random_weights(rows, cols);
    Matrix random_biases(rows, 1);
    random_weights.Random(-kRandomWeights, kRandomWeights);
    random_biases.Random(-kRandomWeights, kRandomWeights);
    layers_[i].SetLayerWeights(random_weights);
    layers_[i].SetLayerBiases(random_biases);
  }
}

std::vector<double> MatrixPerceptron::GetPerceptronOuts() {
  int ouput_layer_size = GetLayersSize()[GetLayersSize().size() - 1];
  std::vector<double> vector_of_answer(ouput_layer_size);
  for (int i = 0; i < ouput_layer_size; ++i)
    vector_of_answer[i] = layers_[CountOfLayers() - 1].GetLayerOuts()(i, 0);
  return vector_of_answer;
}

int MatrixPerceptron::CountOfLayers() {
  return layers_.size();
}

void MatrixPerceptron::CalculatePerceptronOuts(const Matrix& input) {
  layers_[0].SetOuts(input);
  for (int i = 1; i < CountOfLayers(); ++i) {
    Matrix temp_input = layers_[i - 1].GetLayerOuts();
    layers_[i].CalculateLayerOuts(temp_input);
  }
}

void MatrixPerceptron::CalculateResult(const std::vector<double>& sample) {
  int input_layer_size = GetLayersSize()[0];
  Matrix matr_of_sample(input_layer_size, 1);
  matr_of_sample.VectorToMatrix(sample, input_layer_size, 1);
  CalculatePerceptronOuts(matr_of_sample);
}

/* -------------------------------------------------------------------------- */
/*                            training                                        */
/* -------------------------------------------------------------------------- */

void MatrixPerceptron::MakeTraining(const SamplesDataset& sample) {
  CalculatePerceptronOuts(sample.GetMatrixOfSample());
  CalculateDeltaWeigtsForLastLayer(sample);
  for (int i = 1; i <  CountOfLayers() - 1; ++i) {
    CalculateDeltaWeigtsForOtherLayers(CountOfLayers() - i - 1);
  }
  for (int i = 1; i < CountOfLayers(); ++i) {
    s21::Matrix weights_new = layers_[i].GetLayerWeights() - layers_[i].GetDeltaWeights();
    s21::Matrix biases_new = layers_[i].GetLayerBiases() - layers_[i].GetDeltaBiases();
    layers_[i].SetLayerWeights(weights_new);
    layers_[i].SetLayerBiases(biases_new);
  }
}

void MatrixPerceptron:: CalculateCostFunction(const SamplesDataset& sample) {
  CalculatePerceptronOuts(sample.GetMatrixOfSample());
  double cost = 0;
  for (int i = 0; i < sample.GetCntOfOutputs(); ++i) {
    double result = layers_[CountOfLayers() - 1].GetLayerOuts()(i, 0);
    double expected = (i == sample.GetIndexOfAnswerLetter()) ? 1 : 0;
    cost +=  pow(result - expected, 2);
  }
  result_costs_.push_back(cost);
}

void MatrixPerceptron::CalculateDeltaWeigtsForLastLayer(const SamplesDataset& sample) {
  Layer layer = layers_[CountOfLayers() - 1];
  Matrix delta_weights_temp(layer.GetLayerWeights());
  Matrix delta_biases_temp(layer.GetLayerBiases());
  for (int i = 0; i < layer.GetCountOfNeurons(); ++i) {
    double output_of_neuron = layer.GetLayerOuts()(i, 0);
    double x_expected = (i == sample.GetIndexOfAnswerLetter()) ? 1 : 0;
    double derivative_of_sigmoida = output_of_neuron * (1 - output_of_neuron);
    double common_mult_for_all_neurons = 2 * (output_of_neuron - x_expected) * derivative_of_sigmoida;
    delta_biases_temp(i, 0) = common_mult_for_all_neurons;
    for (int j = 0; j < layer.GetCountOfInputs(); ++j) {
      delta_weights_temp(i, j) =
        common_mult_for_all_neurons * layers_[CountOfLayers() - 2].GetLayerOuts()(j, 0);
    }
  }
  Matrix delta_weights_new =
    kLearningKoef * delta_weights_temp - kSmoothingKoef * layers_[CountOfLayers() - 1].GetDeltaWeights();
  Matrix delta_biases_new =
    kLearningKoef * delta_biases_temp - kSmoothingKoef * layers_[CountOfLayers() - 1].GetDeltaBiases();
  layers_[CountOfLayers() - 1].SetDeltaWeights(delta_weights_new);
  layers_[CountOfLayers() - 1].SetDeltaBiases(delta_biases_new);
}

void MatrixPerceptron::CalculateDeltaWeigtsForOtherLayers(int number_of_layer) {
  Layer layer = layers_[number_of_layer];
  Layer layer_next = layers_[number_of_layer + 1];
  Matrix delta_weights_temp(layer.GetLayerWeights().GetRows(), layer.GetLayerWeights().GetColumns());
  Matrix delta_biases_temp(layer.GetLayerBiases().GetRows(), 1);
  Matrix next_delta_biases = layer_next.GetDeltaBiases();
  Matrix next_weights = layer_next.GetLayerWeights();
  Matrix past_delta = next_weights.transpose() * next_delta_biases;
  for (int i = 0; i < layer.GetCountOfNeurons(); ++i) {
    double output_of_neuron = layer.GetLayerOuts()(i, 0);
    double derivative_of_sigmoida = output_of_neuron * (1 - output_of_neuron);
    delta_biases_temp(i, 0) = derivative_of_sigmoida * past_delta(i, 0);
    for (int j = 0; j < layer.GetCountOfInputs(); ++j) {
      delta_weights_temp(i, j) = delta_biases_temp(i, 0) * layers_[number_of_layer - 1].GetLayerOuts()(j, 0);
    }
  }
  Matrix delta_weights_new =
    kLearningKoef * delta_weights_temp - kSmoothingKoef * layers_[number_of_layer].GetDeltaWeights();
  Matrix delta_biases_new =
    kLearningKoef * delta_biases_temp - kSmoothingKoef * layers_[number_of_layer].GetDeltaBiases();

  layers_[number_of_layer].SetDeltaWeights(delta_weights_new);
  layers_[number_of_layer].SetDeltaBiases(delta_biases_new);
}

/* -------------------------------------------------------------------------- */
/*                            save and load weights                            */
/* -------------------------------------------------------------------------- */

void MatrixPerceptron::LoadWeights(const std::string& file_name) {
  std::ifstream file_stream;
  file_stream.open(file_name);
  if (!file_stream.good()) throw std::invalid_argument("invalid file " + file_name);
  LoadLayersInfo(file_stream);
  SetLayersSize(layers_size_);
  SetNeuronsInLayers();
  int cnt_of_matrix = CountOfLayers() - 1;
  std::vector<Matrix> weights(cnt_of_matrix);
  std::vector<Matrix> biases(cnt_of_matrix);
  for (int i = 0; i < cnt_of_matrix; ++i) {
    weights[i] = LoadLayersWeights(file_stream, layers_size_[i + 1], layers_size_[i]);
  }
  for (int i = 0; i < cnt_of_matrix; ++i) {
    biases[i] = LoadLayersWeights(file_stream, layers_size_[i + 1], 1);
  }
  file_stream.close();
  CreatePerceptronFromMatrix(weights, biases);
}

void MatrixPerceptron::LoadLayersInfo(std::ifstream &file_stream) {
  layers_size_.clear();
  std::string line;
  std::getline(file_stream, line);
  std::stringstream stream(line);
  size_t input, position = 0;
  while ((position = line.find(" ")) != std::string::npos) {
    stream >> input;
    if (input > 0) layers_size_.push_back(input);
    line.erase(0, position + 1);
  }
}

Matrix MatrixPerceptron::LoadLayersWeights(std::ifstream &file_stream, int rows, int cols) {
  Matrix result(rows, cols);
  std::string line;
  std::getline(file_stream, line);
  std::vector<double> array(rows * cols);
  int j = 0;
  size_t position = 0;
  std::stringstream stream(line);
  while ((position = line.find(" ")) != std::string::npos && j < rows * cols) {
    std::string token = line.substr(0, position);
    if (token.size() > 0) {
      array[j] = std::stof(token);
      ++j;
    }
    line.erase(0, position + 1);
  }
  if (j != rows * cols && j != 0) throw std::out_of_range("invalid size of weight matrix");
  if (j != 0) {
    result.VectorToMatrix(array, cols, rows);
  } else {
    result = Matrix(1, rows);
  }
  return result.transpose();
}

void MatrixPerceptron::SaveWeights(const std::string& file_name) {
  std::ofstream out;
  out.open(file_name, std::ios::trunc);
  int cnt_of_layers = CountOfLayers();
  for (int i = 0; i < cnt_of_layers - 1; ++i) {
    out << layers_size_[i] << " ";
  }
    for (int i = cnt_of_layers; i < 7; ++i) {
    out << 0 << " ";
  }
  out << layers_size_[cnt_of_layers - 1] << " ";
  out << "\n";
  for (int i = 1; i < cnt_of_layers; ++i) {
    SaveLayerWeights(out, layers_[i].GetLayerWeights());
  }
    for (int i = 1; i < cnt_of_layers; ++i) {
    SaveLayerWeights(out, layers_[i].GetLayerBiases());
  }
  out.close();
}

void MatrixPerceptron::SaveLayerWeights(std::ofstream& out, const Matrix& layer) {
  for (int j = 0; j < layer.GetColumns(); ++j)
    for (int i = 0; i < layer.GetRows(); ++i)
      out << layer(i, j) << " ";
  out << "\n";
}
}   // namespace s21

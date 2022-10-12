#include "graph_perceptron.h"
#include <sstream>
#include <random>
#include <fstream>
#include <cmath>

namespace s21 {
/* -------------------------------------------------------------------------- */
/*                                   Neuron                                   */
/* -------------------------------------------------------------------------- */
GraphPerceptron::Neuron::Neuron(std::vector<Neuron> *next_layer)
  : value_(0), error_(0), next_layer_(next_layer) {
}

GraphPerceptron::Neuron::~Neuron() {}

void GraphPerceptron::Neuron::AddedRandomWeight(int next_layer_size) {
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_int_distribution<int> uniform_dist(-kRandomWeights, kRandomWeights);
  for (int i = 0; i < next_layer_size; ++i) {
    weight_.push_back((double)(uniform_dist(engine) / 100.0));
    weight_delta_.push_back(0.0);
  }
}

void GraphPerceptron::Neuron::SetValue(double value) {
  value_ = value;
}

void GraphPerceptron::Neuron::InputSignal(double input_value) {
  value_ += input_value;
}

void GraphPerceptron::Neuron::OutputSignal() {
  int i = 0;
  for (Neuron &neuron : *next_layer_) {
    double weight = weight_[i++];
    double output_signal = value_ * weight;
    neuron.InputSignal(output_signal);
  }
}

void GraphPerceptron::Neuron::Activation() {
  value_ = 1.0 / (1.0 + exp(-value_));
}

double& GraphPerceptron::Neuron::GetValue() {
  return value_;
}

std::vector<double>* GraphPerceptron::Neuron::GetWeights() {
  return &weight_;
}

int GraphPerceptron::Neuron::GetNextLayerSize() {
  return next_layer_->size();
}

double& GraphPerceptron::Neuron::GetError() {
  return error_;
}

void GraphPerceptron::Neuron::SetError(double error) {
  error_ = error;
}

double& GraphPerceptron::Neuron::GetDelta(int index) {
  return weight_delta_[index];
}

void GraphPerceptron::Neuron::SetDelta(int index, double delta) {
  weight_delta_[index] = delta;
}

void GraphPerceptron::Neuron::UpdateWeights() {
  for (size_t i = 0; i < weight_.size(); ++i) {
    weight_[i] += weight_delta_[i];
  }
}

/* -------------------------------------------------------------------------- */
/*                                   Network                                  */
/* -------------------------------------------------------------------------- */
GraphPerceptron::GraphPerceptron(int number_of_hidden_layers) :
  Perceptron(number_of_hidden_layers, kGraph) {
  InitializePerceptron();
}

GraphPerceptron::~GraphPerceptron() { }

void GraphPerceptron::InitializePerceptron() {
  for (int l = 0; l < number_of_hidden_layers_ + 2; ++l) {
    layers_.push_back(std::vector<Neuron>());
  }
  CreateInputLayer();
  CreateHiddenOutputLayers();
}

void GraphPerceptron::CreateInputLayer() {
  for (int i = 0; i < layers_size_[0]; ++i) {
    layers_[0].push_back(Neuron(&layers_[1]));
  }
}

void GraphPerceptron::CreateHiddenOutputLayers() {
  int total_layers = layers_size_.size() - 1;
  for (int i = 1; i <= total_layers; ++i) {
    if (i == total_layers) {
      CreateOutputLayer(&layers_[i], layers_size_[i]);
    } else if (i <= number_of_hidden_layers_) {
      CreateLayer(&layers_[i], &layers_[i+1], layers_size_[i]);
    }
  }
}

void GraphPerceptron::CreateOutputLayer(std::vector<Neuron>* layer, const int size) {
  for (int i = 0; i < size; ++i) {
    layer->push_back(Neuron());
  }
}

void GraphPerceptron::CreateLayer(std::vector<Neuron>* layer,
                                  std::vector<Neuron>* next_layer, const int size) {
  for (int i = 0; i < size; ++i) {
    layer->push_back(Neuron(next_layer));
  }
}

void GraphPerceptron::SetRandomWeights() {
  for (int l = 0; l < number_of_hidden_layers_ + 1; ++l) {
    for (int n = 0; n < layers_size_[l]; ++n) {
      layers_[l][n].AddedRandomWeight(layers_[l][n].GetNextLayerSize());
    }
  }
}

void GraphPerceptron::ForwardPropagation() {
  const int total_layer = layers_.size();
  for (int l = 0; l < total_layer; ++l) {
    if (l == 0) {
      MakeSignal(&layers_[l], LayerType::kInputLayer);
    } else if (l == total_layer - 1) {
      MakeSignal(&layers_[l], LayerType::kOutputLayer);
    } else {
      MakeSignal(&layers_[l], LayerType::kHiddenLayer);
    }
  }
}

void GraphPerceptron::MakeSignal(std::vector<Neuron>* layer, LayerType layer_type) {
  if (layer_type == LayerType::kInputLayer) {
    for (auto &neuron : *layer) {
      neuron.OutputSignal();
    }
  } else if (layer_type == LayerType::kHiddenLayer) {
    for (auto &neuron : *layer) {
      neuron.Activation();
      neuron.OutputSignal();
    }
  } else if (layer_type == LayerType::kOutputLayer) {
    for (auto &neuron : *layer) {
      neuron.Activation();
    }
  }
}

void GraphPerceptron::CalculateResult(const std::vector<double>& sample) {
  for (int i = 0; i < layers_size_[0]; ++i) {
    layers_[0][i].SetValue(sample[i]);
  }
  ForwardPropagation();
}

/* -------------------------------------------------------------------------- */
/*                                  learning                                  */
/* -------------------------------------------------------------------------- */

void GraphPerceptron::MakeTraining(const SamplesDataset& sample) {
  CalculateResult(sample.GetSample());
  BackPropagation(sample.GetIndexOfAnswerLetter());
}

void GraphPerceptron::BackPropagation(int answer_index) {
  CalculateOutputLayerErrors(answer_index);
  CalculateHiddenInputLayerErrors();
  CalculateDeltaWeight();
  CorrectAllLayerWeights();
}

void GraphPerceptron::CalculateOutputLayerErrors(int answer_index) {
  for (int n = 0; n < kOutput; ++n) {
    double correct = 0;
    Neuron& neuron = layers_[number_of_hidden_layers_ + 1][n];
    double value = neuron.GetValue();
    if (n == answer_index) correct = 1;
    double error = (correct - value) * (value * (1.0 - value));
    neuron.SetError(error);
  }
}

void GraphPerceptron::CalculateHiddenInputLayerErrors() {
  int output_layers = layers_.size() - 1;
  int last_hidden_layer = output_layers - 1;
  int previous_layer = output_layers;

  for (int l = last_hidden_layer; l > 0; --l) {
    for (size_t n = 0; n < layers_[l].size(); ++n) {
      double total_error = 0.0;
      Neuron &current_neuron  = layers_[l][n];
      for (size_t nl = 0; nl < layers_[previous_layer].size(); ++nl) {
        Neuron &neuron = layers_[previous_layer][nl];
        total_error += neuron.GetError() * (*current_neuron.GetWeights())[nl];
      }
      double value = current_neuron.GetValue();
      double error = total_error * value * (1 - value);
      current_neuron.SetError(error);
    }
    --previous_layer;
  }
}

void GraphPerceptron::CalculateDeltaWeight() {
  for (int l = number_of_hidden_layers_; l >= 0; --l) {
    for (size_t n = 0; n < layers_[l].size(); ++n) {
      Neuron &neuron = layers_[l][n];

      for (size_t nl = 0; nl < layers_[l+1].size(); nl++) {
        Neuron &next_neuron = layers_[l+1][nl];
        double error = next_neuron.GetError();
        double value = neuron.GetValue();
        double delta = kLearningKoef * error * value + (0.1 * neuron.GetDelta(nl));
        neuron.SetDelta(nl, delta);
      }
    }
  }
}

void GraphPerceptron::CorrectAllLayerWeights() {
  for (int l = number_of_hidden_layers_; l >= 0; --l) {
    for (Neuron &neuron : layers_[l]) {
      neuron.UpdateWeights();
    }
  }
}

void GraphPerceptron::CalculateCostFunction(const SamplesDataset& sample) {
  double cost = 0.0;
  int answer_number = sample.GetIndexOfAnswerLetter();
  int index = layers_size_.size() - 1;
  auto &output_layer = layers_[index];

  for (int i = 0; i < layers_size_[index]; ++i) {
    double expected = (i == answer_number) ? 1.0 : 0.0;
    double value = output_layer[i].GetValue();
    cost += pow((value - expected), 2.0);
  }
  result_costs_.push_back(cost);
}

std::vector<double> GraphPerceptron::GetPerceptronOuts() {
  std::vector<double> result;
  int last_layer = number_of_hidden_layers_ + 1;
  for (size_t i = 0; i < layers_[last_layer].size(); ++i) {
    result.push_back(layers_[last_layer][i].GetValue());
  }
  return result;
}

/* -------------------------------------------------------------------------- */
/*                                 Load weights                                */
/* -------------------------------------------------------------------------- */

void GraphPerceptron::LoadWeights(const std::string& file_name) {
  std::ifstream file_stream;
  file_stream.open(file_name);
  if (!file_stream.is_open()) throw std::invalid_argument("file error");
  std::string line;
  std::getline(file_stream, line);
  LoadPerceptron(line);
  for (int l = 0; l <= number_of_hidden_layers_; ++l) {
    LoadLayerWeights(file_stream, &layers_[l]);
  }
}

void GraphPerceptron::LoadPerceptron(const std::string& line) {
  std::stringstream stream(line);
  int input = 0, hide_I = 0, hide_II = 0, hide_III = 0, hide_IV = 0, hide_V = 0, output = 0;
  stream >> input >> hide_I >> hide_II >> hide_III >> hide_IV >> hide_V >> output;
  if (kInput != input ||
      kOutput != output ||
      kHidden1 != hide_I ||
      kHidden2 != hide_II ||
      (kHidden3 != hide_III && hide_III != 0) ||
      (kHidden4 != hide_IV && hide_IV != 0) ||
      (kHidden5 != hide_V && hide_V != 0))
        throw std::out_of_range("invalid network size in file");

  int hidden_layers = 2;
  if (hide_III == kHidden3) ++hidden_layers;
  if (hide_IV == kHidden4) ++hidden_layers;
  if (hide_V == kHidden5) ++hidden_layers;
  ChangeLayers(hidden_layers);
}

void GraphPerceptron::ChangeLayers(int hidden_layers) {
  if (number_of_hidden_layers_ != hidden_layers) {
    layers_.clear();
    number_of_hidden_layers_ = hidden_layers;
    layers_size_.clear();
    LayerSizeInitialize(number_of_hidden_layers_);
    InitializePerceptron();
  }
}

void GraphPerceptron::LoadLayerWeights(std::ifstream &file_stream, std::vector<Neuron>* layer) {
  if (layer->size() > 0) {
    std::string line;
    std::getline(file_stream, line);
    WeightsFromFileToNeuronImport(&line, layer);
  }
}

void GraphPerceptron::WeightsFromFileToNeuronImport(std::string *line, std::vector<Neuron>* layer) {
  size_t position = 0;
  int neuron_link = 0;
  int neuron = 0;
  int total_link = layer->size() * (*layer)[0].GetNextLayerSize();
  std::vector<double>* neuron_weights = (*layer)[neuron].GetWeights();
  neuron_weights->clear();
  while ((position = line->find(" ")) != std::string::npos) {
    int next_layer_size = (*layer)[0].GetNextLayerSize();
    std::string token = line->substr(0, position);
    if (token.size() > 0) {
      if (neuron_link == next_layer_size) {
        neuron_weights = (*layer)[++neuron].GetWeights();
        neuron_weights->clear();
        neuron_link = 0;
      }
      neuron_weights->push_back(std::stod(token));
      ++neuron_link;
      --total_link;
    }
    line->erase(0, position + 1);
  }
  if (total_link != 0) throw std::invalid_argument("invalid weight count from file");
}

/* -------------------------------------------------------------------------- */
/*                                Save weights                                */
/* -------------------------------------------------------------------------- */

void GraphPerceptron::SaveWeights(const std::string& file_name) {
  std::ofstream out;
  out.open(file_name, std::ios::trunc);
  if (!out.is_open()) throw std::invalid_argument("file error");
  WriteLayersSize(out);

  for (size_t l = 0; l < layers_.size() - 1; ++l) {
    SaveLayerWeights(out, layers_[l]);
  }
}

void GraphPerceptron::WriteLayersSize(std::ofstream& out) {
  for (int i = 0; i < number_of_hidden_layers_  + 1; ++i) {
    out << layers_size_[i] << " ";
  }
  int zero_layers = 5 - number_of_hidden_layers_;
  while (zero_layers > 0) {
    out << 0 << " ";
    --zero_layers;
  }
  out << layers_size_[number_of_hidden_layers_ + 1] << " ";
  out << "\n";
}

void GraphPerceptron::SaveLayerWeights(std::ofstream& out, const std::vector<Neuron>& layer) {
  if (layer.size() > 0) {
    size_t n = 0;
    for (; n < layer.size(); ++n) {
      auto neuron = layer[n];
      for (auto weight : *neuron.GetWeights()) {
        out << weight << " ";
      }
    }
    out << "\n";
  }
}

}  // namespace s21


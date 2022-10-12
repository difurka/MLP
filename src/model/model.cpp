#include <algorithm>
#include <ctime>
#include <fstream>
#include <sstream>
#include "model.h"

namespace s21 {

Model::Model() : perceptron_(nullptr)  {}

Model::Model(Perceptron* perceptron) : perceptron_(perceptron) {}

void Model::CreateNewPerceptron(TypeOfPerceptron type, int number_of_hidden_layers) {
  if (type == kMatrix) {
    if (perceptron_) {
      delete perceptron_;
    }
    perceptron_ = new MatrixPerceptron(number_of_hidden_layers);
  } else if (type == kGraph) {
    if (perceptron_) {
      delete perceptron_;
    }
    perceptron_ = new GraphPerceptron(number_of_hidden_layers);
  }
}

Model::~Model() {
  if (perceptron_) {
    delete perceptron_;
  }
}

void Model::MakeTesting(double sample_part) {
  if (perceptron_->GetResultCosts().size() > 0) {
    perceptron_->CleanResultCosts();
  }
  std::vector<int> result_true(0);
  std::vector<int> result_get(0);
  // проверка на букве а, её номер 1
  int a = 1;

  int size_of_testing_vector = (int) (GetSamplesDataset().size() * sample_part);

  clock_t t;
  t = clock();
  for (int i = 0; i < size_of_testing_vector; ++i) {
    CalculatePrediction(GetSamplesDataset()[i].GetSample());
    result_true.push_back((GetSamplesDataset()[i].GetIndexOfAnswerLetter() != a) ? 0 : 1);
    result_get.push_back((GetResultOfNet() != a) ? 0 : 1);
  }
  t = clock() - t;
  measurements_.time_spend_ = ((double)t)/CLOCKS_PER_SEC;

  double TP = 0;  // сколько раз буква а правильно определилась как а
  double TN = 0;  // сколько раз другие буквы правильно определилась как не а
  double FN = 0;  // сколько раз буква а определилась как не а
  double FP = 0;  // сколько раз буква не а определилась как а


  for (int i = 0; i < size_of_testing_vector; ++i) {
    TP += ((result_true[i] == result_get[i] && result_true[i] == 1) ? 1 : 0);
    TN += ((result_true[i] == result_get[i] && result_true[i] == 0) ? 1 : 0);
    FN += ((result_true[i] != result_get[i] && result_true[i] == 1) ? 1 : 0);
    FP += ((result_true[i] != result_get[i] && result_true[i] == 0) ? 1 : 0);
  }
  measurements_.average_accuracy_ = (TP + TN) / size_of_testing_vector;
  measurements_.precision_ = (TP + FP > 0) ? TP / (TP + FP) : 0;
  measurements_.recall_ = TP / (TP + FN);
  measurements_.f_measure_ = 2 * measurements_.precision_ * measurements_.recall_
                            / (measurements_.precision_ + measurements_.recall_);
}

Measurements& Model::GetMeasurements() {
  return measurements_;
}

void Model::LoadWeightsForModel(std::string file_name, const TypeOfPerceptron& type) {
  if (perceptron_->GetTypeOfPerceptron() != type) {
    CreateNewPerceptron(type, 2);
  }
  perceptron_->LoadWeights(file_name);
}

void Model::SaveWeightsForModel(std::string file_name) {
  perceptron_->SaveWeights(file_name);
}

void Model::CalculatePrediction(const std::vector<double>& sample) {
  perceptron_->CalculateResult(sample);
}

int Model::GetResultOfNet() {
  std::vector<double> answer = perceptron_->GetPerceptronOuts();
  return std::max_element(answer.begin(), answer.end()) - answer.begin();
}

void Model::StartLearningNet(int number_of_epochs, int number_of_groups) {
  if (perceptron_->GetResultCosts().size() > 0) {
    perceptron_->CleanResultCosts();
  }
  perceptron_->SetRandomWeights();
  measurements_.error_data_ = std::vector<double>(number_of_epochs);
  for (int i = 0; i < number_of_epochs; ++i) {
    perceptron_->CleanResultCosts();
    StartLearningNetOneEpoch(number_of_groups);
    double sum = 0;
    std::vector<double> vector_of_costs = perceptron_->GetResultCosts();
    for (size_t k = 0; k < vector_of_costs.size(); ++k) {
      sum += vector_of_costs[k];
    }
    measurements_.error_data_[i] = sum / vector_of_costs.size();
  }
}

void Model::StartLearningNetOneEpoch(int num_of_gr) {
  int size_of_train_v = GetSamplesDataset().size();
  if (num_of_gr > 1) {
    int num_sam_in_one_gr = (int)(size_of_train_v / num_of_gr);
    for (int num_of_test_gr = 0; num_of_test_gr < num_of_gr; ++num_of_test_gr) {
      for (int i = 0; i < num_of_gr; ++i) {
        if (i != num_of_test_gr) {
          for (int j = 0; j < num_sam_in_one_gr; ++j) {
            perceptron_->MakeTraining(GetSamplesDataset()[i * num_sam_in_one_gr + j]);
          }
        }
      }
      for (int j = 0; j < num_sam_in_one_gr; ++j) {
        perceptron_->CalculateCostFunction(GetSamplesDataset()[num_of_test_gr * num_sam_in_one_gr + j]);
      }
    }
  } else {
    for (int j = 0; j < size_of_train_v; ++j) {
      perceptron_->MakeTraining(GetSamplesDataset()[j]);
      perceptron_->CalculateCostFunction(GetSamplesDataset()[j]);
    }
  }
}

void Model::SetSamplesDatasetFromFile(const std::string& file_of_samples_dataset) {
  set_of_samples_.clear();
  std::ifstream file_stream;
  file_stream.open(file_of_samples_dataset);
  if (!file_stream.good()) throw std::invalid_argument("invalid file for samples" + file_of_samples_dataset);
  std::string line;
  int input_layer_size = perceptron_->GetLayersSize()[0];
  int ouput_layer_size = perceptron_->GetLayersSize()[perceptron_->GetLayersSize().size() - 1];
  SamplesDataset sample(input_layer_size , ouput_layer_size);
  while (file_stream.good()) {
    std::getline(file_stream, line);
    size_t position = 0;
    std::stringstream stream(line);
    if ((position = line.find(",")) != std::string::npos) {
      std::string token = line.substr(0, position);
      if (token.size() > 0) {
        line.erase(0, position + 1);
        sample.SetTrueLetter(std::stod(token));
        sample.SetSample(&line);
        set_of_samples_.push_back(sample);
      }
    }
  }
}

const std::vector<SamplesDataset>& Model::GetSamplesDataset() {
  return set_of_samples_;
}

const TypeOfPerceptron& Model::GetPerceptronType() const {
  return perceptron_->GetTypeOfPerceptron();
}

int Model::GetCountOfHiddenLayer() {
  return perceptron_->GetNumberOfHiddenLayers();
}

}  // namespace s21

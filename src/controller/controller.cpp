#include "controller.h"

namespace s21 {

Controller::Controller() : model_(nullptr) {}

void Controller::CreatePerceptron(TypeOfPerceptron type, int number_of_hidden_layers) {
  model_->CreateNewPerceptron(type, number_of_hidden_layers);
}

Controller::Controller(Model *model) : model_(model) {}

void Controller::MakeRecognition(const std::vector<double>& sample) {
  model_->CalculatePrediction(sample);
}

void Controller::SetInitialWeights() {
  std::string file_with_calculated_weights =
    CURRENT_PATH"/sources/" + std::to_string(model_->GetCountOfHiddenLayer()) +"_5.data";
  model_->LoadWeightsForModel(file_with_calculated_weights, model_->GetPerceptronType());
}

void Controller::ApplyCurrentPerceptronSettings(TypeOfPerceptron type, int number_of_hidden_layers) {
  if (type != model_->GetPerceptronType() ||
      number_of_hidden_layers != model_->GetCountOfHiddenLayer()) {
    model_->CreateNewPerceptron(type, number_of_hidden_layers);
    std::string file_with_calculated_weights =
      CURRENT_PATH"/sources/" + std::to_string(number_of_hidden_layers) +"_5.data";
    model_->LoadWeightsForModel(file_with_calculated_weights, type);
  }
}

void Controller::MakeTestSample(double sample_part) {
  static const std::string kTestSamplesFile = CURRENT_PATH"/sources/emnist-letters-test.csv";
  model_->SetSamplesDatasetFromFile(kTestSamplesFile);
  model_->MakeTesting(sample_part);
}

Measurements& Controller::GetMeasurements() {
  return model_->GetMeasurements();
}

void Controller::LoadWeightsFromFile(const std::string &file_name, const TypeOfPerceptron& type) {
  model_->LoadWeightsForModel(file_name, type);
}

void Controller::SaveWeightsInFile(const std::string &file_name) {
  model_->SaveWeightsForModel(file_name);
}

int Controller::GetLetterAnswerNumber() {
  return model_->GetResultOfNet();
}

void Controller::StartLearningNetwork(int number_of_epochs, int number_of_groups) {
  static const std::string kTrainingSamplesFile = CURRENT_PATH"/sources/emnist-letters-train.csv";
  model_->SetSamplesDatasetFromFile(kTrainingSamplesFile);
  model_->StartLearningNet(number_of_epochs, number_of_groups);
}

int Controller::GetHiddenLayerCount() {
  return model_->GetCountOfHiddenLayer();
}

}  // namespace s21

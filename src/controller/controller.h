#ifndef SRC_CONTROLLER_CONTROLLER_H_
#define SRC_CONTROLLER_CONTROLLER_H_
#include <string>
#include <vector>
#include "../model/model.h"

namespace s21 {

class Controller {
 public:
  Controller();
  explicit Controller(Model *model);
  ~Controller() = default;

  void MakeRecognition(const std::vector<double>& sample);
  void CreatePerceptron(TypeOfPerceptron type, int number_of_hidden_layers);
  void SetInitialWeights();
  void MakeTestSample(double sample_part);
  Measurements& GetMeasurements();
  void LoadWeightsFromFile(const std::string &file_name, const TypeOfPerceptron& type);
  void SaveWeightsInFile(const std::string &file_name);
  int GetLetterAnswerNumber();
  int GetHiddenLayerCount();
  void ApplyCurrentPerceptronSettings(TypeOfPerceptron type, int number_of_hidden_layers);
  void StartLearningNetwork(int number_of_epochs, int number_of_groups);

 private:
  Model *model_;
};

}  // namespace s21

#endif  // SRC_CONTROLLER_CONTROLLER_H_

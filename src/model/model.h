#ifndef SRC_MODEL_MODEL_H_
#define SRC_MODEL_MODEL_H_
#include <string>

#include "samples_dataset.h"
#include "perceptron.h"
#include "matrix_perceptron/matrix_perceptron.h"
#include "graph_perceptron/graph_perceptron.h"

namespace s21 {

struct Measurements{
  double average_accuracy_ = 0;
  double precision_ = 0;
  double f_measure_ = 0;
  double recall_ = 0;
  double time_spend_ = 0;
  std::vector<double> error_data_;
};

class Model {
 public:
  Model();
  explicit Model(Perceptron* perceptron);
  ~Model();
  void CreateNewPerceptron(TypeOfPerceptron type, int number_of_hidden_layers);
  void SetSamplesDatasetFromFile(const std::string& file_of_samples_dataset);
  const TypeOfPerceptron& GetPerceptronType() const;
  int GetCountOfHiddenLayer();

  Measurements& GetMeasurements();
  int GetResultOfNet();
  const std::vector<SamplesDataset>& GetSamplesDataset();

  void CalculatePrediction(const std::vector<double>& sample);
  void MakeTesting(double sample_part);
  void LoadWeightsForModel(std::string file_name, const TypeOfPerceptron& type);
  void SaveWeightsForModel(std::string file_name);

  void StartLearningNet(int number_of_epochs, int number_of_groups);
  inline void StartLearningNetOneEpoch(int num_of_gr);

 private:
  Perceptron* perceptron_;
  Measurements measurements_;
  std::vector<SamplesDataset> set_of_samples_;
};

}  // namespace s21
#endif  // SRC_MODEL_MODEL_H_

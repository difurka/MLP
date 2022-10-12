#ifndef SRC_MODEL_PERCEPTRON_H_
#define SRC_MODEL_PERCEPTRON_H_

#include "../model/samples_dataset.h"

namespace s21 {
enum LayerSize {
  kInput = 784,
  kHidden1 = 200,
  kHidden2 = 200,
  kHidden3 = 200,
  kHidden4 = 200,
  kHidden5 = 200,
  kOutput = 26
};

enum TypeOfPerceptron {
  kMatrix,
  kGraph
};

class Perceptron {
 public:
  Perceptron();
  explicit Perceptron(int number_of_hidden_layers, TypeOfPerceptron type);
  virtual ~Perceptron();

  void SetLayersSize(const std::vector<int> &count_of_neurons_in_layers);

  std::vector<double>& GetResultCosts();
  const TypeOfPerceptron& GetTypeOfPerceptron() const;
  int GetNumberOfHiddenLayers();
  virtual std::vector<double> GetPerceptronOuts() = 0;
  std::vector<int>& GetLayersSize();

  virtual void LoadWeights(const std::string& file_name) = 0;
  virtual void SaveWeights(const std::string& file_name) = 0;
  virtual void CalculateResult(const std::vector<double>& sample) = 0;
  virtual void MakeTraining(const SamplesDataset& sample) = 0;
  virtual void CalculateCostFunction(const SamplesDataset& sample) = 0;
  virtual void SetRandomWeights() = 0;
  void CleanResultCosts();
  void LayerSizeInitialize(int number_of_hidden_layers);

 protected:
  static constexpr double kLearningKoef = 0.3;
  static constexpr double kSmoothingKoef = 0.5;
  static constexpr double kRandomWeights = 30.0;
  std::vector<int> layers_size_;
  int number_of_hidden_layers_;
  TypeOfPerceptron type_of_perceptron_;
  std::vector<double> result_costs_;
};
}  // namespace s21
#endif  // SRC_MODEL_PERCEPTRON_H_

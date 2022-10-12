#ifndef SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_PERCEPTRON_H_
#define SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_PERCEPTRON_H_

#include <vector>
#include <string>
#include "matrix.h"
#include "../samples_dataset.h"
#include "../perceptron.h"

namespace s21 {
class MatrixPerceptron: public Perceptron {
class Layer {
 public:
  Layer() = default;
  Layer(int count_of_neurons, int count_of_inputs);
  ~Layer() = default;

  void SetLayerWeights(const Matrix& weights);
  void SetLayerBiases(const Matrix& biases);
  void SetOuts(const Matrix& outs);
  void SetDeltaBiases(const Matrix& weights);
  void SetDeltaWeights(const Matrix& weights);

  Matrix& GetLayerWeights();
  Matrix& GetLayerBiases();
  Matrix& GetLayerOuts();
  Matrix& GetDeltaBiases();
  Matrix& GetDeltaWeights();
  int GetCountOfInputs() const;
  int GetCountOfNeurons() const;
  void CalculateLayerOuts(const Matrix& inputs);

 private:
  int count_of_neurons_;
  int count_of_inputs_;
  Matrix weights_;
  Matrix biases_;
  Matrix linear_summa_;
  Matrix outs_;
  Matrix delta_weights_;
  Matrix delta_biases_;

  inline void CalculateLinearSumma(const Matrix& inputs);
};

 public:
  explicit MatrixPerceptron(int number_of_hidden_layes);
  explicit MatrixPerceptron(const std::vector<int>& count_of_neurons_in_layers);
  MatrixPerceptron(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases);
  ~MatrixPerceptron();

 private:
  std::vector<Layer> layers_;

  inline void CreatePerceptronFromMatrix
                        (const std::vector<Matrix>& weights, const std::vector<Matrix>& biases);
  inline void SetSamplesDataset(const std::string& file_of_samples_dataset, int cnt_of_samples);
  inline void SetNeuronsInLayers();
  inline void SetRandomWeights() override;
  inline std::vector<double> GetPerceptronOuts() override;

  inline void CalculatePerceptronOuts(const Matrix& input);
  inline void CalculateResult(const std::vector<double>& sample) override;
  inline int CountOfLayers();

  inline void LoadWeights(const std::string& file_name) override;
  inline void LoadLayersInfo(std::ifstream &file_stream);
  inline Matrix LoadLayersWeights(std::ifstream &file_stream, int rows, int cols);
  inline void SaveWeights(const std::string& file_name) override;
  inline void SaveLayerWeights(std::ofstream& out, const Matrix& layer);

  inline void MakeTraining(const SamplesDataset& sample) override;
  inline void MakeOneIterationBackForth();
  inline void CalculateCostFunction(const SamplesDataset& sample) override;
  inline void CalculateDeltaWeigtsForLastLayer(const SamplesDataset& sample);
  inline void CalculateDeltaWeigtsForOtherLayers(int number_of_layer);
};

}  // namespace s21
#endif  // SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_PERCEPTRON_H_

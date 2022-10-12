#ifndef SRC_MODEL_GRAPH_PERCEPTRON_GRAPH_PERCEPTRON_H_
#define SRC_MODEL_GRAPH_PERCEPTRON_GRAPH_PERCEPTRON_H_

#include <vector>
#include <string>
#include "../perceptron.h"

namespace s21 {

class GraphPerceptron : public Perceptron {
enum class LayerType {
  kInputLayer = 0,
  kHiddenLayer,
  kOutputLayer
};

class Neuron {
 public:
  explicit Neuron(std::vector<Neuron> *next_layer = nullptr);
  ~Neuron();

  void InputSignal(double input_value);
  void Activation();
  void OutputSignal();

  double& GetValue();
  void SetValue(double value);
  int GetNextLayerSize();
  double& GetError();
  void SetError(double error);
  double& GetDelta(int index);
  void SetDelta(int index, double value);
  std::vector<double>* GetWeights();
  void AddedRandomWeight(int next_layer_size);
  void UpdateWeights();

 private:
  double value_;
  double error_;
  std::vector<double> weight_;
  std::vector<double> weight_delta_;
  std::vector<Neuron> *next_layer_;
};

 public:
  explicit GraphPerceptron(int number_of_hidden_layers);
  ~GraphPerceptron();

 private:
  std::vector<std::vector<Neuron> > layers_;

  void InitializePerceptron();
  void CreateInputLayer();
  void CreateHiddenOutputLayers();
  void CreateLayer(std::vector<Neuron>* current_layer, std::vector<Neuron>* next_layer, const int size);
  void CreateOutputLayer(std::vector<GraphPerceptron::Neuron>* layer, const int size);

  void CalculateResult(const std::vector<double>& sample) override;
  void MakeTraining(const SamplesDataset& sample) override;
  void CalculateCostFunction(const SamplesDataset& sample) override;
  void SetRandomWeights() override;

  void MakeSignal(std::vector<Neuron>* layer, LayerType layer_type);
  void ForwardPropagation();
  void BackPropagation(int answer_index);
  void CalculateOutputLayerErrors(int answer_index);
  void CalculateHiddenInputLayerErrors();
  void CalculateDeltaWeight();
  void CorrectAllLayerWeights();

  std::vector<double> GetPerceptronOuts() override;

  void LoadWeights(const std::string& file_name) override;
  void SaveWeights(const std::string& file_name) override;
  void WriteLayersSize(std::ofstream& out);
  void WeightsFromFileToNeuronImport(std::string *line, std::vector<Neuron>* layer);
  void LoadLayerWeights(std::ifstream &file_stream, std::vector<Neuron>* layer);
  void SaveLayerWeights(std::ofstream& out, const std::vector<Neuron>& layer);
  void LoadPerceptron(const std::string& line);
  void ChangeLayers(int hidden_layers);
};

}  // namespace s21
#endif  // SRC_MODEL_GRAPH_PERCEPTRON_GRAPH_PERCEPTRON_H_

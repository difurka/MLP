#include <vector>
#include <gtest/gtest.h>
#include <ctime>

#include "../model.h"

const double EPS = 1e-6;

TEST(constructor_by_weights, check_output_forward) {
  std::vector<s21::Matrix> weights;
  weights.resize(2);
  std::vector<s21::Matrix> biases;
  biases.resize(2);

  weights[0].VectorToMatrix({1.0, -2.0, -3.0, 4.0}, 2, 2);
  biases[0].VectorToMatrix({1.0, -2.0}, 2, 1);
  weights[1].VectorToMatrix({1.0, -2.0, -3.0, 4.0, 1, -1}, 3, 2);
  biases[1].VectorToMatrix({1.0, -2.0, 1}, 3, 1);
  s21::Perceptron* net = new s21::MatrixPerceptron(weights, biases);
  std::vector<double> input = {1, 0};
  net->CalculateResult(input);

  ASSERT_NEAR(net->GetPerceptronOuts()[0], 0.8661584678657849, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[1], 0.009799075520213313, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[2], 0.8669324568778629, EPS);
  delete net;
}

TEST(load_from_file, check_output_forward) {
  s21::Perceptron* net = new s21::MatrixPerceptron({2, 8, 8, 9});
  net->LoadWeights("model/tests/weights_samples.data");
  std::vector<double> input = {1, 0};
  net->CalculateResult(input);
  ASSERT_NEAR(net->GetPerceptronOuts()[0], 0.8661584678657849, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[1], 0.009799075520213313, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[2], 0.8669324568778629, EPS);
  net->LoadWeights("model/tests/weights_samples.data");
  s21::Model* model = new s21::Model(net);

  model->SetSamplesDatasetFromFile("model/tests/dataset_test");
  net->CalculateResult(model->GetSamplesDataset()[0].GetSample());

  ASSERT_NEAR(net->GetPerceptronOuts()[0], 0.8006108386669526, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[1], 0.029883769286894936, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[2], 0.8241892368931192, EPS);

  net->CalculateResult(model->GetSamplesDataset()[1].GetSample());

  ASSERT_NEAR(net->GetPerceptronOuts()[0], 0.8155049919645286, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[1], 0.024100912317978045, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[2], 0.833058745024583, EPS);

  net->CalculateResult(model->GetSamplesDataset()[2].GetSample());

  ASSERT_NEAR(net->GetPerceptronOuts()[0], 0.8243087308581273, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[1], 0.021147724041141857, EPS);
  ASSERT_NEAR(net->GetPerceptronOuts()[2], 0.8381484680870984, EPS);
  delete model;
}

TEST(calculate_cost_function, one_step_forward) {
  s21::Perceptron* net = new s21::MatrixPerceptron({2, 8, 8, 9});
  net->LoadWeights("model/tests/weights_samples.data");
  s21::Model* model = new s21::Model(net);
  model->SetSamplesDatasetFromFile("model/tests/dataset_test");
  s21::SamplesDataset sample = model->GetSamplesDataset()[0];
  net->CalculateResult(sample.GetSample());
  net->CalculateCostFunction(sample);
  ASSERT_NEAR(net->GetResultCosts()[0], 0.7199369755, EPS);

  sample = model->GetSamplesDataset()[1];
  net->CalculateResult(sample.GetSample());
  net->CalculateCostFunction(sample);
  ASSERT_NEAR(net->GetResultCosts()[1], 2.311414294, EPS);

  sample = model->GetSamplesDataset()[2];
  net->CalculateResult(sample.GetSample());
  net->CalculateCostFunction(sample);
  ASSERT_NEAR(net->GetResultCosts()[2], 0.7061280284, EPS);
  delete model;
}

TEST(load_from_dataset, check_samples) {
  s21::Matrix set_original;

  s21::Perceptron* net = new s21::MatrixPerceptron({5, 6, 7});
  s21::Model* model = new s21::Model(net);

  model->SetSamplesDatasetFromFile("model/tests/dataset_test");

  set_original.VectorToMatrix({0.9 / 255, 20 / 255.0, 0, 0, 30 / 255.0}, 5, 1);
  ASSERT_TRUE(model->GetSamplesDataset()[0].GetMatrixOfSample() == set_original);
  set_original.VectorToMatrix({1 / 255.0, 2 / 255.0, 3 / 255.0, 4 / 255.0, 5 / 255.0}, 5, 1);
  ASSERT_TRUE(model->GetSamplesDataset()[1].GetMatrixOfSample() == set_original);
  set_original.VectorToMatrix({22 / 255.0, 3 / 255.0, 4 / 255.0, 7.8 / 255.0, 0.1 / 255.0}, 5, 1);
  ASSERT_TRUE(model->GetSamplesDataset()[2].GetMatrixOfSample() == set_original);
  set_original.VectorToMatrix({7 / 255.0, 0.9 / 255.0, 0.8 / 255.0, 6 / 255.0, 7 / 255.0}, 5, 1);
  ASSERT_TRUE(model->GetSamplesDataset()[3].GetMatrixOfSample() == set_original);
  delete model;
}


TEST(algorithm, index_of_max_element) {
  std::vector<double> answer = {1, 2, 3, 4, 6, 4, 3};
  int index = std::max_element(answer.begin(), answer.end()) - answer.begin();
  ASSERT_EQ(index, 4);

  answer = {0.01, 1, 2, 3, 4, 6, 4, 3, 6.009};
  index = std::max_element(answer.begin(), answer.end()) - answer.begin() + 1;
  ASSERT_EQ(index, 9);

  answer = {100.01, 1, 2, 3, 4, 6, 4, 3, 6.009};
  index = std::max_element(answer.begin(), answer.end()) - answer.begin() + 1;
  ASSERT_EQ(index, 1);
}

TEST(training, for_simple_case) {
  s21::Perceptron* net = new s21::MatrixPerceptron({16, 20, 20, 10});
  net->SetRandomWeights();
  s21::Model* model = new s21::Model(net);
  for (int i = 0; i < 10; ++i) {
    s21::SamplesDataset teach(16, 10);
    std::string str = "1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0";
    teach.SetSample(&str);
    teach.SetTrueLetter(4);
    net->CalculateResult(teach.GetSample());
    std::vector<double> answer;
    for (int k = 0; k < 1000; ++k) {
      net->MakeTraining(teach);
    }
    s21::SamplesDataset test(16, 10);
    str = "1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1";
    test.SetSample(&str);
    net->CalculateResult(test.GetSample());
    answer = net->GetPerceptronOuts();
    ASSERT_EQ(std::max_element(answer.begin(), answer.end()) - answer.begin() + 1, 4);
  }
  delete model;
}

TEST(graph_perceptron, graph_perceptron) {  // throw generate one leaks
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kGraph, 5);
  ASSERT_THROW(
  model.LoadWeightsForModel("error_path.data", s21::TypeOfPerceptron::kGraph)
  , std::invalid_argument);

  ASSERT_ANY_THROW(
  model.LoadWeightsForModel("model/tests/error_file_layer_size.data",
    s21::TypeOfPerceptron::kGraph));

  ASSERT_THROW(
  model.LoadWeightsForModel("model/tests/error_file_weights.data", s21::TypeOfPerceptron::kGraph)
  , std::invalid_argument);

  ASSERT_EQ(model.GetPerceptronType(), s21::TypeOfPerceptron::kGraph);

  s21::GraphPerceptron gp(2);
  ASSERT_EQ(gp.GetTypeOfPerceptron(), s21::TypeOfPerceptron::kGraph);

  s21::Perceptron* prc = new s21::GraphPerceptron(2);
  ASSERT_EQ(prc->GetTypeOfPerceptron(), s21::TypeOfPerceptron::kGraph);
  delete prc;
}

TEST(measurements, measurements) {
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kMatrix, 5);
  model.LoadWeightsForModel("sources/5_5.data", s21::TypeOfPerceptron::kMatrix);
  model.SetSamplesDatasetFromFile("sources/emnist-letters-test.csv");
  model.MakeTesting(0.1);
  auto &measur = model.GetMeasurements();
  ASSERT_GT(measur.average_accuracy_, 0);
  ASSERT_GT(measur.precision_, 0);
  ASSERT_GT(measur.f_measure_, 0);
  ASSERT_GT(measur.recall_, 0);
  ASSERT_GT(measur.time_spend_, 0);
}

TEST(graph_perceptron, recognise_5) {
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kGraph, 5);
  model.LoadWeightsForModel("sources/5_5.data", s21::TypeOfPerceptron::kGraph);
  model.SetSamplesDatasetFromFile("sources/emnist-letters-test.csv");
  auto& data_set = model.GetSamplesDataset();
  for (int i = 0; i < 10; i += 10) {
    auto& sample = data_set[i].GetSample();
    model.CalculatePrediction(sample);
    ASSERT_EQ(data_set[i].GetIndexOfAnswerLetter(), model.GetResultOfNet());
  }
}

TEST(graph_perceptron, recognise_4) {
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kGraph, 4);
  model.LoadWeightsForModel("sources/4_5.data", s21::TypeOfPerceptron::kGraph);
  model.SetSamplesDatasetFromFile("sources/emnist-letters-test.csv");
  auto& data_set = model.GetSamplesDataset();
  for (int i = 0; i < 10; i += 10) {
    auto& sample = data_set[i].GetSample();
    model.CalculatePrediction(sample);
    ASSERT_EQ(data_set[i].GetIndexOfAnswerLetter(), model.GetResultOfNet());
  }
}

TEST(graph_perceptron, recognise_3) {
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kGraph, 3);
  model.LoadWeightsForModel("sources/3_5.data", s21::TypeOfPerceptron::kGraph);
  model.SetSamplesDatasetFromFile("sources/emnist-letters-test.csv");
  auto& data_set = model.GetSamplesDataset();
  for (int i = 0; i < 10; i += 10) {
    auto& sample = data_set[i].GetSample();
    model.CalculatePrediction(sample);
    ASSERT_EQ(data_set[i].GetIndexOfAnswerLetter(), model.GetResultOfNet());
  }
}

TEST(graph_perceptron, recognise_2) {
  s21::Model model;
  model.CreateNewPerceptron(s21::TypeOfPerceptron::kGraph, 2);
  model.LoadWeightsForModel("sources/2_5.data", s21::TypeOfPerceptron::kGraph);
  model.SetSamplesDatasetFromFile("sources/emnist-letters-test.csv");
  auto& data_set = model.GetSamplesDataset();
  for (int i = 0; i < 10; i += 10) {
    auto& sample = data_set[i].GetSample();
    model.CalculatePrediction(sample);
    ASSERT_EQ(data_set[i].GetIndexOfAnswerLetter(), model.GetResultOfNet());
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

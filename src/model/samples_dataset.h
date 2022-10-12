#ifndef SRC_MODEL_SAMPLES_DATASET_H_
#define SRC_MODEL_SAMPLES_DATASET_H_

#include <string>
#include <vector>
#include "matrix_perceptron/matrix.h"


namespace s21 {

class SamplesDataset {
 public:
  SamplesDataset(int cnt_of_inputs, int cnt_of_outputs);

  void SetSample(std::string* sample_of_letter);
  void SetTrueLetter(int letter);
  void SetCntInputs(int n);
  void SetCntOutputs(int n);

  int GetIndexOfAnswerLetter() const;
  const std::vector<double>&  GetSample() const;
  Matrix GetMatrixOfSample() const;
  int GetCntOfOutputs() const;

 private:
  static constexpr double kMaxColor_ = 255.0;
  int answer_nuber_letter_;
  std::vector<double> sample_;

  int cnt_of_outputs_;
  int cnt_of_inputs_;
};

}  // namespace s21
#endif  // SRC_MODEL_SAMPLES_DATASET_H_

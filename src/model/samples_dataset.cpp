#include <stdexcept>
#include "samples_dataset.h"

namespace s21 {
SamplesDataset::SamplesDataset(int cnt_of_inputs, int cnt_of_outputs)
  : answer_nuber_letter_(0), cnt_of_outputs_(cnt_of_outputs), cnt_of_inputs_(cnt_of_inputs) { }

void SamplesDataset::SetSample(std::string* sample_of_letter) {
  sample_ = std::vector<double>(cnt_of_inputs_);
  size_t position = 0;
  int j = 0;
  while ((position = sample_of_letter->find(",")) != std::string::npos && j < cnt_of_inputs_ - 1) {
      std::string token = sample_of_letter->substr(0, position);
      if (token.size() > 0) {
        sample_[j] = std::stof(token) / kMaxColor_;
        ++j;
      }
      sample_of_letter->erase(0, position + 1);
  }
  sample_[j] = std::stof(*sample_of_letter) / kMaxColor_;
  if (j != cnt_of_inputs_ - 1)
    throw std::out_of_range("wrong file for sets, number of inputs is less than need");
}

void SamplesDataset::SetTrueLetter(int letter) {
  answer_nuber_letter_ = letter - 1;
}

void SamplesDataset::SetCntOutputs(int n) {
  cnt_of_outputs_ = n;
}

void SamplesDataset::SetCntInputs(int n) {
  cnt_of_inputs_ = n;
}

int SamplesDataset::GetIndexOfAnswerLetter() const {
  return answer_nuber_letter_;
}

const std::vector<double>& SamplesDataset::GetSample() const {
  return sample_;
}

Matrix SamplesDataset::GetMatrixOfSample() const {
  Matrix result(cnt_of_inputs_, 1);
  result.VectorToMatrix(sample_, cnt_of_inputs_, 1);
  return result;
}

int SamplesDataset::GetCntOfOutputs() const {
  return cnt_of_outputs_;
}
}  // namespace s21

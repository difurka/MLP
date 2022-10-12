#ifndef SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_H_
#define SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_H_

#include <vector>

namespace s21 {

class Matrix {
  friend Matrix operator*(const double num, const Matrix& other);

 public:
  Matrix();
  Matrix(int rows, int cols);
  Matrix(const Matrix& other);
  Matrix(Matrix&& other);
  ~Matrix();

  void SetRows(int);
  void SetColumns(int);
  int GetRows() const;
  int GetColumns() const;

  Matrix TakeRow(int i) const;
  void AppEndRow(const Matrix& row);

  bool EqMatrix(const Matrix& other);
  void SumMatrix(const Matrix& other);
  void SubMatrix(const Matrix& other);
  void MulNumber(const double num);
  void MulMatrix(const Matrix& other);
  Matrix transpose();

  Matrix operator+(const Matrix& other) const;
  Matrix operator-(const Matrix& other) const;
  Matrix operator*(const Matrix& other) const;
  Matrix operator*(const double num) const;
  bool operator==(const Matrix& other);
  Matrix& operator=(const Matrix& other);
  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);
  Matrix& operator*=(const double num);
  Matrix& operator*=(const Matrix& other);
  double& operator()(int row, int col) const;

  void Random(double a, double b);
  void VectorToMatrix(const std::vector<double>& vector, int n, int m);
  void Print() const;

 private:
  int rows_, cols_;
  double **matrix_;
  void Create();
  void Destroy();
};

void PrintVector(const std::vector<double>&);
}  // namespace s21
#endif  // SRC_MODEL_MATRIX_PERCEPTRON_MATRIX_H_

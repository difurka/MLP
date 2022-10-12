#include <math.h>
#include <iostream>
#include "matrix.h"
#include <random>

namespace s21 {

Matrix::Matrix(): rows_(0), cols_(0), matrix_(nullptr) {}

Matrix::Matrix(int rows, int cols) {
  if (rows <= 0 || cols <=0)
    throw std::out_of_range("Incorrect input, index is out of range");
  rows_ = rows;
  cols_ = cols;
  Create();
}

Matrix::Matrix(const Matrix& other):
  rows_(other.GetRows()), cols_(other.GetColumns()) {
  Create();
  for (int i = 0; i < rows_; i++)
    for (int j = 0; j < cols_; j++)
      matrix_[i][j] = other.matrix_[i][j];
}

Matrix::Matrix(Matrix&& other):
  rows_(0), cols_(0), matrix_(nullptr) {
  std::swap(rows_, other.rows_);
  std::swap(cols_, other.cols_);
  std::swap(matrix_, other.matrix_);
}

Matrix::~Matrix() {
  Destroy();
}
void Matrix::Destroy() {
  for (int i = 0; i < rows_; i++)
    if (matrix_[i] != nullptr) delete [] matrix_[i];
  if (matrix_ != nullptr) delete[] matrix_;
}

void Matrix::Create() {
  matrix_ = new double*[rows_];
  for (int i = 0; i <rows_; i++)
    matrix_[i] = new double[cols_];
  for (int i = 0; i < rows_; i++)
    for (int j = 0; j < cols_; j++)
      matrix_[i][j] = 0;
}

int Matrix::GetColumns() const {
  return cols_;
}

int Matrix::GetRows() const {
  return rows_;
}

void Matrix::SetColumns(int cols_) {
  if (cols_ != this->cols_) {
    int rows_ = this->rows_;
    Destroy();
    this->cols_ = cols_;
    this->rows_ = rows_;
    Create();
  }
}

void Matrix::SetRows(int rows_) {
  if (rows_ != this->rows_) {
    int cols_ = this->cols_;
    Destroy();
    this->cols_ = cols_;
    this->rows_ = rows_;
    Create();
  }
}

Matrix Matrix::TakeRow(int i) const {
  if (rows_ <= i) {
    throw std::out_of_range("Incorrect input, different matrix dimensions for take row");
  }
  Matrix row = Matrix(1, cols_);
  for (int j = 0; j < cols_; ++j) row(0, j) = matrix_[i][j];
  return row;
}

void Matrix::AppEndRow(const Matrix& row) {
  if (cols_ != row.GetColumns() || row.GetRows() != 1) {
    throw std::out_of_range("Incorrect row for append.");
  }
  Matrix temp = Matrix(rows_ + 1, cols_);
  for (int i = 0; i < rows_; ++i)
    for (int j = 0; j < cols_; ++j)
      temp(i, j) = matrix_[i][j];
  for (int j = 0; j < cols_; ++j) {
    temp(rows_, j) = row.matrix_[0][j];
  }
  *this = temp;
}

Matrix Matrix::transpose() {
    Matrix result = Matrix(cols_, rows_);
    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++)
            result(j, i) = matrix_[i][j];
    return result;
}


bool Matrix::EqMatrix(const Matrix& other) {
  bool result = true;
  if (rows_ != other.rows_ || cols_ != other.cols_) result = false;
  for (int i = 0; i < rows_ && result; i++)
    for (int j = 0; j < cols_ && result; j++)
      if (fabs(matrix_[i][j] - other.matrix_[i][j]) > 1e-4) result = false;
  return result;
}

void Matrix::SumMatrix(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::out_of_range("Incorrect input, different matrix dimensions");
  }
  for (auto i = 0; i < rows_; i++)
    for (auto j = 0; j < cols_; j++)
      matrix_[i][j] += other.matrix_[i][j];
}

void Matrix::SubMatrix(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::out_of_range("Incorrect input, different matrix dimensions");
  }
  for (auto i = 0; i < rows_; i++)
    for (auto j = 0; j < cols_; j++)
      matrix_[i][j] -= other.matrix_[i][j];
}

void Matrix::MulNumber(const double num) {
  for (auto i = 0; i < rows_; i++)
    for (auto j = 0; j < cols_; j++)
      matrix_[i][j] *= num;
}

void Matrix::MulMatrix(const Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::out_of_range("Incorrect input, the number of columns and rows are not equal");
  }
  Matrix temp(*this);
  Destroy();
  cols_ = other.cols_;
  Create();
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] = 0;
      for (int k = 0; k < temp.cols_; k++) {
        matrix_[i][j] += (temp.matrix_[i][k])*(other.matrix_[k][j]);
      }
    }
  }
}

Matrix Matrix::operator+(const Matrix& other) const {
  Matrix result(*this);
  result.SumMatrix(other);
  return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  Matrix result(*this);
  result.SubMatrix(other);
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  Matrix result(*this);
  result.MulMatrix(other);
  return result;
}

Matrix Matrix::operator*(const double num) const {
  Matrix result(*this);
  result.MulNumber(num);
  return result;
}

Matrix operator*(const double num, const Matrix& other) {
  Matrix result(other);
  result.MulNumber(num);
  return result;
}

bool Matrix::operator==(const Matrix& other) {
  return EqMatrix(other);
}

Matrix& Matrix::operator=(const Matrix& other) {
  if (&other != this) {
    Destroy();
    cols_ = other.cols_;
    rows_ = other.rows_;
    Create();
    for (int i = 0; i < rows_; i++)
      for (int j = 0; j < cols_; j++)
        matrix_[i][j] = other.matrix_[i][j];
  }
  return *this;
}

Matrix& Matrix::operator+=(const Matrix& other) {
  SumMatrix(other);
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  SubMatrix(other);
  return *this;
}

Matrix& Matrix::operator*=(const double num) {
  MulNumber(num);
  return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
  MulMatrix(other);
  return *this;
}

double& Matrix::operator()(int row, int col) const {
  if (row > rows_ || row < 0 || col < 0 || col > cols_)
    throw std::out_of_range("Incorrect input, index is out of range");
  return matrix_[row][col];
}


void Matrix::VectorToMatrix(const  std::vector<double>& vector, int n, int m) {
  size_t matrix_size = n * m;
  if (vector.size() < matrix_size)
    throw std::out_of_range("Vector is smaller then dim of matrix");
  SetRows(n);
  SetColumns(m);
  int k = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      matrix_[i][j] = vector[k++];
}

void Matrix::Random(double a, double b) {
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_int_distribution<int> uniform_dist(a, b);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] = ((double)(uniform_dist(engine) / 100.0));
    }
  }
}

void Matrix::Print() const {
    std::cout << "Rows: " << GetRows();
    std::cout << "\nColumns: " << GetColumns() << "\n";
    for (int i = 0; i < GetRows(); i++) {
        for (int j = 0; j < GetColumns(); j++)
            std::cout << matrix_[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void PrintVector(const std::vector<double>& vector) {
  for (size_t i = 0; i < vector.size(); ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}
}  // namespace s21

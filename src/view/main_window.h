#ifndef SRC_VIEW_MAIN_WINDOW_H_
#define SRC_VIEW_MAIN_WINDOW_H_

#include <iostream>

#include <QMainWindow>
#include <QFileDialog>
#include <QFile>
#include <QPushButton>
#include <QPainter>
#include <memory>
#include <QIcon>
#include <QThread>

#include "../controller/controller.h"

namespace Ui {
class MainWindow;
}

namespace s21 {

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(s21::Controller *controller, QWidget *parent = nullptr);
  ~MainWindow();

 private slots:
  void on_load_weights_button_clicked();
  void on_save_weights_button_clicked();

  void on_start_testing_clicked();
  void on_start_learning_clicked();
  void on_check_box_cross_validation_toggled(bool checked);

  void on_pushButton_clear_clicked();
  void on_pushButton_recognise_clicked();
  void on_pushButton_load_img_clicked();
  void on_pushButton_close_report_clicked();

  void on_check_box_learn_report_clicked(bool change);
  void on_check_box_test_report_clicked(bool change);

 private:
  enum  Report {
    kTest = 0,
    kLearn,
    kAnswer,
    kComplited,
    kError
  };

  static constexpr int window_hight = 745;
  static constexpr int window_width = 615;

  Ui::MainWindow *ui;
  s21::Controller *controller_;

  void MakeTestReport();
  void MakeLearnReport();
  void IconInitialize();
  void ShowReport(int page);
  void Recognize();
  int GetCurrentHiddenLayerCount();
  s21::TypeOfPerceptron GetCurrentPerceptronType();
  std::vector<double> ReadPixels(const QImage &image);
  void UpdatePerceprtonSettings(int number_hidden_layers);
};

}  // namespace s21

#endif  // SRC_VIEW_MAIN_WINDOW_H_

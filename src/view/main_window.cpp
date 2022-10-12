#include "main_window.h"
#include "ui_main_window.h"
#include <exception>

namespace s21 {

MainWindow::MainWindow(s21::Controller *controller, QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow),
  controller_(controller) {
  ui->setupUi(this);
  ui->stackedWidget_report->setStyleSheet("background-color: QColor(255, 255, 255, 255)");
  ui->stackedWidget_report->hide();
  IconInitialize();
  this->setFixedSize(window_hight, window_width);
  this->setStyleSheet("background-color: #323232; color: white");
  ui->label_answer->setStyleSheet("QLabel {color : #323232}");
  ui->label_message->setAlignment(Qt::AlignHCenter);

  try {
    controller_->CreatePerceptron(GetCurrentPerceptronType(), GetCurrentHiddenLayerCount());
    controller_->SetInitialWeights();
  } catch (std::exception &ex) {
    ui->label_message->setText(ex.what());
    ShowReport(Report::kError);
  }
}

void MainWindow::IconInitialize() {
  const int kIconSize = 32;
  const int kPictureSize = 100;
  ui->pushButton_load_img->setIcon(QIcon(CURRENT_PATH"/icons/open_.png"));
  ui->pushButton_load_img->setIconSize(QSize(kIconSize, kIconSize));

  ui->pushButton_clear->setIcon(QIcon(CURRENT_PATH"/icons/clear_.png"));
  ui->pushButton_clear->setIconSize(QSize(kIconSize, kIconSize));

  ui->pushButton_recognise->setIcon(QIcon(CURRENT_PATH"/icons/recognise_.png"));
  ui->pushButton_recognise->setIconSize(QSize(kIconSize, kIconSize));

  QIcon icon(CURRENT_PATH"/icons/neuro_.png");
  QPixmap pixmap = icon.pixmap(QSize(kPictureSize, kPictureSize));
  ui->label_image->setPixmap(pixmap);

  ui->pushButton_close_report->setIcon(QIcon(CURRENT_PATH"/icons/close_.png"));
  ui->pushButton_close_report->setIconSize(QSize(kIconSize, kIconSize));
  ui->pushButton_close_report->hide();
}


MainWindow::~MainWindow() {
  delete ui;
}

/* -------------------------------------------------------------------------- */
/*                                   Buttons                                  */
/* -------------------------------------------------------------------------- */

void MainWindow::on_load_weights_button_clicked() {
  std::string file_name =
    QFileDialog::getOpenFileName(this, ("Load weights"), CURRENT_PATH, "*.data").toStdString();
  if (file_name.length() > 0) {
      try {
        controller_->LoadWeightsFromFile(file_name, s21::TypeOfPerceptron::kGraph);
        UpdatePerceprtonSettings(controller_->GetHiddenLayerCount());
        on_pushButton_close_report_clicked();
      } catch (std::exception &ex) {
        ui->label_message->setText(ex.what());
        ShowReport(Report::kError);
      }
  }
}

void MainWindow::UpdatePerceprtonSettings(int number_hidden_layers) {
  if (number_hidden_layers == 2) {
    ui->radio_button_2->setChecked(true);
  } else if (number_hidden_layers == 3) {
    ui->radio_button_3->setChecked(true);
  } else if (number_hidden_layers == 4) {
    ui->radio_button_4->setChecked(true);
  } else if (number_hidden_layers == 5) {
    ui->radio_button_5->setChecked(true);
  }
}

void MainWindow::on_save_weights_button_clicked() {
  std::string file_name =
    QFileDialog::getSaveFileName(this, ("Save weights"), CURRENT_PATH, "*.data").toStdString();
  if (file_name.length() > 0) {
    controller_->SaveWeightsInFile(file_name);
  }
}

void MainWindow::on_start_testing_clicked() {
  try {
    controller_->ApplyCurrentPerceptronSettings(GetCurrentPerceptronType(), GetCurrentHiddenLayerCount());
    controller_->MakeTestSample(ui->sample_part->value());
    MakeTestReport();
  } catch (std::exception &ex) {
    ui->label_message->setText(ex.what());
    ShowReport(Report::kError);
  }
}

void MainWindow::MakeTestReport() {
  s21::Measurements mesure = controller_->GetMeasurements();
  QString accuracy = "average accuracy: \t" + QString::number((int)(mesure.average_accuracy_ * 100)) + "%";
  QString precision = "precision: \t\t" + QString::number((int)(mesure.precision_ * 100)) + "%";
  QString f_measure = "f measure: \t\t" + QString::number((int)(mesure.f_measure_ * 100)) + "%";
  QString recal = "recall: \t\t\t" + QString::number((int)(mesure.recall_ * 100)) + "%";
  QString time = "total time spend: \t\t" + QString::number((mesure.time_spend_)) + " seconds";

  ui->label_precision->setText(accuracy);
  ui->label_accuracy->setText(precision);
  ui->label_f_measure->setText(f_measure);
  ui->label_recal->setText(recal);
  ui->label_time->setText(time);
  if (ui->check_box_test_report->isChecked()) {
    ShowReport(Report::kTest);
  } else {
    ShowReport(Report::kComplited);
  }
}

void MainWindow::on_start_learning_clicked() {
  int cross_validation = ui->check_box_cross_validation->isChecked() ? ui->crossv_groups_count->value() : 1;
  try {
    controller_->CreatePerceptron(GetCurrentPerceptronType(), GetCurrentHiddenLayerCount());
    controller_->StartLearningNetwork(ui->epochs_count->value(), cross_validation);
    MakeTestReport();
  } catch (std::exception &ex) {
    ui->label_message->setText(ex.what());
    ShowReport(Report::kError);
  }
  on_save_weights_button_clicked();
  if (ui->check_box_learn_report->isChecked()) {
    MakeLearnReport();
  } else {
    ShowReport(Report::kComplited);
  }
}

void MainWindow::MakeLearnReport() {
  ui->draw_widget->clear();
  ShowReport(Report::kLearn);
  ui->draw_widget->drawReport(controller_->GetMeasurements().error_data_);
}

void MainWindow::ShowReport(int page) {
  on_pushButton_close_report_clicked();
  if (page != kAnswer)   ui->draw_widget->clear();
  ui->stackedWidget_report->setCurrentIndex(page);
  ui->draw_widget->setLock(true);
  ui->pushButton_close_report->show();
  ui->stackedWidget_report->show();
}

void MainWindow::on_check_box_cross_validation_toggled(bool checked) {
  ui->crossv_groups_count->setEnabled(checked);
  ui->label_number_of_groups->setEnabled(checked);
}

void MainWindow::on_pushButton_clear_clicked() {
  if (ui->stackedWidget_report->currentIndex() == 2) {
    on_pushButton_close_report_clicked();
  }
  if (!ui->stackedWidget_report->isVisible()) {
    ui->draw_widget->clear();
  }
}

void MainWindow::on_pushButton_recognise_clicked() {
  try {
    controller_->ApplyCurrentPerceptronSettings(GetCurrentPerceptronType(), GetCurrentHiddenLayerCount());
    Recognize();
    QString  answer(char(controller_->GetLetterAnswerNumber() + 65));
    ui->label_answer->setText(answer);
    ui->stackedWidget_report->setCurrentIndex(Report::kAnswer);
    ui->stackedWidget_report->show();
  } catch (std::exception &ex) {
    ui->label_message->setText(ex.what());
    ShowReport(Report::kError);
  }
}

void MainWindow::Recognize() {
  try {
    QImage image = ui->draw_widget->getCurrentImage();
    std::vector<double> pixels = ReadPixels(image);
    controller_->MakeRecognition(pixels);
  } catch (std::exception &ex) {
    ui->label_message->setText(ex.what());
    ShowReport(Report::kError);
  }
}

std::vector<double> MainWindow::ReadPixels(const QImage &image) {
  const double kMaxColor = 255.0;
  const int kPixelSize = 28;
  std::vector<double> image_pixels;
  for (int i = 0; i < kPixelSize; ++i) {
    for (int j = 0; j < kPixelSize; ++j) {
        QRgb pixel = image.pixel(i, j);
        double normalize = (pixel & 0xFF) / kMaxColor;
        image_pixels.push_back(normalize);
      }
  }
  return image_pixels;
}

void MainWindow::on_pushButton_load_img_clicked() {
  QString file_name = QFileDialog::getOpenFileName(this, ("Load image"), CURRENT_PATH"", "*.bmp");
  if (file_name.length() > 0) {
    QImage image;
    image.load(file_name);
    image = image.convertToFormat(QImage::Format_Grayscale8);
    image = image.scaled(QSize(512, 512), Qt::IgnoreAspectRatio);
    ui->draw_widget->setImg(image);
  }
}

void MainWindow::on_pushButton_close_report_clicked() {
  ui->stackedWidget_report->hide();
  ui->pushButton_close_report->hide();
  ui->draw_widget->setLock(false);
  ui->draw_widget->clear();
}


void MainWindow::on_check_box_learn_report_clicked(bool change) {
  if (change == true) {
    MakeLearnReport();
    if  ( ui->check_box_test_report->isChecked())
      ui->check_box_test_report->setChecked(false);
  } else if (change == false) {
    on_pushButton_close_report_clicked();
    ui->draw_widget->clear();
  }
}

void MainWindow::on_check_box_test_report_clicked(bool change) {
  if (change == true) {
    MakeTestReport();
    if  ( ui->check_box_learn_report->isChecked())
      ui->check_box_learn_report->setChecked(false);
  } else if (change == false) {
    on_pushButton_close_report_clicked();
  }
}

int MainWindow::GetCurrentHiddenLayerCount() {
  int count = 0;
  if (ui->radio_button_2->isChecked()) {
    count = 2;
  } else if (ui->radio_button_3->isChecked()) {
    count = 3;
  } else if (ui->radio_button_4->isChecked()) {
    count = 4;
  } else if (ui->radio_button_5->isChecked()) {
    count = 5;
  }
  return count;
}

s21::TypeOfPerceptron MainWindow::GetCurrentPerceptronType() {
  return (ui->radio_button_matrix_type->isChecked()) ?
    s21::TypeOfPerceptron::kGraph : s21::TypeOfPerceptron::kMatrix;
}

}  // namespace s21

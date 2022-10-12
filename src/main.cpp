#include <QtWidgets>
#include "view/main_window.h"

int main(int argc, char **argv) {
  QApplication application(argc, argv);
  s21::Model model;
  s21::Controller controller(&model);
  s21::MainWindow window(&controller);
  window.show();
  return application.exec();
}

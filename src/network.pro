QT += widgets
QT += gui
QT += core

INCLUDEPATH += $$PWD

FORMS += \
  view/main_window.ui

HEADERS += \
  view/main_window.h \
  controller/controller.h \
  view/paint_widget.h \
  model/model.h \
  model/perceptron.h \
  model/graph_perceptron/graph_perceptron.h \
  model/samples_dataset.h \
  model/matrix_perceptron/matrix_perceptron.h \
  model/matrix_perceptron/matrix.h

SOURCES += \
  view/main_window.cpp \
  controller/controller.cpp \
  main.cpp \
  view/paint_widget.cpp \
  model/model.cpp  \
  model/perceptron.cpp \
  model/graph_perceptron/graph_perceptron.cpp \
  model/samples_dataset.cpp \
  model/matrix_perceptron/matrix_perceptron.cpp \
  model/matrix_perceptron/matrix.cpp

RESOURCES += \

DEFINES += CURRENT_PATH='\\"$${PWD}\\"'

#ifndef SRC_VIEW_PAINT_WIDGET_H_
#define SRC_VIEW_PAINT_WIDGET_H_

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <vector>
#include <memory>
namespace s21 {

class PaintWidget : public QWidget {
  Q_OBJECT

 public:
  explicit PaintWidget(QWidget *parent = nullptr);
  void paintEvent(QPaintEvent *event) override;

  void resizeEvent(QResizeEvent *event) override;
  void mousePressEvent(QMouseEvent * event) override;
  void mouseMoveEvent(QMouseEvent * event) override;

  void drawReport(const std::vector<double> &mesure);
  QImage getCurrentImage();
  void draw(const QPoint & pos, Qt::MouseButton event);
  void clear();
  void setImg(QImage image);
  void setLock(bool is_lock);

 private:
  static const int kDefaultPenWidth = 80;
  static const int kReportPenWidth = 40;

  QPixmap pixmap_;
  QPoint lastPos_;
  std::unique_ptr<QPen> pen_;

  bool is_locked_;
  bool is_write_;
};

}  // namespace s21

#endif  // SRC_VIEW_PAINT_WIDGET_H_

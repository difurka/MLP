#include "paint_widget.h"
namespace s21 {

PaintWidget::PaintWidget(QWidget *parent)
  : QWidget{parent},
  is_locked_(false),
  is_write_(true) {
  pen_ = std::make_unique<QPen>
    (QPen(Qt::white, kDefaultPenWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
}

void PaintWidget::paintEvent(QPaintEvent *event) {  // override
  QPainter painter(this);
  painter.drawPixmap(0, 0, pixmap_);
  this->update();
}

void PaintWidget::resizeEvent(QResizeEvent *event)  {  // override
  auto newRect = pixmap_.rect().united(rect());
  if (!(newRect == pixmap_.rect())) {
    QPixmap newPixmap{newRect.size()};
    QPainter painter{&newPixmap};
    painter.fillRect(newPixmap.rect(), Qt::black);
    painter.drawPixmap(0, 0, pixmap_);
    pixmap_ = newPixmap;
  }
}

void PaintWidget::mousePressEvent(QMouseEvent* event)  {
  if (event->button() == Qt::RightButton && is_locked_ == false) {
    clear();
    is_write_ = false;
    lastPos_ = event->pos();
  } else if (event->button() == Qt::LeftButton) {
    is_write_ = true;
    lastPos_ = event->pos();
    draw(event->pos(), event->button());
  }
}

void PaintWidget::mouseMoveEvent(QMouseEvent* event)  {
  draw(event->pos(), event->button());
}

void PaintWidget::draw(const QPoint& pos, Qt::MouseButton event) {  // override
  if (is_locked_ == false && is_write_) {
    QPainter painter{&pixmap_};
    painter.setPen(*pen_);
    painter.drawLine(lastPos_, pos);
    lastPos_ = pos;
    update();
  }
}

void PaintWidget::clear() {
  pixmap_.fill(Qt::black);
  this->clearMask();
}

void PaintWidget::setImg(QImage image) {
  pixmap_ = QPixmap::fromImage(image);
}

void PaintWidget::setLock(bool is_lock) {
  is_locked_ = is_lock;
}

void PaintWidget::drawReport(const std::vector<double> &mesure) {
  QPainter painter{&pixmap_};
  pen_->setWidth(kReportPenWidth);
  painter.setPen(*pen_);
  double x = 70.0;
  double y = 450.0;
  double width = 30.0;
  double step = 90.0;
  double scale = 3.5;
  painter.drawText(QPoint(x - 60, y + 40), QString("Errors:"));
  int num = 1;
  for (auto& hight : mesure) {
    int result = double(hight * 100);
    painter.drawRect(QRect(x, y, width, - result * scale));
    painter.drawText(QPoint(x, y +  40), QString::number(result));
    double epoch_text_position = y - 30 - result * scale;
    painter.drawText(QPoint(x + 10, epoch_text_position), QString::number(num++));
    x += step;
  }
  pen_->setWidth(kDefaultPenWidth);
  update();
}

QImage PaintWidget::getCurrentImage() {
  int pad_width = 1;
  const int kPaintAreaSize = 512;
  const int kPixelSize = 28;
  QImage image(kPaintAreaSize, kPaintAreaSize, QImage::Format_Grayscale8);
  QImage padded_image(512 * pad_width, kPaintAreaSize * pad_width, QImage::Format_Grayscale8);
  QPainter painter(&padded_image);
  this->render(&painter);
  image = padded_image.copy(pad_width, pad_width, kPaintAreaSize, kPaintAreaSize);
  image  = image.scaled(QSize(kPixelSize, kPixelSize), Qt::IgnoreAspectRatio);
  return image;
}

}  // namespace s21

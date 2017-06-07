#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;




KalmanFilter::KalmanFilter() {}


KalmanFilter::~KalmanFilter() {}


void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}


void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = (F_ * P_ * F_.transpose()) + Q_;
}


void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_prediction = H_ * x_;
  VectorXd y = z - z_prediction;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = (P_ * H_.transpose()) * S.inverse();

  // new estimated state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}


void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float pi = 3.14159;

  float rho = sqrt(px*px + py*py);
  float theta = atan2(py, px);
  float rho_dot = (px*vx + py*vy) / rho;
  VectorXd z_prediction = VectorXd(3);
  z_prediction << rho, theta, rho_dot;

  VectorXd _y = z - z_prediction;
  // Normalize the resulting 'theta' value to +- pi
  _y(1) = fmod(_y(1), pi);

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = (P_ * H_.transpose()) * S.inverse();

  // new estimated state
  x_ = x_ + (K * _y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define lr 0.0001f
#define EPOCHS 100

double loss(double y_true, double y_pred);
double tanh_derivative(double x);
double loss_derivative(double y_true, double y_pred);
double rand_weight();


typedef struct {
  double wxh;
  double whh;
  double why;
  double h;

} RNN;

double loss(double y_true, double y_pred) {
  double l = y_true - y_pred;
  return l * l;
}

double tanh_derivative(double x) {
  return 1.0f - x * x;  // x is tanh
}

double loss_derivative(double y_true, double y_pred) {
  return 2.0f * (y_true - y_pred);
}

double rand_weight(){
  return ((double)rand() / RAND_MAX - 0.5) * 0.1; // between -0.05 and 0.05
}

int main() {
  
  srand(time(NULL));
  double X[] = {1, 2, 3, 4};
  double target[] = {2, 3, 4, 5};
  size_t n = sizeof(X) / sizeof(*X);

  RNN rnn = {.wxh = rand_weight(), .whh = rand_weight(), .why = rand_weight(), .h = 0.0};

  for (int e = 0; e < EPOCHS; e++) {
    double total_loss = 0.0f;
    rnn.h =0;
    for (int t = 0; t < n; t++) {
      double x = X[t];
      double h_prev = rnn.h;
      double h_row = rnn.wxh * x + rnn.whh * h_prev;
      rnn.h = tanh(h_row);
      double y_pred = rnn.why * rnn.h;
      double l = loss(target[t], y_pred);
      total_loss += l;
      // Gradients
      double dl_dy = loss_derivative(target[t], y_pred);
      double dy_dwhy = rnn.h;
      double dl_dwhy = dl_dy * dy_dwhy;

      double dy_dh = rnn.why;
      double dh_dhraw = tanh_derivative(rnn.h);
      double dhraw_dwhh = h_prev;
      double dl_dwhh = dl_dy * dy_dh * dh_dhraw * dhraw_dwhh;
      double dh_row_dwxh = x;

      double dl_dwxh = dl_dy * dy_dh * dh_dhraw * dh_row_dwxh;

      // updating
      rnn.why -= lr * dl_dwhy;
      rnn.whh -= lr * dl_dwhh;
      rnn.wxh -= lr * dl_dwxh;
    }

    printf("Epoch %d | Loss = %.4f\n", e, total_loss / n);
  }
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define lr 0.005f
#define EPOCHS 200
#define CLIP(x) (x > 5.0 ? 5.0 : (x < -5.0 ? -5.0 : x))

typedef struct {
  double wxh;
  double whh;
  double why;
  double bxh;
  double by;
  double* h;
  double* h_row;
  double* y_pred;

  int max_timesteps;

} RNN;

void freeRNN(RNN rnn);
double loss(double y_true, double y_pred);
double tanh_derivative(double x);
double loss_derivative(double y_true, double y_pred);
double rand_weight();
RNN create_rnn(int max_timesteps);

RNN create_rnn(int n) {
  RNN rnn = {.wxh = rand_weight(),
             .whh = rand_weight(),
             .why = rand_weight(),
             .bxh = rand_weight(),
             .by = rand_weight(),
             .h = (double*)calloc(n + 1, sizeof(double)),
             .h_row = (double*)calloc(n, sizeof(double)),
             .y_pred = (double*)calloc(n, sizeof(double)),
             .max_timesteps = n};
  return rnn;
}

void freeRNN(RNN rnn) {
  free(rnn.h);
  free(rnn.h_row);
  free(rnn.y_pred);
}

double loss(double y_true, double y_pred) {
  double l = y_true - y_pred;
  return l * l;
}

double tanh_derivative(double x) {
  return 1.0f - x * x;  // x is tanh
}

double loss_derivative(double y_true, double y_pred) {
  return 2.0f * (y_pred - y_true);
}

double rand_weight() {
  return ((double)rand() / RAND_MAX - 0.5) * 0.1;  // between -0.05 and 0.05
}

int main() {
  srand(time(NULL));
  double X[] = {1, 2, 3, 4};
  double target[] = {2, 3, 4, 5};
  size_t n = sizeof(X) / sizeof(*X);

  RNN rnn = create_rnn(n);

  for (int e = 0; e < EPOCHS; e++) {
    double total_loss = 0.0f;
    // forward pass
    rnn.h[0] = 0;
    for (int t = 0; t < n; t++) {
      double x = X[t];

      rnn.h_row[t] = rnn.wxh * x + rnn.whh * rnn.h[t] + rnn.bxh;
      rnn.h[t + 1] = tanh(rnn.h_row[t]);
      rnn.y_pred[t] = rnn.why * rnn.h[t + 1] + rnn.by;
      total_loss += loss(target[t], rnn.y_pred[t]);
    }

    // Gradients --  Backward pass (BPTT)

    double dwhy = 0.0;
    double dwhh = 0.0;
    double dwxh = 0.0;
    double dh_next = 0.0;
    double dbxh = 0.0;
    double dby = 0.0;

    for (int t = n - 1; t >= 0; t--) {
      double dl_dy = loss_derivative(target[t], rnn.y_pred[t]);
      dwhy += dl_dy * rnn.h[t + 1];
      double dy_dh = rnn.why;
      double dl_dh = dl_dy * dy_dh + dh_next;
      double dh_raw = dl_dh * tanh_derivative(rnn.h_row[t]);
      dby += dl_dy;
      dbxh += dh_raw;
      dwhh += dh_raw * rnn.h[t];
      dwxh += dh_raw * X[t];

      dh_next = dh_raw * rnn.whh;
    }

    dwhy = CLIP(dwhy);
    dwhh = CLIP(dwhh);
    dwxh = CLIP(dwxh);
    dbxh = CLIP(dbxh);
    dby = CLIP(dby);
    rnn.wxh -= lr * dwxh;
    rnn.whh -= lr * dwhh;
    rnn.why -= lr * dwhy;

    rnn.bxh -= lr * dbxh;
    rnn.by -= lr * dby;

    printf("Epoch %d | Loss = %.4f\n", e, total_loss / n);
  }

  freeRNN(rnn);






  /*
  
  
  Epoch 195 | Loss = 0.0374
  Epoch 196 | Loss = 0.0364
  Epoch 197 | Loss = 0.0354
  Epoch 198 | Loss = 0.0344
  Epoch 199 | Loss = 0.0335
  
  
  */
  return 0;
}

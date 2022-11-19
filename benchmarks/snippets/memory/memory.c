#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <float.h>

// https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/main.c

int main(int argc, char **argv)
{
  int SIZE = 800000;
  int NUMBENCH = 1000;
  int NTIMES = 1000;
  int ARRAY_ALIGNMENT = 64;

  size_t bytesPerWord = sizeof(double);
  size_t N = SIZE;
  double *a, *b, *c, *d;
  double scalar, tmp;

  a = (double *)malloc(N * bytesPerWord);
  b = (double *)malloc(N * bytesPerWord);
  c = (double *)malloc(N * bytesPerWord);
  d = (double *)malloc(N * bytesPerWord);

  for (int i = 0; i < N; i++)
  {
    a[i] = 2.0;
    b[i] = 2.0;
    c[i] = 0.5;
    d[i] = 1.0;
  }

  scalar = 3.0;

  for (int k = 0; k < NTIMES; k++)
  {
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/init.c
    for (int i = 0; i < N; i++)
    {
      a[i] = scalar;
    }
    //
    tmp = a[10];
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/sum.c
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
      sum += a[i];
    }
    /* make the compiler think this makes actually sense */
    a[10] = sum;
    //
    a[10] = tmp;
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/copy.c
    // for (int i = 0; i < N; i++)
    // {
    //   a[i] = b[i];
    // }
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/update.c
    for (int i = 0; i < N; i++)
    {
      a[i] = a[i] * scalar;
    }
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/triad.c
    // for (int i = 0; i < N; i++)
    // {
    //   a[i] = b[i] + scalar * c[i];
    // }
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/daxpy.c
    // for (int i = 0; i < N; i++)
    // {
    //   a[i] = a[i] + scalar * b[i];
    // }
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/striad.c
    // for (int i = 0; i < N; i++)
    // {
    //   a[i] = b[i] + d[i] * c[i];
    // }
    // https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/daxpy.c
    // for (int i = 0; i < N; i++)
    // {
    //   a[i] = a[i] + scalar * b[i];
    // }
  }

  // return check(a, b, c, d, N);

  double res = a[0];
  free(a);
  free(b);
  free(c);
  free(d);

  return res;
}

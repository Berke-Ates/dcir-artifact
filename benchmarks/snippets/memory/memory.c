/*
 * =======================================================================================
 *
 *      Author:   Jan Eitzinger (je), jan.eitzinger@fau.de
 *      Copyright (c) 2020 RRZE, University Erlangen-Nuremberg
 *
 *      Permission is hereby granted, free of charge, to any person obtaining a copy
 *      of this software and associated documentation files (the "Software"), to deal
 *      in the Software without restriction, including without limitation the rights
 *      to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *      copies of the Software, and to permit persons to whom the Software is
 *      furnished to do so, subject to the following conditions:
 *
 *      The above copyright notice and this permission notice shall be included in all
 *      copies or substantial portions of the Software.
 *
 *      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *      IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *      FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *      LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *      OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *      SOFTWARE.
 *
 * =======================================================================================
 */

// Source: https://github.com/RRZE-HPC/TheBandwidthBenchmark/blob/master/src/main.c
// Commit Hash: 91896f1cc0064317bdae0b71c38c29fc8434fb6d

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <float.h>

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

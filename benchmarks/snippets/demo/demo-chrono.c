#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main()
{
  clock_t start = clock();

  int *A = (int *)malloc(100000 * sizeof(int));
  int *B = (int *)malloc(100000 * sizeof(int));

  for (int i = 0; i < 100000; ++i)
  {
    A[i] = 5;

    for (int j = 0; j < 100000; ++j)
    {
      B[j] = A[i];
    }

    for (int j = 0; j < 10000; ++j)
    {
      A[j] = A[i];
    }
  }

  int res = B[0];
  free(A);
  free(B);

  clock_t diff = clock() - start;
  printf("%lf\n", ((double)diff * 1000) / CLOCKS_PER_SEC);

  return res;
}

// Source: https://github.com/FilipHa5/Performance-benchmarks/blob/master/mem_line_size.c
// Commit Hash: c3a20004b9e2de32710ab484be49b76c7f1743b1

#include <stdlib.h>
#include <string.h>

#define WORKING_SET_SIZE 409600

int main()
{
  long int i, j;
  long int array_size = WORKING_SET_SIZE;
  int stride = 1;
  int stride_max = 20;

  for (stride; stride <= stride_max; stride++)
  {
    double *array = malloc(array_size * sizeof(double));

    for (i = 0; i < array_size; i += 8)
    {
      array[i]++;
    }

    for (i = 0; i < 1000; i++)
    {
      for (j = 0; j < array_size; j += stride)
      {
        array[j]++;
      }
    }

    free(array);
  }

  return 0;
}

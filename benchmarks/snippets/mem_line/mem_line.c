#include <stdlib.h>
#include <string.h>

// https://github.com/FilipHa5/Performance-benchmarks/blob/master/mem_line_size.c

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

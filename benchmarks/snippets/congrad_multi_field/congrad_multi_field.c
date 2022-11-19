// https://github.com/milc-qcd/milc_qcd/blob/master/arb_overlap/congrad_multi_field.c
#include <stdlib.h>
#include <stdio.h>

typedef float Real;

int main()
{

  // For benchmark
  int Norder = 10000;
  int MaxCG = 1000;
  int total_iters = 0;
  Real *shift = (Real *)malloc(Norder * sizeof(Real));
  for (int i = 0; i < Norder; i++)
    shift[i] = i;

  // -----------------

  int N_iter;
  register int i, j;
  // register site *s;

  int iteration; /* counter for iterations */
  double rsq, rsqnew, source_norm, rsqmin, rsqstop;
  // complex ctmp;
  double c1, c2, cd;
  double *zeta_i, *zeta_im1, *zeta_ip1;
  double *beta_i, *beta_im1, *alpha;
  int *converged;
  double rsqj;
  Real floatvar, floatvar2, *floatvarj, *floatvark; /* SSE kluge */

  zeta_i = (double *)malloc(Norder * sizeof(double));
  zeta_im1 = (double *)malloc(Norder * sizeof(double));
  zeta_ip1 = (double *)malloc(Norder * sizeof(double));
  beta_i = (double *)malloc(Norder * sizeof(double));
  beta_im1 = (double *)malloc(Norder * sizeof(double));
  alpha = (double *)malloc(Norder * sizeof(double));

  floatvarj = (Real *)malloc(Norder * sizeof(Real));
  floatvark = (Real *)malloc(Norder * sizeof(Real));

  converged = (int *)malloc(Norder * sizeof(int));

  // For benchmark
  iteration = 0;
  rsq = 1;
  rsqnew = 1;
  source_norm = 1;
  rsqmin = 1;
  rsqstop = 1;
  c1 = 1;
  c2 = 1;
  cd = 1;
  for (i = 0; i < Norder; i++)
  {
    // zeta_i[i] = 0;
    // zeta_im1[i] = 0;
    zeta_ip1[i] = 0;
    beta_i[i] = 0;
    // beta_im1[i] = 0;
    // alpha[i] = 0;
    floatvarj[i] = 0;
    floatvark[i] = 0;
  }
  // -----------------

  for (i = 0; i < Norder; i++)
    converged[i] = 0;

  for (j = 0; j < Norder; j++)
  {
    zeta_im1[j] = zeta_i[j] = 1.0;
    alpha[j] = 0.0;
    beta_im1[j] = 1.0;
  }

  // for (N_iter = 0; N_iter < MaxCG //&& rsq > rsqstop
  //      ;
  //      ++N_iter)
  // {

  iteration++;
  total_iters++;

  /* beta_i[0]= - (r,r)/(pm,Mpm)  */
  cd = 0.0;

  // For benchmark
  cd = 1;
  // -----------------

  beta_i[0] = -rsq / cd;

  /* beta_i(sigma), zeta_ip1(sigma) */

  zeta_ip1[0] = 1.0;
  for (j = 1; j < Norder; j++)
  {
    if (converged[j] == 0)
    {
      zeta_ip1[j] = zeta_i[j] * zeta_im1[j] * beta_im1[0];
      c1 = beta_i[0] * alpha[0] * (zeta_im1[j] - zeta_i[j]);
      c2 = zeta_im1[j] * beta_im1[0] * (1.0 - (shift[j] - shift[0]) * beta_i[0]);
      zeta_ip1[j] /= c1 + c2;

      beta_i[j] = beta_i[0] * zeta_ip1[j] / zeta_i[j];
    }
  }

  /* psim[j] = psim[j] - beta[j]*pm[j]  */
  floatvar = -(Real)beta_i[0];
  for (j = 1; j < Norder; j++)
    if (converged[j] == 0)
      floatvarj[j] = -(Real)beta_i[j];

  /* r = r + beta[0]*mp */
  floatvar = (Real)beta_i[0];

  /* alpha_ip1[j] */
  rsqnew = 0.0;
  alpha[0] = rsqnew / rsq;

  /*alpha_ip11--note shifted indices wrt eqn 2.43! */

  for (j = 1; j < Norder; j++)
  {
    if (converged[j] == 0)
    {
      alpha[j] = alpha[0] * zeta_ip1[j] * beta_i[j] / zeta_i[j] / beta_i[0];
    }
  }

  /* pm[j]=zeta_ip1[j]r +alpha[j]pm[j] */

  floatvar = (Real)zeta_ip1[0];
  floatvar2 = (Real)alpha[0];
  for (j = 1; j < Norder; j++)
  {
    floatvarj[j] = (Real)zeta_ip1[j];
    floatvark[j] = (Real)alpha[j];
  }

  /* test for convergence */
  rsq = rsqnew;
  for (j = 1; j < Norder; j++)
  {
    if (converged[j] == 0)
    {
      rsqj = rsq * zeta_ip1[j] * zeta_ip1[j];
      if (rsqj <= rsqstop)
      {
        converged[j] = 1;
        /*
        node0_printf(" vector %d converged in %d steps %e\n",
         j,N_iter,rsqj);
        */
      }
    }
  }
  /*
        if(this_node==0 && ((N_iter / 1)*1==N_iter) ){
  printf("iter %d residue %e\n",N_iter,
      (double)(rsq));
  fflush(stdout);}
  */
  /* and scroll scalars */
  for (j = 0; j < Norder; j++)
  {
    if (converged[j] == 0)
    {
      beta_im1[j] = beta_i[j];
      zeta_im1[j] = zeta_i[j];
      zeta_i[j] = zeta_ip1[j];
    }
  }
  // }

  // free(pm);
  // free(rm);

  // free(mpm);
  // free(pm0);

  free(zeta_i);
  free(zeta_ip1);
  free(zeta_im1);
  free(beta_im1);
  free(beta_i);
  free(alpha);
  free(converged);

  free(floatvarj);
  free(floatvark);

  free(shift);

  return rsq;
}

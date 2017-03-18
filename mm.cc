#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "timer.c"
#include <x86intrin.h>

#define N_ 4096
#define K_ 4096
#define M_ 4096

typedef double dtype;

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR\n"); else printf("SUCCESS\n");
}

void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

void cb_helper(dtype *C, dtype *A, dtype *B, int M, int K, int i, int j, int k, int blockSize)
{
	dtype sum;
	for (int x = i * blockSize; x < (i + 1) * blockSize; x++)
	{
		for (int y = j * blockSize; y < (j + 1) * blockSize; y++)
		{
			sum = 0;
			for (int z = k * blockSize; z < (k + 1) * blockSize; z++)
			{
				sum += A[x * K + z] * B[z * M + y];
			}
			C[x * M + y] += sum;
		}
	}
}

double mm_cb (dtype *C_cb, dtype *A, dtype *B, int N, int K, int M, int blockSize)
{
	double gflops = 0.0;
	double temp = 0.0;

	for (int i = 0; i < N/blockSize; i++)
	{
		for (int j = 0; j < M/blockSize; j++)
		{
			for (int k = 0; k < K/blockSize; k++)
			{
				cb_helper(C_cb, A, B, M, K, i, j, k, blockSize);
			}
		}
	}
	return gflops;
  /* =======================================================+ */
  /* Implement your own cache-blocked matrix-matrix multiply  */
  /* =======================================================+ */
}

void sv_helper(dtype *C, dtype *A, dtype *B, int M, int K, int i, int j, int k, int blockSize)
{
	int fit = 128 / (sizeof(dtype) * 8);
	dtype sum;
	__m128 sumSIMD;

	if (fit < blockSize)
	{
		for (int x = i * blockSize; x < (i * blockSize + fit); x++)
		{
			for (int y = j * blockSize; y < (j * blockSize + fit); y++)
			{
				sum = 0;
				for (int z = k * blockSize; z < (k * blockSize + fit); z++)
				{
					sum += A[x * K + z] * B[z * M + y];
				}
				C[x * M + y] += sum;
			}
		}
	}
	else
	{
		for (int x = i * blockSize; x < ((i + 1) * blockSize)/fit; x += fit)
		{
			for (int y = j * blockSize; y < ((j + 1) * blockSize)/fit; y += fit)
			{
				sumSIMD = _mm_set_ps1(0);
				for (int z = k * blockSize; z < ((k + 1) * blockSize)/fit; z += fit)
				{
					/* sum += A[x * K + z] * B[z * M + y]; */
					// TODO: Do magic here
				}
				C[x * M + y] += sum;
			}
		}
	}

}

double mm_sv (dtype *C_sv, dtype *A, dtype *B, int N, int K, int M, int blockSize)
{
	double gflops = 0.0;
	double temp = 0.0;

	for (int i = 0; i < N/blockSize; i++)
	{
		for (int j = 0; j < M/blockSize; j++)
		{
			for (int k = 0; k < K/blockSize; k++)
			{
				sv_helper(C_sv, A, B, M, K, i, j, k, blockSize);
			}
		}
	}
	return gflops;
  /* =======================================================+ */
  /* Implement your own SIMD-vectorized matrix-matrix multiply  */
  /* =======================================================+ */
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M, subBlock;
  double gflops = 0.0;
  double temp = 0.0;

 /* if(argc == 4) {
    N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);		
    printf("N: %d K: %d M: %d\n", N, K, M);
  }*/ 
  if(argc == 5){
	N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);	
	subBlock = atoi (argv[4]);	
    printf("N: %d K: %d M: %d Sub Block size: %d\n", N, K, M, subBlock);
  }else {
    N = N_;
    K = K_;
    M = M_;
	// TODO: Probably should initialize subBlock size too
    printf("N: %d K: %d M: %d\n", N, K, M);	
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds\n\n", t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  temp = mm_cb (C_cb, A, B, N, K, M, subBlock);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds\n", t);
  gflops = temp/t;
  printf("temp = %f - Gflops = %f\n", temp, gflops);
  
  /* verify answer */
  verify (C_cb, C, N, M);

  printf("SIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  temp = mm_sv (C_sv, A, B, N, K, M, subBlock);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}

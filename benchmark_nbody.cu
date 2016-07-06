#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define TAM_BLOCO 256
#define FATOR 1e-9f



void setaPosicao(float *data, int corpos) {
	for (int i = 0; i < 4 * corpos; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}
void zeraAceleracao(float *data, int corpos) {
	for (int i = 4 * corpos; i < 8 * corpos; i++) {
		data[i] = 0.0f;
	}
}


__device__ float3 tile_aceleracao(float4 minhaPosicao, float3 acel){
	int i;
	//Cria link com os dados da memoria compartilhada
	extern __shared__ float4 posCompartilhada[];
	//Faz o calculo em relação a todos os corpos do bloco
	for (i = 0; i < TAM_BLOCO; i++) {
		float3 r;
		//calcula a distancia entre os corpos
		r.x = posCompartilhada[i].x - minhaPosicao.x;
		r.y = posCompartilhada[i].y - minhaPosicao.y;
		r.z = posCompartilhada[i].z - minhaPosicao.z;
		//modulo da distancia ao quadrado + fator de amaciamento
		float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + FATOR;
		//eleva a 6 potencia
		float dist_a_6 = distSqr * distSqr * distSqr;
		//tira a raiz quadrada e inverte o resultado
		float invDistanciaCubo = 1.0f / sqrtf(dist_a_6);
		//multiplica pela massa do corpo
		float s = posCompartilhada[i].w * invDistanciaCubo;
		//multiplica pelo vetor distancia e realiza mais 1 soma do 
		//somatorio
		acel.x += r.x * s;
		acel.y += r.y * s;
		acel.z += r.z * s;
	}
	return acel;
}

__global__ void calcula_aceleracao(void *devX, void *devA, int ncorpos, int nBlocos){
	//Cria link com os dados da memoria compartilhada
	extern __shared__ float4 posCompartilhada[];
	//Cria vetor com as posicoes na memoria global
	float4 *globalX = (float4 *)devX;
	//Cria vetor com as aceleracoes na memoria global
	float4 *globalA = (float4 *)devA;
	float4 minhaPosicao;
	int i, tile;
	float3 acc = { 0.0f, 0.0f, 0.0f };
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	minhaPosicao = globalX[gtid];
	//Faz o calculo em relação a todos os blocos da grid
	for (tile = 0; tile < gridDim.x; tile++) {
		int idx = tile * blockDim.x + threadIdx.x;
		//Carrega as posicoes da memoria global para a compartilhada
		posCompartilhada[threadIdx.x] = globalX[idx];
		__syncthreads();
		acc = tile_aceleracao(minhaPosicao, acc);
		__syncthreads();
	}
	// salva o resultado na memoria global para o passo da integracao.  
	float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
	globalA[gtid] = acc4;
}


int main(const int argc, const char** argv) {
	int i;//i=aplicacao corrente
	float score = 0;
	int nquad;
	int nBodies, nIters;
	float temp0, temp1, temp2, temp3, temp4, temp5;
	int qntCores, frequenciaGpu;
	double eficiencia;

	for (i = 0; i < 6; i++){

		if (i == 0){
			nBodies = 1024;
			nIters = 1000;
		}
		else if (i == 1){
			nBodies = 2048;
			nIters = 1000;
		}
		else if (i == 2){
			nBodies = 4096;
			nIters = 1000;
		}
		else if (i == 3){
			nBodies = 8192;
			nIters = 1000;
		}
		else if (i == 4){
			nBodies = 16384;
			nIters = 500;
		}
		else if (i == 5){
			nBodies = 32768;
			nIters = 250;
		}
		nquad = nBodies*nBodies;

		const float dt = 0.01f; // tempo

		int bytes = 2 * nBodies*sizeof(float4);
		float *buf = (float*)malloc(bytes);
		float4 *bp = (float4*)buf;
		float4 *ba = ((float4*)buf) + nBodies;

		setaPosicao(buf, nBodies); // Inicia posicao e acel
		zeraAceleracao(buf, nBodies);

		float *d_buf;
		cudaMalloc(&d_buf, bytes);
		float4 *d_bp = (float4*)d_buf;
		float4 *d_ba = ((float4*)d_buf) + nBodies;

		int nBlocks = (nBodies + TAM_BLOCO - 1) / TAM_BLOCO;
		double totalTime = 0.0;

		for (int iter = 1; iter <= nIters; iter++) {
			StartTimer();

			//Passa os dados da memoria para a GPU
			cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
			//Calcula a aceleracao de cada corpo (cada thread cuida de um corpo)
			//O terceiro parametro na chamada da funcao em cuda e o tamanho da memoria compartilhada
			calcula_aceleracao << <nBlocks, TAM_BLOCO, (TAM_BLOCO*sizeof(float4)) >> >(d_bp, d_ba, nBodies, nBlocks);
			//Passa os dados da GPU para a memoria
			cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

			//Faz a integracao das posicoes
			for (int i = 0; i < nBodies; i++) {
				bp[i].x += ba[i].x*dt*dt;
				bp[i].y += ba[i].y*dt*dt;
				bp[i].z += ba[i].z*dt*dt;


			}

			const double tElapsed = GetTimer() / 1000.0;
			if (iter > 1) {
				totalTime += tElapsed;
			}

			printf("Ciclo %d Iteracao %d: %.3f segundos\n", i, iter, tElapsed);

		}
		double avgTime = totalTime / (double)(nIters - 1);
		score += (nquad / totalTime);



		free(buf);
		cudaFree(d_buf);

		if (i == 0) temp0 = totalTime;
		if (i == 1) temp1 = totalTime;
		if (i == 2) temp2 = totalTime;
		if (i == 3) temp3 = totalTime;
		if (i == 4) temp4 = totalTime;
		if (i == 5) temp5 = totalTime;
	}
	//qntCores = 384;
	//frequenciaGpu = 980;
	printf("\nInsira a quantidade de cuda cores presentes em sua GPU\n");
	scanf("%d", &qntCores);
	printf("\nInsira a frequencia de sua GPU em MHz\n");
	scanf("%d", &frequenciaGpu);
	eficiencia = (1.0 / qntCores)*(500.0 / frequenciaGpu)*score;

	printf("\n Score %f\n", score);
	printf("\n Eficiencia %f\n", eficiencia);

	printf("\nTempo ciclo 0 %f", temp0);
	printf("\nTempo ciclo 1 %f", temp1);
	printf("\nTempo ciclo 2 %f", temp2);
	printf("\nTempo ciclo 3 %f", temp3);
	printf("\nTempo ciclo 4 %f", temp4);
	printf("\nTempo ciclo 5 %f\n", temp5);
}

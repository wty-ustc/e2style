//
// MEX-File: dv_extract
// Usage:
// uint8 vector msg = dv_extract(uint8 vector stego, int scalar msgLength, int scalar constrLength = 10)

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#include <cstdlib>
#include <cstring>
#include <cmath>
#include "mex.h"
#include "common.h"

#define MATLAB_in_stego 0
#define MATLAB_in_msgLength 1
#define MATLAB_in_constrLength 2

#define MATLAB_out_msg 0

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int matrixheight, vectorlength, i, j, k, index, index2, syndromelength, matrixwidth, size[2], base, height;
	
	u8 *vector, *syndrome, *closestData, *ssedone, *stepdown, *binmat[2], *msgData;
	int *matrices, *widths;

	mxArray *msg;

	if(nlhs == 0)
		return;

	if(nrhs < 2) {
		mexErrMsgTxt("Too few input parameters (2 or 3 expected)");
		return;
	}
	
	if(!mxIsUint8(prhs[MATLAB_in_stego])) {
		mexErrMsgTxt("The stego vector must be of type uint8.");
		return;
	}
	vector = (u8*)mxGetPr(prhs[MATLAB_in_stego]);
	vectorlength = (int)mxGetM(prhs[MATLAB_in_stego]);
	
	syndromelength = (int)mxGetScalar(prhs[MATLAB_in_msgLength]);

	if(nrhs > 2)
		matrixheight = (int)mxGetScalar(prhs[MATLAB_in_constrLength]);
	else
		matrixheight = 10;

	// end of matlab interface

	height = matrixheight;

	if(matrixheight > 31) {
		mexErrMsgTxt("Submatrix height must not exceed 31.");
		return;
	}

	{
		double invalpha;
		int shorter, longer, worm;
		u32 *columns[2];

		matrices = (int *)malloc(syndromelength * sizeof(int));
		widths = (int *)malloc(syndromelength * sizeof(int));

		invalpha = (double)vectorlength / syndromelength;
		if(invalpha < 1) {
			mexErrMsgTxt("The message cannot be longer than the cover object.\n");
			return;
		}
		shorter = (int)floor(invalpha);
		longer = (int)ceil(invalpha);
		if((columns[0] = getMatrix(shorter, matrixheight)) == NULL)
			return;
		if((columns[1] = getMatrix(longer, matrixheight)) == NULL)
			return;
		worm = 0;
		for(i = 0; i < syndromelength; i++) {
			if(worm + longer <= (i + 1) * invalpha + 0.5) {
				matrices[i] = 1;
				widths[i] = longer;
				worm += longer;
			} else {
				matrices[i] = 0;
				widths[i] = shorter;
				worm += shorter;
			}
		}
		binmat[0] = (u8*)malloc(shorter * matrixheight * sizeof(u8));
		binmat[1] = (u8*)malloc(longer * matrixheight * sizeof(u8));
		for(i = 0, index = 0; i < shorter; i++) {
			for(j = 0; j < matrixheight; j++, index++) {
				binmat[0][index] = (columns[0][i] & (1 << j)) ? 1 : 0;
			}
		}
		for(i = 0, index = 0; i < longer; i++) {
			for(j = 0; j < matrixheight; j++, index++) {
				binmat[1][index] = (columns[1][i] & (1 << j)) ? 1 : 0;
			}
		}
		free(columns[0]);
		free(columns[1]);
	}

	size[0] = syndromelength;
	size[1] = 1;
	
	msg = mxCreateNumericArray(2, size, mxUINT8_CLASS, mxREAL);
	msgData = (u8*)mxGetPr(msg);

	for(i = 0; i < syndromelength; i++) {
		msgData[i] = 0;
	}

	for(index = 0, index2 = 0; index2 < syndromelength; index2++) {
		for(k = 0, base = 0; k < widths[index2]; k++, index++, base += matrixheight) {
			if(vector[index]) {
				for(i = 0; i < height; i++) {
					msgData[index2 + i] ^= binmat[matrices[index2]][base + i];
				}
			}
		}
		if(syndromelength - index2 <= matrixheight)
			height--;
	}

	free(matrices);
	free(widths);
	free(binmat[0]);
	free(binmat[1]);

	plhs[MATLAB_out_msg] = msg;

	return;
}

//
// MEX-File: dv_embed
// Usage:
// [double scalar price, uint8 vector stego] = dv_embed(uint8 vector cover, uint8 vector message, uint8/double vector pricevec, int scalar constrLength = 10)

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <limits>
#include <emmintrin.h>
#include "mex.h"
#include "common.h"

#define MATLAB_in_cover 0
#define MATLAB_in_message 1
#define MATLAB_in_pricevec 2
#define MATLAB_in_constrLength 3

#define MATLAB_out_price 0
#define MATLAB_out_stego 1

void *aligned_malloc(unsigned int bytes, int align)
{
	int shift;
	char *temp = (char *)malloc(bytes + align);

	if(temp == NULL)
		return temp;
	shift = align - (int)(((unsigned long long)temp) & (align - 1));
	temp = temp + shift;
	temp[-1] = shift;
	return (void *)temp;
}

void aligned_free(void *vptr)
{
	char *ptr = (char *)vptr;
	free(ptr - ptr[-1]);
	return;
}

inline __m128i maxLessThan255(const __m128i v1, const __m128i v2)
{
	register __m128i mask = _mm_set1_epi32(0xffffffff);
	return _mm_max_epu8(_mm_andnot_si128(_mm_cmpeq_epi8(v1, mask), v1), _mm_andnot_si128(_mm_cmpeq_epi8(v2, mask), v2));
}

inline u8 max16B(__m128i maxp)
{
	u8 mtemp[4];
	maxp = _mm_max_epu8(maxp, _mm_srli_si128(maxp, 8));
	maxp = _mm_max_epu8(maxp, _mm_srli_si128(maxp, 4));
	*((int*)mtemp) = _mm_cvtsi128_si32(maxp);
	if(mtemp[2] > mtemp[0])
		mtemp[0] = mtemp[2];
	if(mtemp[3] > mtemp[1])
		mtemp[1] = mtemp[3];
	if(mtemp[1] > mtemp[0])
		return mtemp[1];
	else 
		return mtemp[0];
}

inline u8 min16B(__m128i minp)
{
	u8 mtemp[4];
	minp = _mm_min_epu8(minp, _mm_srli_si128(minp, 8));
	minp = _mm_min_epu8(minp, _mm_srli_si128(minp, 4));
	*((int*)mtemp) = _mm_cvtsi128_si32(minp);
	if(mtemp[2] < mtemp[0])
		mtemp[0] = mtemp[2];
	if(mtemp[3] < mtemp[1])
		mtemp[1] = mtemp[3];
	if(mtemp[1] < mtemp[0])
		return mtemp[1];
	else 
		return mtemp[0];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int matrixheight, vectorlength, height, i, j, k, l, index, index2, parts, m, sseheight, altm, pathindex, syndromelength, matrixwidth;
	u32 column, colmask, state;
	double totalprice;
	bool usefloat;

	void *temp, *pricevectorv;
	u8 *vector, *syndrome, *closestData, *ssedone, *stepdown;
	u32 *path, *columns[2];
	int *matrices, *widths;

	if(nrhs < 3) {
		mexErrMsgTxt("Too few input parameters (3 or 4 expected)");
		return;
	}
	
	if(!mxIsUint8(prhs[MATLAB_in_cover])) {
		mexErrMsgTxt("The input vector must be of type uint8.");
		return;
	}
	vector = (u8*)mxGetPr(prhs[MATLAB_in_cover]);
	vectorlength = (int)mxGetM(prhs[MATLAB_in_cover]);
	
	if(!mxIsUint8(prhs[MATLAB_in_message])) {
		mexErrMsgTxt("The syndrome must be of type uint8.");
		return;
	}
	syndrome = (u8*)mxGetPr(prhs[MATLAB_in_message]);
	syndromelength = (int)mxGetM(prhs[MATLAB_in_message]);

	if(!mxIsDouble(prhs[MATLAB_in_pricevec])) {
		if(!mxIsUint8(prhs[MATLAB_in_pricevec])) {
			mexErrMsgTxt("The price vector must be of type double or uint8.");
			return;
		} else {
			usefloat = false;
		}
	} else {
		usefloat = true;
	}
	pricevectorv = (void*)mxGetPr(prhs[MATLAB_in_pricevec]);

	if(nrhs > 3)
		matrixheight = (int)mxGetScalar(prhs[MATLAB_in_constrLength]);
	else
		matrixheight = 10;

	// end of MATLAB interface

	if(matrixheight > 31) {
		mexErrMsgTxt("Submatrix height must not exceed 31.");
		return;
	}

	height = 1 << matrixheight;
	colmask = height - 1;
	height = (height + 31) & (~31);

	parts = height >> 5;
	
	if(nlhs > 1) {
		path = (u32*)malloc(vectorlength * parts * sizeof(u32));
		if(path == NULL) {
			char errmsg[100];
			sprintf(errmsg, "Not enough memory (%u byte array could not be allocated).\n", vectorlength * parts * sizeof(u32));
			mexErrMsgTxt(errmsg);
			return;
		}
		pathindex = 0;
	}

	{
		int shorter, longer, worm;
		double invalpha;

		matrices = (int *)malloc(syndromelength * sizeof(int));
		widths = (int *)malloc(syndromelength * sizeof(int));

		invalpha = (double)vectorlength / syndromelength;
		if(invalpha < 1) {
			free(matrices);
			free(widths);
			if(nlhs > 1)
				free(path);
			mexErrMsgTxt("The message cannot be longer than the cover object.\n");
			return;
		}
		if(invalpha < 2) {
			mexWarnMsgTxt("The relative payload is greater than 1/2. This may result in poor embedding efficiency.\n");
		}
		shorter = (int)floor(invalpha);
		longer = (int)ceil(invalpha);
		if((columns[0] = getMatrix(shorter, matrixheight)) == NULL) {
			free(matrices);
			free(widths);
			if(nlhs > 1)
				free(path);
			return;
		}
		if((columns[1] = getMatrix(longer, matrixheight)) == NULL) {
			free(columns[0]);
			free(matrices);
			free(widths);
			if(nlhs > 1)
				free(path);
			return;
		}
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
	}

	if(usefloat) {
	/*
		SSE FLOAT VERSION
	*/
		int pathindex8 = 0;
		int shift[2] = {0, 4};
		u8 mask[2] = {0xf0, 0x0f};
		float *prices;
		u8 *path8 = (u8*)path;
		double *pricevector = (double*)pricevectorv;
		double total = 0;
		double inf = std::numeric_limits<double>::infinity();

		sseheight = height >> 2;
		ssedone = (u8*)malloc(sseheight * sizeof(u8));
		prices = (float*)aligned_malloc(height * sizeof(float), 16);

		{
			__m128 fillval = _mm_set1_ps(inf);
			for(i = 0; i < height; i+= 4) {
				_mm_store_ps(&prices[i], fillval);
				ssedone[i >> 2] = 0;
			}
		}

		prices[0] = 0.0f;

		for(index = 0, index2 = 0; index2 < syndromelength; index2++) {
			register __m128 c1, c2;

			for(k = 0; k < widths[index2]; k++, index++) {
				column = columns[matrices[index2]][k] & colmask;

				if(vector[index] == 0) {
					c1 = _mm_setzero_ps();
					c2 = _mm_set1_ps((float)pricevector[index]);
				} else {
					c1 = _mm_set1_ps((float)pricevector[index]);
					c2 = _mm_setzero_ps();
				}

				total += pricevector[index];

				for(m = 0; m < sseheight; m++) {
					if(!ssedone[m]) {
						register __m128 v1, v2, v3, v4;
						altm = (m ^ (column >> 2));
						v1 = _mm_load_ps(&prices[m << 2]);
						v2 = _mm_load_ps(&prices[altm << 2]);
						v3 = v1;
						v4 = v2;
						ssedone[m] = 1;
						ssedone[altm] = 1;
						switch(column & 3) {
						case 0:
							break;
						case 1:
							v2 = _mm_shuffle_ps(v2, v2, 0xb1);
							v3 = _mm_shuffle_ps(v3, v3, 0xb1);
							break;
						case 2:
							v2 = _mm_shuffle_ps(v2, v2, 0x4e);
							v3 = _mm_shuffle_ps(v3, v3, 0x4e);
							break;
						case 3:
							v2 = _mm_shuffle_ps(v2, v2, 0x1b);
							v3 = _mm_shuffle_ps(v3, v3, 0x1b);
							break;
						}
						v1 = _mm_add_ps(v1, c1);
						v2 = _mm_add_ps(v2, c2);
						v3 = _mm_add_ps(v3, c2);
						v4 = _mm_add_ps(v4, c1);

						v1 = _mm_min_ps(v1, v2);
						v4 = _mm_min_ps(v3, v4);
											
						_mm_store_ps(&prices[m << 2], v1);
						_mm_store_ps(&prices[altm << 2], v4);

						if(nlhs > 1) {
							v2 = _mm_cmpeq_ps(v1, v2);
							v3 = _mm_cmpeq_ps(v3, v4);
							path8[pathindex8 + (m >> 1)] = (path8[pathindex8 + (m >> 1)] & mask[m & 1]) | (_mm_movemask_ps(v2) << shift[m & 1]);
							path8[pathindex8 + (altm >> 1)] = (path8[pathindex8 + (altm >> 1)] & mask[altm & 1]) | (_mm_movemask_ps(v3) << shift[altm & 1]);
						}
					}
				}

				for(i = 0; i < sseheight; i++) {
					ssedone[i] = 0;
				}

				pathindex += parts;
				pathindex8 += parts << 2;
			}

			if(syndrome[index2] == 0) {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) {
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0x88));
				}
			} else {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) {
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0xdd));
				}
			}

			if(syndromelength - index2 <= matrixheight)
				colmask >>= 1;

			{
				register __m128 fillval = _mm_set1_ps(inf);
				for(l >>= 2; l < sseheight; l++) {
					_mm_store_ps(&prices[l << 2], fillval);
				}
			}
		}

		totalprice = prices[0];

		aligned_free(prices);
		free(ssedone);

		if(totalprice >= total) {
			free(matrices);
			free(widths);
			free(columns[0]);
			free(columns[1]);
			if(nlhs > 1)
				free(path);
			mexErrMsgTxt("The syndrome is not in the range of the syndrome matrix.\n");
			return;
		}
	} else {
	/*
		SSE UINT8 VERSION
	*/
		int pathindex16 = 0, subprice = 0;
		u8 maxc = 0, minc = 0;
		u8 *prices, *pricevector = (u8*)pricevectorv;
		u16 *path16 = (u16 *)path;
		__m128i *prices16B;

		sseheight = height >> 4;
		ssedone = (u8*)malloc(sseheight * sizeof(u8));
		prices = (u8*)aligned_malloc(height * sizeof(u8), 16);
		prices16B = (__m128i*)prices;

		{
			__m128i napln = _mm_set1_epi32(0xffffffff);
			for(i = 0; i < sseheight; i++) {
				_mm_store_si128(&prices16B[i], napln);
				ssedone[i] = 0;
			}
		}

		prices[0] = 0;

		for(index = 0, index2 = 0; index2 < syndromelength; index2++) {
			register __m128i c1, c2, maxp, minp;

			if((u32)maxc + pricevector[index] >= 254) {
				aligned_free(path);
				free(ssedone);
				free(matrices);
				free(widths);
				free(columns[0]);
				free(columns[1]);
				if(nlhs > 1)
					free(path);
				mexErrMsgTxt("Price vector limit exceeded.");
				return;
			}

			for(k = 0; k < widths[index2]; k++, index++) {
				column = columns[matrices[index2]][k] & colmask;

				if(vector[index] == 0) {
					c1 = _mm_setzero_si128();
					c2 = _mm_set1_epi8(pricevector[index]);
				} else {
					c1 = _mm_set1_epi8(pricevector[index]);
					c2 = _mm_setzero_si128();
				}

				minp = _mm_set1_epi8(-1);
				maxp = _mm_setzero_si128();

				for(m = 0; m < sseheight; m++) {
					if(!ssedone[m]) {
						register __m128i v1, v2, v3, v4;
						altm = (m ^ (column >> 4));
						v1 = _mm_load_si128(&prices16B[m]);
						v2 = _mm_load_si128(&prices16B[altm]);
						v3 = v1;
						v4 = v2;
						ssedone[m] = 1;
						ssedone[altm] = 1;
						if(column & 8) {
							v2 = _mm_shuffle_epi32(v2, 0x4e);
							v3 = _mm_shuffle_epi32(v3, 0x4e);
						}
						if(column & 4) {
							v2 = _mm_shuffle_epi32(v2, 0xb1);
							v3 = _mm_shuffle_epi32(v3, 0xb1);
						}
						if(column & 2) {
							v2 = _mm_shufflehi_epi16(v2, 0xb1);
							v3 = _mm_shufflehi_epi16(v3, 0xb1);
							v2 = _mm_shufflelo_epi16(v2, 0xb1);
							v3 = _mm_shufflelo_epi16(v3, 0xb1);
						}
						if(column & 1) {
							v2 = _mm_or_si128(_mm_srli_epi16(v2, 8), _mm_slli_epi16(v2, 8));
							v3 = _mm_or_si128(_mm_srli_epi16(v3, 8), _mm_slli_epi16(v3, 8));
						}
						v1 = _mm_adds_epu8(v1, c1);
						v2 = _mm_adds_epu8(v2, c2);
						v3 = _mm_adds_epu8(v3, c2);
						v4 = _mm_adds_epu8(v4, c1);

						v1 = _mm_min_epu8(v1, v2);
						v4 = _mm_min_epu8(v3, v4);
											
						_mm_store_si128(&prices16B[m], v1);
						_mm_store_si128(&prices16B[altm], v4);

						minp = _mm_min_epu8(minp, _mm_min_epu8(v1, v4));
						maxp = _mm_max_epu8(maxp, maxLessThan255(v1, v4));
						
						if(nlhs > 1) {
							v2 = _mm_cmpeq_epi8(v1, v2);
							v3 = _mm_cmpeq_epi8(v3, v4);
							path16[pathindex16 + m] = (u16)_mm_movemask_epi8(v2);
							path16[pathindex16 + altm] = (u16)_mm_movemask_epi8(v3);
						}
					}
				}

				maxc = max16B(maxp);
				minc = min16B(minp);

				maxc -= minc;
				subprice += minc;
				{
					register __m128i mask = _mm_set1_epi32(0xffffffff);
					register __m128i m = _mm_set1_epi8(minc);
					for(i = 0; i < sseheight; i++) {
						register __m128i res;
						register __m128i pr = prices16B[i];
						res = _mm_andnot_si128(_mm_cmpeq_epi8(pr, mask), m);
						prices16B[i] = _mm_sub_epi8(pr, res);
						ssedone[i] = 0;
					}
				}

				pathindex += parts;
				pathindex16 += parts << 1;
			}

			{
				register __m128i mask = _mm_set1_epi32(0x00ff00ff);

				if(minc == 255) {
					aligned_free(path);
					free(ssedone);
					free(matrices);
					free(widths);
					free(columns[0]);
					free(columns[1]);
					if(nlhs > 1)
						free(path);
					mexErrMsgTxt("The syndrome is not in the syndrome matrix range.\n");
					return;
				}

				if(syndrome[index2] == 0) {
					for(i = 0, l = 0; i < sseheight; i += 2, l++) {
						_mm_store_si128(&prices16B[l], _mm_packus_epi16(_mm_and_si128(_mm_load_si128(&prices16B[i]), mask), 
							_mm_and_si128(_mm_load_si128(&prices16B[i + 1]), mask)));
					}
				} else {
					for(i = 0, l = 0; i < sseheight; i += 2, l++) {
						_mm_store_si128(&prices16B[l], _mm_packus_epi16(_mm_and_si128(_mm_srli_si128(_mm_load_si128(&prices16B[i]), 1), mask), 
							_mm_and_si128(_mm_srli_si128(_mm_load_si128(&prices16B[i + 1]), 1), mask)));
					}
				}

				if(syndromelength - index2 <= matrixheight)
					colmask >>= 1;

				register __m128i fillval = _mm_set1_epi32(0xffffffff);
				for(; l < sseheight; l++)
					_mm_store_si128(&prices16B[l], fillval);
			}
		}
			
		totalprice = subprice + prices[0];
		
		aligned_free(prices);
		free(ssedone);
	}
	
	if(nlhs > 1) {
		int size[2];
		u8 *closestData;
		mxArray *closest;

		size[0] = vectorlength;
		size[1] = 1;
		
		closest = mxCreateNumericArray(2, size, mxUINT8_CLASS, mxREAL);
		closestData = (u8*)mxGetPr(closest);

		pathindex -= parts;
		index--;
		index2--;
		state = 0;

		int h = syndromelength;
		state = 0;
		colmask = 0;
		for(; index2 >= 0; index2--) {
			for(k = widths[index2] - 1; k >= 0; k--, index--) {
				if(k == widths[index2] - 1) {
					state = (state << 1) | syndrome[index2];
					if(syndromelength - index2 <= matrixheight)
						colmask = (colmask << 1) | 1;
				}

				if(path[pathindex + (state >> 5)] & (1 << (state & 31))) {
					closestData[index] = 1;
					state = state ^ (columns[matrices[index2]][k] & colmask);
				} else {
					closestData[index] = 0;
				}

				pathindex -= parts;
			}
		}
		plhs[MATLAB_out_stego] = closest;
		free(path);
	}

	free(matrices);
	free(widths);
	free(columns[0]);
	free(columns[1]);

	if(nlhs > 0) {
		int size[2];
		mxArray *arr;

		size[0] = size[1] = 1;
		arr = mxCreateNumericArray(2, size, mxDOUBLE_CLASS, mxREAL);
		(mxGetPr(arr))[0] = totalprice;
		plhs[MATLAB_out_price] = arr;
	}

	return;
}

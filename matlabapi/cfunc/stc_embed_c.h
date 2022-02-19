#ifndef STC_EMBED_C_H
#define STC_EMBED_C_H

#include "common.h"
/* Inputs:
	cover - the binary cover vector
	coverlength - length of the cover vector
	message - the binary message to be hidden
	messagelength - length of the message
	profile - the vector of distortion weights (either double if usedouble = true, or u8 id usedouble = false)
	usedouble - true = use double precision weight, false = use u8 weights
	stego - pointer to an array of length 'coverlength' to receive the stego message; this parameter can be NULL
	constr_height - the constraint height of the matrix; the higher, the better the efficiency but the greater the embedding time

Return value:
	On success, the function returns the total distortion introduced by the embedding.
	On error, the function returns -1.
*/

double stc_embed(const u8 *cover, int coverlength, const u8 *message, int messagelength, const void *profile, bool usedouble, u8 *stego, int constr_height = 10);

#endif

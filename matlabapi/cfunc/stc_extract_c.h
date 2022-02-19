#ifndef STC_EXTRACT_C_H
#define STC_EXTRACT_C_H

#include "common.h"

/* Inputs:
	stego - the binary stego vector
	stegolength - the length of the stego vector
	message - pointer to an array of legth 'messagelength' to receive the extracted message
	messagelegth - the length of the embedded message
	constr_height - the constraint height of the matrix used for embedding the message

Return values:
	0 on succes, -1 on error
*/

int stc_extract(const u8 *stego, int stegolength, u8 *message, int messagelength, int constr_height = 10);

#endif

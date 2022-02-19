#include <cstdio>
#include <cstdlib>
#include "stc_embed_c.h"
#include "stc_extract_c.h"

const int coverlength = 1000000;
const int msglength = 123456;

int main(int argc, char **argv)
{
	u8 *cover;
	u8 *msg, *msg2;
	u8 *stego;
	u8 *profile;
	int i;
	double dist;

	cover = new u8[coverlength];
	msg = new u8[msglength];
	msg2 = new u8[msglength];
	stego = new u8[coverlength];
	profile = new u8[coverlength];

	for(i = 0; i < coverlength; i++) {
		cover[i] = (u8)(rand() & 1);
		profile[i] = 1;
	}
	
	for(i = 0; i < msglength; i++) {
		msg[i] = (u8)(rand() & 1);
	}

	dist = stc_embed(cover, coverlength, msg, msglength, (void*)profile, false, stego);
	printf("Distortion: %lf\n", dist);
	stc_extract(stego, coverlength, msg2, msglength);

	for(i = 0; i < msglength; i++) {
		if(msg[i] != msg2[i]) {
			printf("Some error occurred at position %d.\n", i);
			break;
		}
	}
	if(i == msglength)
		printf("Everything works.\n");
	
	printf("Press enter to exit.\n");
	getchar();

	delete[] profile;
	delete[] stego;
	delete[] msg2;
	delete[] msg;
	delete[] cover;

	return 0;
}

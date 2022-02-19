#
# Embedding and Extraction Algorithm Using syndrome Trellis Coding
# for MATLAB

Contents of folder:
- stc_embed.cpp
	C++ source code of the syndrome trellis embedding algorithm for MATLAB. Requires the SSE2 instruction set for compilation and running.
- stc_extract.cpp
	C++ source code of the syndrome trellis extraction algorithm for MATLAB
- common.h
	Include file with optimized matrices for payloads 1/i, 2 <= i <= 20 and constraint heights 6 <= h <= 12 and shared functions
	(required file for compilation)
- readme.txt
	This file
- stc_embed.mexw32
	MATLAB MEX file with the embedding function, compiled with MATLAB R2006a. You can use this without compiling anything, but it's not guaranteed to work on all machines, operating systems, or versions of MATLAB.
- stc_extract.mexw32
	MATLAB MEX file with the extraction function. See above for details.
- example.m
	A MATLAB file illustrating the usage of the algorithm
- ex_linear.m
	An example MATLAB file for generating a graph of embedding efficiency vs. inverse payload for a given constraint length and a linear profile
- ex_wet.m
	An example MATLAB file illustrating the usage of the algorithm on a wet channel.
- cfunc/
	Folder with non-MATLAB versions of the embedding and extraction algorithm (C++ source code). See the readme.txt file in the folder for details.

------------------------------------

Installation:
Either:
	1) Copy the files stc_embed.mexw32, stc_extract.mexw32 into a MATLAB work directory (this method doesn't have to work on all machines)
Or:
	1) Copy the files stc_embed.cpp, stc_extract.cpp and common.h to a MATLAB work directory
	2) Run 
		mex stc_embed.cpp
		mex stc_extract.cpp
	in the MATLAB prompt. You need a C++ compiler with SSE2 intrinsics support.
	
------------------------------------

Usage:
Embedding:
Use the function [dist, stego] = stc_embed(cover, message, profile, constr_height = 10)
where the inputs are:
	cover - the binary cover vector (must be of type uint8)
	message - the binary message to be communicated (must be of type uint8)
	profile - your distortion profile (must be of type uint8 or double)
	constr_height (optional) - the constraint height of the used matrix. 
		This should be a number between 6 and 15 (a higher number means bigger efficiency but longer embedding time), default is 10.
and the outputs:
	dist - the total distortion introduced by embedding the message
	stego (optional) - the binary stego vector communicating the message

Extraction:
Use the function [message] = stc_extract(stego, message_length, constr_height = 10)
where the output is the embedded message and the inputs are:
	stego - the binary stego vector (must be of type uint8)
	message_length - the length of the hidden message
	constr_height (optional) - the constraint height of the matrix used for embedding the message, default is 10
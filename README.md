# BigNoiseyBoi
A program to generate an image meeting 2 conditions:

1. Every pixel must be a unique colour
2. The image must not contain any colours on an exclusion list

As fast as (non)humanly possible.

This project started after my friend needed this for modding a Paradox game. I created a quick JavaScript solution, but while he was satisfied, I wasn't. I wanted this to be fast. As fuck. I also wanted a chance to learn GPU acceleration with CUDA and to try my hand at C++ again, and so this is that.

## Benchmarks

On my home PC with an AMDFX-8350 and Nvidia GTX 1060 6GB.

Generating a 4000x4000 image (16 million unique colours), averaged over 10 runs, in seconds.

**No exlusions:**

|CPU Algorithm|GPU Algorithm|
|---|---|
|0.07520231|0.04212675|

**1,231 excluded colours:**

|CPU Algorithm|GPU Algorithm|
|---|---|
|0.10181006|0.04179284|

## Usage

BigNoiseyBoi is built as a command line tool, though I have included a .bat wrapper to make it easier to use for those not wanting so much pain.

### Using the BigNoiseyWrapper.bat
The BigNoiseyWrapper does nothing more than take an input of arguments and runs BigNoiseyBoi.exe with those arguments, and repeat. Type in whatever arguments you want to run BigNoiseyBoi with into the input, hit enter, and the output will be shown.

Example:
`BigNoiseyBoi.exe > 100 100 -v`

### Using BigNoiseyBoi.exe from the command line
`BigNoiseyBoi.exe <args>`

### Arguments
A full list and description of arguments can be seen by using the `-h` command:

`BigNoiseyBoi.exe -h`

## Licensing
This program is entirely open source under the MIT license. Go wild.

## Future
If anyone wants to continue it and further improve on the (admittedly messy and terrible) code I have written, I'd be happy to add contributors. I might revisit this but for now I am moving on to other things.
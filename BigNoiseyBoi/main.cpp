#include "defines.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "input.hpp"
#include "imgutils.hpp"
#include "imgprocessing.cuh"

using namespace std;
using namespace cv;

void printHelp() {
    cout << endl;
    cout << "=============================================================" << endl;
    cout << "======================= BIG NOISEY BOI ======================" << endl;
    cout << "=============================================================" << endl;
    cout << endl;
    cout << "For years desperate Paradox Games modders have wandered in lonely homes, looking at blank screens, screamingly inwardly, begging God for that one thing they need; only one thing that lies between them and the creation of a masterpiece of a mod..." << endl;
    cout << endl;
    cout << "... whatever that thing is, it's almost certainly not a ridiculously optimised GPU-accelerated unique colour map generator. But that's what I give to you, like it or not." << endl;
    cout << endl;
    cout << "=========================== USAGE ===========================" << endl;
    cout << endl;
    cout << "Command: BigNoiseyBoi.exe <width> <height> [options]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "\t-e <filepath>" << endl;
    cout << "\t\tExclusions: Accepts a path to a CSV file of excluded RGB colours to avoid in generation." << endl;
    cout << "\t-V" << endl;
    cout << "\t\tVersion: prints the current version of BigNoiseyBoi" << endl;
    cout << "\t-v" << endl;
    cout << "\t\tVerbose: outputs extra info for nerds." << endl;
    cout << "\t-vv" << endl;
    cout << "\t\tVery Verbose: prints out data for extra debug.\n\t\t!!WARNING!! Use only for small images, outputs a LOT of data!" << endl;
    cout << "\t-va" << endl;
    cout << "\t\tValidate: thorough check to verify that an image's pixels are all unique colours.\n\t\tIntensive for large images!" << endl;
    cout << "\t-h" << endl;
    cout << "\t\tHelp: prints out the help message." << endl;
    cout << "\t-cpu" << endl;
    cout << "\t\tCPU: Force the image generation to use the CPU. Useful if you do not have an NVIDIA (CUDA compatible) GPU." << endl;
    cout << "\t-cmp" << endl;
    cout << "\t\tCompare: Run both GPU and CPU algorithms and compare them. Mainly useful for debug." << endl;
    cout << "\t-b" << endl;
    cout << "\t\tBenchmark: Only for the brave. Test your CPU and GPU to their limits. Mainly just for fun, using any stats from this test seriously is heavily discouraged." << endl;
    cout << endl;
    cout << "\t-s <width> <height>" << endl;
    cout << "\t\tSize: Set a size for the image generated to be stretched/squished into. Defaults to the initial sizes." << endl;
}

void printMatrix (Mat* M, string device) {
    cout << device << " Generated image matrix:" << endl << endl;
    for (int i = 0; i < M->rows; i++) {
        cout << "   B    G    R ";
    }
    cout << endl << endl << *M << endl;
}

int main(int argc, char* argv[])
{

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // ARGUMENT PARSING
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Arguments acceptable before width and height:
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0) {
            printHelp();
            return 0;
        }
        else if (strcmp(argv[1], "-V") == 0) {
            cout << "BIG NOISEY BOI " << VERSION << " by Ben Durkin 2020" << endl;
            cout << "Open source under the MIT license at https://github.com/dodo721/BigNoiseyBoi" << endl;
            return 0;
        }
    }

    if (argc < 3) {
        cout << "Usage: BigNoiseyBoi.exe <width> <height> [options]\nUse -h for help." << endl;
        return 0;
    }

    int sizeX = getNumInput(argv[1]);
    int sizeY = getNumInput(argv[2]);

    if (sizeX == 0 || sizeY == 0) {
        cout << "Usage: BigNoiseyBoi.exe <width> <height> [options]\nUse -h for help." << endl;
        return 0;
    }

    int winSizeX = sizeX;
    int winSizeY = sizeY;

    bool verbose = false;
    bool veryverbose = false;
    bool validate = false;
    bool cpu = false;
    bool gpu = true;
    bool comparing = false;
    colour* exclusions = nullptr;
    size_t exclLength = 0;

    // Arguments that must come after width and height (including those before):
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 3; i < args.size(); ++i) {
        if (args[i] == "-v")
            verbose = true;
        else if (args[i] == "-vv") {
            veryverbose = true;
        }
        else if (args[i] == "-h") {
            printHelp();
            return 0;
        }
        else if (args[i] == "-s") {
            winSizeX = getNumInput(argv[i + 1]);
            winSizeY = getNumInput(argv[i + 2]);
            if (winSizeX == 0 || winSizeY == 0) {
                cout << "Usage: BigNoiseyBoi.exe <width> <height> [options]\nUse -h for help." << endl;
                return 0;
            }
            i += 2;
        }
        else if (args[i] == "-V") {
            cout << "BIG NOISEY BOI " << VERSION << " by Ben Durkin 2020" << endl;
            cout << "Open source under the MIT license at https://github.com/dodo721/BigNoiseyBoi" << endl;
            return 0;
        }
        else if (args[i] == "-cpu") {
            cpu = true;
            gpu = false;
        }
        else if (args[i] == "-cmp") {
            comparing = true;
        }
        else if (args[i] == "-va") {
            validate = true;
        }
        else if (args[i] == "-e") {
            string filepath = args[i + 1];
            try {
                exclusions = readColourCSV(filepath, &exclLength);
                exclLength /= 3;
            }
            catch (...) {
                cerr << "Could not read CSV! Check the filepath is correct and the file is not malformed." << endl;
                return 0;
            }
            i++;
        }
        else {
            cout << "Unrecognized argument: " << args[i] << endl << "Use -h for help." << endl;
            return 0;
        }
    }

    if (comparing) {
        cpu = true;
        gpu = true;
    }

    if (veryverbose)
        verbose = true;

    if (verbose) {
        cout << "Run configuration:\n\tGPU: " << gpu << "\n\tCPU: " << cpu << endl;

        if (gpu) {
            cout << "Initialising CUDA..." << endl;
            InitialiseCUDA();
        }
    }

    cout << "Generating " << sizeX << "x" << sizeY << " image..." << endl;

    const unsigned int imgDataLength = sizeX * sizeY * 3;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // CPU GEN
    ////////////////////////////////////////////////////////////////////////////////////////////////

    colour* imgDataCPU;

    if (cpu) {

        imgDataCPU = new colour[imgDataLength];

        double t = (double)getTickCount();

        LinearGenImageCPU(imgDataCPU, imgDataLength / 3);

        t = ((double)getTickCount() - t) / getTickFrequency();
        if (verbose)
            cout << "Single thread CPU time: " << t << endl;

        if (validate) {
            cout << "Checking CPU image is valid..." << endl;
            bool valid = ValidateImage(imgDataCPU, imgDataLength / 3);
            cout << "Validity: " << valid << endl;
        }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // GPU GEN
    ////////////////////////////////////////////////////////////////////////////////////////////////

    colour* imgDataGPU;

    if (gpu) {

        imgDataGPU = new colour[imgDataLength];

        double t = (double)getTickCount();

        LinearGenImageGPU(imgDataGPU, imgDataLength, exclusions, exclLength, verbose);

        t = ((double)getTickCount() - t) / getTickFrequency();
        if (verbose)
            cout << "Multi thread GPU time: " << t << endl;

        if (validate) {
            cout << "Checking GPU image is valid..." << endl;
            bool valid = ValidateImage(imgDataGPU, imgDataLength / 3);
            cout << "Validity: " << valid << endl;
        }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // OUTPUT
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (exclusions != nullptr) delete exclusions;

    if (cpu) {
        Mat M1(sizeX, sizeY, CV_8UC3, imgDataCPU);
        if (veryverbose) {
            printMatrix(&M1, "CPU");
        }

        showImg(&M1, winSizeX, winSizeY, "CPU Generated");
    }

    if (gpu) {
        Mat M2(sizeX, sizeY, CV_8UC3, imgDataGPU);
        if (veryverbose) {
            printMatrix(&M2, "GPU");
        }

        showImg(&M2, winSizeX, winSizeY, "GPU Generated");
    }

    waitKey(0);

    return 0;

}
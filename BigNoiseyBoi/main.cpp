#include "defines.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <ppl.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "input.hpp"
#include "imgutils.hpp"
#include "colour_indexer.cuh"
#include "imgprocessing.cuh"

using namespace std;
using namespace cv;
using namespace concurrency;

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
    cout << "\t-s <width> <height>" << endl;
    cout << "\t\tSize: Set a size for the image generated to be stretched/squished into. Defaults to the initial sizes." << endl;
    cout << "\t-sv <filepath>" << endl;
    cout << "\t\tSave: Save the generated image(s) after generation to the designated folder." << endl;
}

void printMatrix(Mat* M, string device) {
    cout << device << " Generated image matrix:" << endl << endl;
    for (int i = 0; i < M->rows; i++) {
        cout << "   B    G    R ";
    }
    cout << endl << endl << *M << endl;
}

colour* GPUGen(size_t imgDataLength, ColourEntry* exclusionIndex, bool verbose, bool validate) {

    colour* imgDataGPU = new colour[imgDataLength];

    double t = (double)getTickCount();

    LinearGenImageGPU(imgDataGPU, imgDataLength, exclusionIndex, verbose);

    t = ((double)getTickCount() - t) / getTickFrequency();
    if (verbose)
        cout << "Multi thread GPU time: " << t << endl;

    if (validate) {
        cout << "Checking GPU image is valid..." << endl;
        bool valid = ValidateImage(imgDataGPU, imgDataLength / 3);
        cout << "Validity: " << valid << endl;
    }
    return imgDataGPU;
}

colour* CPUGen(size_t imgDataLength, ColourEntry* exclusionIndex, bool verbose, bool validate) {

    colour* imgDataCPU = new colour[imgDataLength];

    double t = (double)getTickCount();

    LinearGenImageCPU(imgDataCPU, imgDataLength / 3, exclusionIndex, verbose);

    t = ((double)getTickCount() - t) / getTickFrequency();
    if (verbose)
        cout << "Single thread CPU time: " << t << endl;

    if (validate) {
        cout << "Checking CPU image is valid..." << endl;
        bool valid = ValidateImage(imgDataCPU, imgDataLength / 3);
        cout << "Validity: " << valid << endl;
    }

    return imgDataCPU;
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

    if (sizeX == -1 || sizeY == -1) {
        cout << "Usage: BigNoiseyBoi.exe <width> <height> [options]\nUse -h for help." << endl;
        return 0;
    }

    int winSizeX = sizeX;
    int winSizeY = sizeY;

    bool verbose = false;
    bool validate = false;
    bool cpu = false;
    bool gpu = true;
    bool comparing = false;
    bool benchmark = false;
    bool save = false;
    string savePath = "";
    string fileType = "PNG";
    ColourEntry* exclusionIndex = nullptr;
    colour* exclusions = nullptr;
    size_t exclLength = 0;

    // Arguments that must come after width and height (including those before):
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 3; i < args.size(); ++i) {
        if (args[i] == "-v")
            verbose = true;
        else if (args[i] == "-h") {
            printHelp();
            return 0;
        }
        else if (args[i] == "-s") {
            if (i + 1 >= args.size() || i + 2 >= args.size()) {
                cout << "Too few arguments for -s!\nUse -h for help." << endl;
                return 0;
            }
            winSizeX = getNumInput(argv[i + 1]);
            winSizeY = getNumInput(argv[i + 2]);
            if (winSizeX == -1 || winSizeY == -1) {
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
        else if (args[i] == "-b") {
            benchmark = true;
        }
        else if (args[i] == "-sv") {
            save = true;
            if (i + 1 >= args.size()) {
                cout << "Too few arguments for -sv!\nUse -h for help." << endl;
                return 0;
            }
            savePath = argv[i + 1];
            i++;
            struct stat info;
            if (stat(savePath.c_str(), &info) != 0) {
                cerr << "Cannot access " << savePath << " for saving!" << endl;
                return 0;
            }
            else if (!(info.st_mode & S_IFDIR)) {  // S_ISDIR() doesn't exist on my windows 
                cerr << savePath << " is not a directory!" << endl;
                return 0;
            }
        }
        else if (args[i] == "-e") {
            if (i + 1 >= args.size()) {
                cout << "Too few arguments for -e!\nUse -h for help." << endl;
                return 0;
            }
            string filepath = args[i + 1];
            try {
                exclusions = readColourCSV(filepath, &exclLength);
                exclLength /= 3;
                if (exclusions == nullptr) {
                    cerr << "Could not read CSV! Check the filepath is correct and the file is not malformed." << endl;
                    return 0;
                }
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

    // Check image size is not larger than number of possible unique colours with exclusions
    int maxUniqueColours = (256 * 256 * 256) - exclLength;
    // Cast to size_t to avoid overflow
    size_t imgSize = (size_t)sizeX * sizeY;
    if (imgSize > maxUniqueColours) {
        int maxDim = floor(sqrt(maxUniqueColours));
        cout << "The given image size is too large, and would contain non-unique pixels.\nThe maximum image size possible with the given exclusions is " << maxUniqueColours << " pixels (equivalent to " << maxDim << "x" << maxDim << ")." << endl;
        return 0;
    }

    if (comparing) {
        cpu = true;
        gpu = true;
    }

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
    // BENCHMARK
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (benchmark) {

        cout << "Benchmarking 1,000,000 image generations - CPU" << endl;
        exclusionIndex = CreateIndex(exclusions, exclLength, false, verbose);
        double t = getTickCount();
        parallel_for(size_t(0), (size_t)1000000, [&](size_t i) {
            delete[] CPUGen(imgDataLength, exclusionIndex, verbose, validate);
        });
        FreeIndex(exclusionIndex, false);
        t = (getTickCount() - t) / getTickFrequency();
        cout << "Generated 1,000,000 images using the CPU algorithm in " << t << "s" << endl;

        cout << "GPU benchmarking is still in development. Sorry!" << endl;
        
        // TODO: parallelize multiple images within GPU algorithm, to properly test GPU

        /*cout << "Benchmarking 1,000,000 image generations - GPU" << endl;
        exclusionIndex = CreateIndex(exclusions, exclLength, true, verbose);
        t = getTickCount();
        for (int i = 0; i < 1000000; i++) {
            colour* imgDataCPU = GPUGen(imgDataLength, exclusionIndex, verbose, validate);
            delete[] imgDataCPU;
        }
        FreeIndex(exclusionIndex, true);
        t = (getTickCount() - t) / getTickFrequency();
        cout << "Generated 1,000,000 images using the GPU algorithm in " << t << "s" << endl;*/

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // CPU GEN
    ////////////////////////////////////////////////////////////////////////////////////////////////

    colour* imgDataCPU = nullptr;

    if (cpu) {

        exclusionIndex = CreateIndex(exclusions, exclLength, false, verbose);
        imgDataCPU = CPUGen(imgDataLength, exclusionIndex, verbose, validate);
        FreeIndex(exclusionIndex, false);

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // GPU GEN
    ////////////////////////////////////////////////////////////////////////////////////////////////

    colour* imgDataGPU = nullptr;

    if (gpu) {

        exclusionIndex = CreateIndex(exclusions, exclLength, true, verbose);
        imgDataGPU = GPUGen(imgDataLength, exclusionIndex, verbose, validate);
        FreeIndex(exclusionIndex, true);

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // OUTPUT
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (exclusions != nullptr) delete exclusions;

    Mat* M1 = nullptr; 
    if (cpu) {
        M1 = new Mat(sizeX, sizeY, CV_8UC3, imgDataCPU);
        showImg(M1, winSizeX, winSizeY, "CPU Generated");
    }

    Mat* M2 = nullptr;
    if (gpu) {
        M2 = new Mat(sizeX, sizeY, CV_8UC3, imgDataGPU);
        showImg(M2, winSizeX, winSizeY, "GPU Generated");
    }

    waitKey(0);

    if (cpu) {
        if (save) {
            string cpuImgPath = savePath + "/BigNoiseyGen_" + to_string(sizeX) + "x" + to_string(sizeY) + "_CPU" + (exclusionIndex != nullptr ? "_excl" : "") + "." + fileType;
            imwrite(cpuImgPath, *M1);
            cout << "Saved CPU image to " << cpuImgPath << endl;
        }
        delete M1;
    }
    if (gpu) {
        if (save) {
            string gpuImgPath = savePath + "/BigNoiseyGen_" + to_string(sizeX) + "x" + to_string(sizeY) + "_GPU" + (exclusionIndex != nullptr ? "_excl" : "") + "." + fileType;
            imwrite(gpuImgPath, *M2);
            cout << "Saved GPU image to " << gpuImgPath << endl;
        }
        delete M2;
    }

    delete[] imgDataCPU;
    delete[] imgDataGPU;

    return 0;

}
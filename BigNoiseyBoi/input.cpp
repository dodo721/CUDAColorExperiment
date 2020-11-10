#include "defines.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <fstream>
#include "input.hpp"

using namespace std;

int getNumInput(char* inp) {

    string arg = inp;
    try {
        size_t pos;
        int x = stoi(arg, &pos);
        if (pos < arg.size()) {
            cerr << "Trailing characters after number: " << arg << '\n';
        }
        return x;
    }
    catch (invalid_argument const& ex) {
        cerr << "Invalid number: " << arg << '\n';
    }
    catch (out_of_range const& ex) {
        cerr << "Number out of range: " << arg << '\n';
    }
    return 0;
}

colour* parseColour(string line, int lineNo) {
    string delimiter = ",";
    size_t pos = 0;
    colour* rgb = new colour[3];
    int count = 0;
    string token;
    while ((pos = line.find(delimiter)) != string::npos) {
        token = line.substr(0, pos);
        if (count > 1)
            throw;
        int temp = stoi(token);
        if (temp > 255) {
            cerr << "Error at line " << lineNo << ": Colour above 255! " << line << endl;
            throw;
            return nullptr;
        }
        rgb[count] = temp;
        count++;
        line.erase(0, pos + delimiter.length());
    }
    rgb[2] = stoi(line);
    return rgb;
}

colour* readColourCSV(string filepath, size_t* length) {

    ifstream fileCounter(filepath);
    if (!fileCounter.good()) return nullptr;
    size_t lineCount = std::count(istreambuf_iterator<char>(fileCounter),
        istreambuf_iterator<char>(), '\n');
    fileCounter.close();

    ifstream file(filepath);
    colour* pixels = new colour[lineCount * 3];
    string line;
    int i = 0;
    while (getline(file, line))
    {
        int index = i * 3;
        colour* pixel;
        try {
            pixel = parseColour(line, i);
        }
        catch (...) {
            file.close();
            delete[] pixels;
            return nullptr;
        }
        pixels[index] = pixel[0];
        pixels[index + 1] = pixel[1];
        pixels[index + 2] = pixel[2];
        //cout << "Read line colour " << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << endl;
        i++;
        if (i > lineCount) {
            cout << "WARNING exceeded line count!" << endl;
            break;
        }
    }

    *length = lineCount * 3;
    file.close();
    return pixels;

}
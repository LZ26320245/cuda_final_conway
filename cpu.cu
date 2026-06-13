#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include <filesystem>
#include <random>
#include <iomanip>
#include <sstream>
#include <chrono>
using namespace std;
namespace fs = std::filesystem;

int countNeighbors(const uint8_t* grid,int x,int y,int width,int height)
{
    int count = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0)
                nx += width;
            else if (nx >= width)
                nx -= width;

            if (ny < 0)
                ny += height;
            else if (ny >= height)
                ny -= height;

            count += grid[ny * width + nx];
        }
    }

    return count;
}

void saveGrid(const vector<uint8_t>& grid,int iteration,const string& folder,int width,int height)
{
    ostringstream filename;

    filename << folder<< "/iter_"<< setw(3)<< setfill('0')<< iteration<< ".txt";

    ofstream out(filename.str());

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            out << int(grid[y * width + x]);
        }
        out << '\n';
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        cout << "Usage:\n";
        cout << argv[0] << " WIDTH HEIGHT ITERATIONS\n";
        return 1;
    }

    int WIDTH = stoi(argv[1]);
    int HEIGHT = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);

    string outputFolder = "output_cpu";

    fs::create_directory(outputFolder);

    vector<uint8_t> current(WIDTH * HEIGHT);
    vector<uint8_t> next(WIDTH * HEIGHT);

    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12);

    for (auto& cell : current)
    {
        cell = (dist(rng) == 0) ? 1 : 0;
    }

    const uint8_t* currentGrid;
    uint8_t* nextGrid;

    saveGrid(current,0,outputFolder,WIDTH,HEIGHT);

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        currentGrid = current.data();
        nextGrid = next.data();

        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                int idx = y * WIDTH + x;

                int neighbors = countNeighbors(currentGrid,x,y,WIDTH,HEIGHT);

                if (currentGrid[idx])
                {
                    nextGrid[idx] = (neighbors == 2 || neighbors == 3);
                }
                else
                {
                    nextGrid[idx] = (neighbors == 3);
                }
            }
        }

        saveGrid(next,iter,outputFolder,WIDTH,HEIGHT);

        swap(current, next);
    }

    auto end = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double, milli>(end - start).count();

    cout << "\nExecution Time: " << elapsed << " ms\n";

    return 0;
}
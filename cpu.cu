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

constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;
constexpr int ITERATIONS = 100;

namespace fs = std::filesystem;

int countNeighbors(
    const uint8_t* grid,
    int x,
    int y)
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

            // 邊界採用循環邊界(最左邊會循環到最右邊，最上面會循環到最下面)
            if (nx<0){
                nx = nx+WIDTH;
            }
            else if (nx>=WIDTH){
                nx = nx-WIDTH;
            }

            if (ny<0){
                ny = ny+HEIGHT;
            }
            else if (ny>=HEIGHT){
                ny = ny-HEIGHT;
            }

            count += grid[ny * WIDTH + nx];
        }
    }

    return count;
}

void saveGrid(
    const vector<uint8_t>& grid,
    int iteration,
    const string& folder)
{
    ostringstream filename;

    filename << folder << "/iter_" << setw(3) << setfill('0') << iteration << ".txt";

    ofstream out(filename.str());

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            out << int(grid[y * WIDTH + x]);
        }
        out << '\n';
    }
}

int main()
{
    string outputFolder = "output_cpu";

    fs::create_directory(outputFolder);

    vector<uint8_t> current(WIDTH * HEIGHT);
    vector<uint8_t> next(WIDTH * HEIGHT);

    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12); //決定alive細胞的比例

    // random initialization
    for (auto& cell : current)
    {
        cell = (dist(rng) == 0) ? 1 : 0;
    }

    // 迭代更新採用 current和next swap的方式 節省記憶體空間
    const uint8_t* currentGrid;
    uint8_t* nextGrid;
    saveGrid(current, 0, outputFolder);
    
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

                int neighbors = countNeighbors(currentGrid, x, y);

                if (currentGrid[idx] == 1)
                {
                    if (neighbors == 2 || neighbors == 3)
                        nextGrid[idx] = 1;
                    else
                        nextGrid[idx] = 0;
                }
                else
                {
                    if (neighbors == 3)
                        nextGrid[idx] = 1;
                    else
                        nextGrid[idx] = 0;
                }
            }
        }

        saveGrid(next, iter, outputFolder);

        swap(current, next);
    }
    auto end = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double, milli>(end - start).count();

    cout<<"\nExecution Time: "<< elapsed << " ms\n";
    return 0;
}
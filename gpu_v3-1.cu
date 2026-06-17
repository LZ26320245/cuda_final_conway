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

__constant__ uint8_t d_lut[512];

void saveGrid(const vector<uint8_t>& grid, int iteration, const string& folder, int width, int height)
{
    ostringstream filename;
    filename << folder << "/iter_" << setw(3) << setfill('0') << iteration << ".txt";
    ofstream out(filename.str());
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            out << int(grid[y * width + x]);
        out << '\n';
    }
}

__global__ void conwayKernel(const uint8_t* __restrict__ current, uint8_t* __restrict__ next,
                              int width, int height, int coarsen)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int xstart = (blockIdx.x * blockDim.x + tx) * coarsen;
    int y      =  blockIdx.y * blockDim.y + ty;

    if (y >= height) return;

    for (int c = 0; c < coarsen; c++)
    {
        int x = xstart + c;
        if (x >= width) break;

        // 計算 8 個鄰居的座標（循環邊界）
        int xL = (x == 0)         ? width  - 1 : x - 1;
        int xR = (x == width - 1) ? 0          : x + 1;
        int yU = (y == 0)         ? height - 1 : y - 1;
        int yD = (y == height - 1)? 0          : y + 1;

        // 直接從 global memory 讀 9 個 cell，組成 9-bit pattern
        unsigned int pattern =
            current[yU * width + xL]       |
            (current[yU * width + x ] << 1)|
            (current[yU * width + xR] << 2)|
            (current[y  * width + xL] << 3)|
            (current[y  * width + x ] << 4)|
            (current[y  * width + xR] << 5)|
            (current[yD * width + xL] << 6)|
            (current[yD * width + x ] << 7)|
            (current[yD * width + xR] << 8);

        next[y * width + x] = d_lut[pattern];
    }
}

int main(int argc, char* argv[])
{
    if (argc != 7)
    {
        cout << "Usage: " << argv[0] << " WIDTH HEIGHT ITERATIONS BLOCK_X BLOCK_Y COARSEN\n";
        return 1;
    }

    int WIDTH      = stoi(argv[1]);
    int HEIGHT     = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X    = stoi(argv[4]);
    int BLOCK_Y    = stoi(argv[5]);
    int coarsen    = stoi(argv[6]);

    string outputFolder = "output_gpu_v3-1";
    fs::create_directory(outputFolder);

    vector<uint8_t> current(WIDTH * HEIGHT);

    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12);
    for (auto& cell : current)
        cell = (dist(rng) == 0) ? 1 : 0;

    saveGrid(current, 0, outputFolder, WIDTH, HEIGHT);

    // 建立 LUT
    uint8_t h_lut[512];
    for (int pattern = 0; pattern < 512; pattern++)
    {
        int alive  = (pattern >> 4) & 1;
        int neighbors = __builtin_popcount(pattern) - alive;
        if (alive)
            h_lut[pattern] = (neighbors == 2 || neighbors == 3);
        else
            h_lut[pattern] = (neighbors == 3);
    }
    cudaMemcpyToSymbol(d_lut, h_lut, sizeof(h_lut));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint8_t *d_current, *d_next;
    size_t bytes = WIDTH * HEIGHT * sizeof(uint8_t);
    cudaMalloc(&d_current, bytes);
    cudaMalloc(&d_next,    bytes);
    cudaMemcpy(d_current, current.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((WIDTH  + BLOCK_X * coarsen - 1) / (BLOCK_X * coarsen),
              (HEIGHT + BLOCK_Y           - 1) /  BLOCK_Y);

    cudaEventRecord(start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        conwayKernel<<<grid, block>>>(d_current, d_next, WIDTH, HEIGHT, coarsen);

        //cudaDeviceSynchronize();

        swap(d_current, d_next);

        // cudaMemcpy(current.data(), d_current, bytes, cudaMemcpyDeviceToHost);
        // saveGrid(current, iter, outputFolder, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "\nExecution Time: " << elapsed << " ms\n";
    cout << "Throughput: " << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6 << " Gcells/s\n";

    cudaFree(d_current);
    cudaFree(d_next);
    return 0;
}

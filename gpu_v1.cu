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

            if (nx<0)
                nx = nx+WIDTH;
            else if (nx>=WIDTH)
                nx = nx-WIDTH;

            if (ny<0)
                ny = ny+HEIGHT;
            else if (ny>=HEIGHT)
                ny = ny-HEIGHT;

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

__global__ void conwayKernel(const uint8_t* current,uint8_t* next,int width,int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

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
            if (nx < 0)
                nx += width;
            else if (nx >= width)
                nx -= width;

            if (ny < 0)
                ny += height;
            else if (ny >= height)
                ny -= height;

            count += current[ny * width + nx];
        }
    }

    int idx = y * width + x;

    if (current[idx])
    {
        next[idx] = (count == 2 || count == 3);
    }
    else
    {
        next[idx] = (count == 3);
    }
}

int main()
{
    string outputFolder = "output_gpu_v1";

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
    saveGrid(current, 0, outputFolder);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    uint8_t* d_current;
    uint8_t* d_next;

    size_t bytes =
        WIDTH * HEIGHT * sizeof(uint8_t);

    cudaMalloc(&d_current, bytes);
    cudaMalloc(&d_next, bytes);

    cudaMemcpy(d_current,current.data(),bytes,cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid((WIDTH + block.x - 1) / block.x,(HEIGHT + block.y - 1) / block.y);
    cudaEventRecord(start);
    for (int iter = 1;iter <= ITERATIONS;iter++)
    {
        conwayKernel<<<grid, block>>>(d_current,d_next,WIDTH,HEIGHT);

        cudaDeviceSynchronize();

        std::swap(d_current, d_next);

        cudaMemcpy(current.data(),d_current,bytes,cudaMemcpyDeviceToHost);
        saveGrid(current, iter, outputFolder);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed,start,stop);

    cout<<"\nExecution Time: "<< elapsed << " ms\n";
    return 0;
}
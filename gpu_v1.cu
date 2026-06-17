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

void saveGrid(const vector<uint8_t>& grid,int iteration,const string& folder,int width,int height)
{
    ostringstream filename;

    filename<< folder<< "/iter_"<< setw(3)<< setfill('0')<< iteration<< ".txt";

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

__global__ void conwayKernel(const uint8_t* current,uint8_t* next,int width,int height, int coarsen)
{
    int xStart = (blockIdx.x * blockDim.x + threadIdx.x) * coarsen;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(y >= height)
        return;

    for(int c = 0; c < coarsen; c++)
    {
        int x = xStart + c;

        if(x >= width)
            break;

        int count = 0;

        for(int dy = -1; dy <= 1; dy++)
        {
            for(int dx = -1; dx <= 1; dx++)
            {
                if(dx == 0 && dy == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;

                if(nx < 0)
                    nx += width;
                else if(nx >= width)
                    nx -= width;

                if(ny < 0)
                    ny += height;
                else if(ny >= height)
                    ny -= height;

                count += current[ny * width + nx];
            }
        }

        int idx = y * width + x;

        if(current[idx])
        {
            next[idx] = (count == 2 || count == 3);
        }
        else
        {
            next[idx] = (count == 3);
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 7)
    {
        cout << "Usage:\n";
        cout << argv[0] << " WIDTH HEIGHT ITERATIONS BLOCK_X BLOCK_Y\n";
        return 1;
    }

    int WIDTH = stoi(argv[1]);
    int HEIGHT = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X = stoi(argv[4]);
    int BLOCK_Y = stoi(argv[5]);
    int coarsen = stoi(argv[6]);

    string outputFolder = "output_gpu_v1";

    fs::create_directory(outputFolder);

    vector<uint8_t> current(WIDTH * HEIGHT);
    vector<uint8_t> next(WIDTH * HEIGHT);

    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12);

    for (auto& cell : current)
    {
        cell = (dist(rng) == 0) ? 1 : 0;
    }

    saveGrid(current, 0, outputFolder, WIDTH, HEIGHT);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint8_t* d_current;
    uint8_t* d_next;

    size_t bytes = WIDTH * HEIGHT * sizeof(uint8_t);

    cudaMalloc(&d_current, bytes);
    cudaMalloc(&d_next, bytes);

    cudaMemcpy(d_current,current.data(),bytes,cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);

    dim3 grid((WIDTH + BLOCK_X * coarsen - 1) / (BLOCK_X*coarsen),(HEIGHT + BLOCK_Y - 1) / BLOCK_Y);

    cudaEventRecord(start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        conwayKernel<<<grid, block>>>(d_current,d_next,WIDTH,HEIGHT,coarsen);

        //cudaDeviceSynchronize();

        swap(d_current, d_next);

        // cudaMemcpy(current.data(),d_current,bytes,cudaMemcpyDeviceToHost);

        // saveGrid(current,iter,outputFolder,WIDTH,HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cout << "\nExecution Time: "<< elapsed << " ms\n";
    cout << "Throughput: " << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6 << " Gcells/s\n";

    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
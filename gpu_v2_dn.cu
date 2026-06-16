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
    extern __shared__ uint8_t tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int xstart = (blockIdx.x * blockDim.x + tx) * coarsen;
    int y = blockIdx.y * blockDim.y + ty;

    int sharedWidth = blockDim.x*coarsen + 2;
    int sharedHeight = blockDim.y + 2;
    int tileSize = sharedWidth * sharedHeight;

    int tid =  ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;

    for(int i = tid; i < tileSize;i += numThreads)
    {
        int sx = i % sharedWidth;
        int sy = i / sharedWidth;

        int gx = blockIdx.x * blockDim.x * coarsen + sx - 1;
        int gy = blockIdx.y * blockDim.y + sy - 1;

        if(gx < 0)
            gx += width;
        else if(gx >= width)
            gx -= width;
        if(gy < 0)
            gy += height;
        else if(gy >= height)
            gy -= height;

        tile[i] = current[gy * width + gx];
    }

    __syncthreads();

    if (y >= height)
        return;

    for(int c=0; c<coarsen; c++)
    {
        int x = xstart + c;
        if(x>=width)
            break;

        int sx = tx*coarsen + c + 1;
        int sy = ty + 1;
        int count = tile[(sy-1)*sharedWidth + (sx-1)] + tile[(sy-1)*sharedWidth + sx] + tile[(sy-1)*sharedWidth + (sx+1)] +
                    tile[ sy   *sharedWidth + (sx-1)] + tile[ sy   *sharedWidth + (sx+1)] +
                    tile[(sy+1)*sharedWidth + (sx-1)] + tile[(sy+1)*sharedWidth + sx] + tile[(sy+1)*sharedWidth + (sx+1)];
        
        int idx = y * width + x;

        uint8_t alive = tile[sy * sharedWidth + sx];

        if(alive)
        {
            next[idx] = (count == 3 || count == 4 || count == 6 || count == 7 || count == 8);
        }
        else
        {
            next[idx] = (count == 3 || count == 6 || count == 7 || count == 8);
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

    string outputFolder = "output_gpu_v2_dn";

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

    dim3 grid((WIDTH + BLOCK_X*coarsen - 1) / (BLOCK_X*coarsen),(HEIGHT + BLOCK_Y - 1) / BLOCK_Y);

    size_t sharedBytes =(BLOCK_X*coarsen + 2)*(BLOCK_Y + 2)*sizeof(uint8_t);

    cudaEventRecord(start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        conwayKernel<<<grid, block, sharedBytes>>>(d_current,d_next,WIDTH,HEIGHT,coarsen);

        cudaDeviceSynchronize();

        swap(d_current, d_next);

        cudaMemcpy(current.data(),d_current,bytes,cudaMemcpyDeviceToHost);

        saveGrid(current,iter,outputFolder,WIDTH,HEIGHT);
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
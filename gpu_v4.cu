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

void packGrid(const vector<uint8_t>& flat, vector<uint32_t>& packed, int width, int height)
{
    int pw = (width + 31) / 32;
    packed.assign(height * pw, 0u);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (flat[y * width + x])
                packed[y * pw + (x >> 5)] |= (1u << (x & 31));
}

void unpackGrid(const vector<uint32_t>& packed, vector<uint8_t>& flat, int width, int height)
{
    int pw = (width + 31) / 32;
    flat.assign(height * width, 0u);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            flat[y * width + x] = (packed[y * pw + (x >> 5)] >> (x & 31)) & 1u;
}

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

__device__ __forceinline__
void addBit(uint32_t b, uint32_t& ones, uint32_t& twos)
{
    uint32_t carry = ones & b;
    ones  ^= b;
    twos  |= carry;
}

__global__ void conwayKernel(const uint32_t* __restrict__ current, uint32_t* __restrict__ next, int packed_width, int height, int coarsen)
{
    // Shared memory tile
    extern __shared__ uint32_t tile[];

    int sw = blockDim.x * coarsen + 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx_start = (blockIdx.x * blockDim.x + tx) * coarsen;
    int gy = blockIdx.y * blockDim.y + ty; 

    // Load shared tile halo
    int tileSize = sw * (blockDim.y + 2);
    int tid      = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    for (int i = tid; i < tileSize; i += nthreads)
    {
        int sx = i % sw;
        int sy = i / sw;

        int col = blockIdx.x * blockDim.x * coarsen + sx - 1;
        int row = blockIdx.y * blockDim.y + sy - 1;

        if (row < 0)       
            row += height;
        else if (row >= height) 
            row -= height;

        uint32_t val;
        if (col < 0) 
        {
            col += packed_width;
            val  = current[row * packed_width + col];
        } 
        else if (col >= packed_width) 
        {
            col -= packed_width;
            val  = current[row * packed_width + col];
        } 
        else 
        {
            val = current[row * packed_width + col];
        }

        tile[i] = val;
    }

    __syncthreads();

    if (gy >= height)
        return;

    int sy = ty + 1;

    for(int c=0; c<coarsen; c++)
    {
        int gx = gx_start+c;
        if(gx >= packed_width)
            break;
        
        int sx = tx * coarsen + c + 1;

        uint32_t row_above_L = tile[(sy-1)*sw + (sx-1)];
        uint32_t row_above_C = tile[(sy-1)*sw +  sx   ];
        uint32_t row_above_R = tile[(sy-1)*sw + (sx+1)];

        uint32_t row_mid_L = tile[sy*sw + (sx-1)];
        uint32_t row_mid_C = tile[sy*sw + sx];
        uint32_t row_mid_R = tile[sy*sw + (sx+1)];

        uint32_t row_below_L = tile[(sy+1)*sw + (sx-1)];
        uint32_t row_below_C = tile[(sy+1)*sw + sx];
        uint32_t row_below_R = tile[(sy+1)*sw + (sx+1)];

        uint32_t a_L = (row_above_C << 1) | (row_above_L >> 31); // up-left
        uint32_t a_C =  row_above_C;                              // up
        uint32_t a_R = (row_above_C >> 1) | (row_above_R << 31); // up-right

        uint32_t m_L = (row_mid_C   << 1) | (row_mid_L   >> 31); // left
        uint32_t m_R = (row_mid_C   >> 1) | (row_mid_R   << 31); // right

        uint32_t b_L = (row_below_C << 1) | (row_below_L >> 31); // down-left
        uint32_t b_C =  row_below_C;                              // down
        uint32_t b_R = (row_below_C >> 1) | (row_below_R << 31); // down-right

        uint32_t ones = 0, twos = 0, fours = 0;

        auto hadd = [&](uint32_t b) 
        {
            uint32_t c = ones & b;
            ones ^= b;
            uint32_t c2 = twos & c;
            twos ^= c;
            fours |= c2;
        };

        hadd(a_L); hadd(a_C); hadd(a_R);
        hadd(m_L); hadd(m_R);
        hadd(b_L); hadd(b_C); hadd(b_R);

        uint32_t eq2 = ~fours & twos & ~ones;
        uint32_t eq3 = ~fours & twos & ones;

        uint32_t center = row_mid_C;
        next[gy * packed_width + gx] = (center & (eq2 | eq3)) | (~center & eq3);
    }
}


int main(int argc, char* argv[])
{
    if (argc != 7) {
        cout << "Usage: " << argv[0] << " WIDTH HEIGHT ITERATIONS BLOCK_X BLOCK_Y coarsen\n";
        return 1;
    }

    int WIDTH = stoi(argv[1]);
    int HEIGHT = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X = stoi(argv[4]);
    int BLOCK_Y = stoi(argv[5]);
    int coarsen = stoi(argv[6]);

    int packed_width = (WIDTH + 31) / 32;

    string outputFolder = "output_gpu_v4";
    fs::create_directory(outputFolder);

    vector<uint8_t> flat(WIDTH * HEIGHT);
    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12);
    for (auto& c : flat) c = (dist(rng) == 0) ? 1 : 0;

    vector<uint32_t> packed;
    packGrid(flat, packed, WIDTH, HEIGHT);

    saveGrid(flat, 0, outputFolder, WIDTH, HEIGHT);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t *d_cur, *d_nxt;
    size_t bytes = packed_width * HEIGHT * sizeof(uint32_t);
    cudaMalloc(&d_cur, bytes);
    cudaMalloc(&d_nxt, bytes);
    cudaMemcpy(d_cur, packed.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((packed_width + BLOCK_X*coarsen - 1) / (BLOCK_X*coarsen),(HEIGHT + BLOCK_Y - 1) / BLOCK_Y);

    size_t sharedBytes = (BLOCK_X*coarsen + 2) * (BLOCK_Y + 2) * sizeof(uint32_t);

    cudaEventRecord(start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        conwayKernel<<<grid, block, sharedBytes>>>(d_cur, d_nxt, packed_width, HEIGHT, coarsen);

        //cudaDeviceSynchronize();

        swap(d_cur, d_nxt);

        // cudaMemcpy(packed.data(), d_cur, bytes, cudaMemcpyDeviceToHost);
        // unpackGrid(packed, flat, WIDTH, HEIGHT);
        // saveGrid(flat, iter, outputFolder, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << "\nExecution Time: " << elapsed << " ms\n";
    cout << "Throughput: " << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6 << " Gcells/s\n";

    cudaFree(d_cur);
    cudaFree(d_nxt);
    return 0;
}

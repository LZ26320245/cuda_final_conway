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

// ─── Kernel ──────────────────────────────────────────────────────────────────
// 5×5 鄰居（24個），規則 B6/S567
// Bitpack: LE convention, bit 0 = 最左邊的 cell
// 每個 thread 處理 coarsen 個 word（每個 word = 32 cells）
//
// Shift 方向：
//   左移 n 格 (x-n) = word << n，從左邊 word 補高位
//   右移 n 格 (x+n) = word >> n，從右邊 word 補低位
//
// 跨 word 補位（2格為例）：
//   (C << 2) | (L >> 30)  → 取 L 的 bit30~31 補到 bit0~1
//   (C >> 2) | (R << 30)  → 取 R 的 bit0~1  補到 bit30~31

__global__ void conwayKernel5x5(
    const uint32_t* __restrict__ current,
          uint32_t* __restrict__ next,
    int packed_width, int height, int coarsen)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx_start = (blockIdx.x * blockDim.x + tx) * coarsen;
    int gy       =  blockIdx.y * blockDim.y + ty;

    if (gy >= height) return;

    // ── 5 個 row index（循環邊界）────────────────────────────────────────────
    int r2u = gy - 2; if (r2u < 0)        r2u += height;
    int r1u = gy - 1; if (r1u < 0)        r1u += height;
    // gy = 中間行
    int r1d = gy + 1; if (r1d >= height)  r1d -= height;
    int r2d = gy + 2; if (r2d >= height)  r2d -= height;

    for (int c = 0; c < coarsen; c++)
    {
        int gx = gx_start + c;
        if (gx >= packed_width) break;

        // ── 5 個 col index（循環邊界）────────────────────────────────────────
        int c2L = gx - 2; if (c2L < 0)             c2L += packed_width;
        int c1L = gx - 1; if (c1L < 0)             c1L += packed_width;
        // gx = 中間 col
        int c1R = gx + 1; if (c1R >= packed_width)  c1R -= packed_width;
        int c2R = gx + 2; if (c2R >= packed_width)  c2R -= packed_width;

        // ── 讀取 5×5 = 25 個 word（從 global memory）────────────────────────
        // 命名：w_[行]_[列]，行: 2u/1u/m/1d/2d，列: 2L/1L/C/1R/2R
        uint32_t w_2u_2L = current[r2u * packed_width + c2L];
        uint32_t w_2u_1L = current[r2u * packed_width + c1L];
        uint32_t w_2u_C  = current[r2u * packed_width + gx ];
        uint32_t w_2u_1R = current[r2u * packed_width + c1R];
        uint32_t w_2u_2R = current[r2u * packed_width + c2R];

        uint32_t w_1u_2L = current[r1u * packed_width + c2L];
        uint32_t w_1u_1L = current[r1u * packed_width + c1L];
        uint32_t w_1u_C  = current[r1u * packed_width + gx ];
        uint32_t w_1u_1R = current[r1u * packed_width + c1R];
        uint32_t w_1u_2R = current[r1u * packed_width + c2R];

        uint32_t w_m_2L  = current[gy  * packed_width + c2L];
        uint32_t w_m_1L  = current[gy  * packed_width + c1L];
        uint32_t w_m_C   = current[gy  * packed_width + gx ];
        uint32_t w_m_1R  = current[gy  * packed_width + c1R];
        uint32_t w_m_2R  = current[gy  * packed_width + c2R];

        uint32_t w_1d_2L = current[r1d * packed_width + c2L];
        uint32_t w_1d_1L = current[r1d * packed_width + c1L];
        uint32_t w_1d_C  = current[r1d * packed_width + gx ];
        uint32_t w_1d_1R = current[r1d * packed_width + c1R];
        uint32_t w_1d_2R = current[r1d * packed_width + c2R];

        uint32_t w_2d_2L = current[r2d * packed_width + c2L];
        uint32_t w_2d_1L = current[r2d * packed_width + c1L];
        uint32_t w_2d_C  = current[r2d * packed_width + gx ];
        uint32_t w_2d_1R = current[r2d * packed_width + c1R];
        uint32_t w_2d_2R = current[r2d * packed_width + c2R];

        // ── 建立 24 個位元平面（每個 bit p = cell x=p 在該位置的鄰居狀態）──
        // LE: 左移 n = 取左邊鄰居，右移 n = 取右邊鄰居
        // 1格跨界: >> 31 或 << 31（取 1 bit）
        // 2格跨界: >> 30 或 << 30（取 2 bits）

        // 最上行 (y-2)，5 個位置
        uint32_t p_2u_2L = (w_2u_C  << 2) | (w_2u_1L >> 30);
        uint32_t p_2u_1L = (w_2u_C  << 1) | (w_2u_1L >> 31);
        uint32_t p_2u_C  =  w_2u_C;
        uint32_t p_2u_1R = (w_2u_C  >> 1) | (w_2u_1R << 31);
        uint32_t p_2u_2R = (w_2u_C  >> 2) | (w_2u_1R << 30);

        // 上行 (y-1)，5 個位置
        uint32_t p_1u_2L = (w_1u_C  << 2) | (w_1u_1L >> 30);
        uint32_t p_1u_1L = (w_1u_C  << 1) | (w_1u_1L >> 31);
        uint32_t p_1u_C  =  w_1u_C;
        uint32_t p_1u_1R = (w_1u_C  >> 1) | (w_1u_1R << 31);
        uint32_t p_1u_2R = (w_1u_C  >> 2) | (w_1u_1R << 30);

        // 中間行 (y)，4 個位置（跳過自己）
        uint32_t p_m_2L  = (w_m_C   << 2) | (w_m_1L  >> 30);
        uint32_t p_m_1L  = (w_m_C   << 1) | (w_m_1L  >> 31);
        uint32_t p_m_1R  = (w_m_C   >> 1) | (w_m_1R  << 31);
        uint32_t p_m_2R  = (w_m_C   >> 2) | (w_m_1R  << 30);

        // 下行 (y+1)，5 個位置
        uint32_t p_1d_2L = (w_1d_C  << 2) | (w_1d_1L >> 30);
        uint32_t p_1d_1L = (w_1d_C  << 1) | (w_1d_1L >> 31);
        uint32_t p_1d_C  =  w_1d_C;
        uint32_t p_1d_1R = (w_1d_C  >> 1) | (w_1d_1R << 31);
        uint32_t p_1d_2R = (w_1d_C  >> 2) | (w_1d_1R << 30);

        // 最下行 (y+2)，5 個位置
        uint32_t p_2d_2L = (w_2d_C  << 2) | (w_2d_1L >> 30);
        uint32_t p_2d_1L = (w_2d_C  << 1) | (w_2d_1L >> 31);
        uint32_t p_2d_C  =  w_2d_C;
        uint32_t p_2d_1R = (w_2d_C  >> 1) | (w_2d_1R << 31);
        uint32_t p_2d_2R = (w_2d_C  >> 2) | (w_2d_1R << 30);

        // ── 5-bit 並行計數器，累加 24 個鄰居 ────────────────────────────────
        // (sixteens, eights, fours, twos, ones) 每個 bit p 是 cell x=p 的鄰居總數
        uint32_t ones = 0, twos = 0, fours = 0, eights = 0, sixteens = 0;

        auto hadd = [&](uint32_t b) {
            uint32_t c1 = ones   & b;  ones   ^= b;
            uint32_t c2 = twos   & c1; twos   ^= c1;
            uint32_t c3 = fours  & c2; fours  ^= c2;
            uint32_t c4 = eights & c3; eights ^= c3;
            sixteens |= c4;
        };

        // 最上行 5 個
        hadd(p_2u_2L); hadd(p_2u_1L); hadd(p_2u_C);
        hadd(p_2u_1R); hadd(p_2u_2R);
        // 上行 5 個
        hadd(p_1u_2L); hadd(p_1u_1L); hadd(p_1u_C);
        hadd(p_1u_1R); hadd(p_1u_2R);
        // 中間行 4 個（無自己）
        hadd(p_m_2L);  hadd(p_m_1L);
        hadd(p_m_1R);  hadd(p_m_2R);
        // 下行 5 個
        hadd(p_1d_2L); hadd(p_1d_1L); hadd(p_1d_C);
        hadd(p_1d_1R); hadd(p_1d_2R);
        // 最下行 5 個
        hadd(p_2d_2L); hadd(p_2d_1L); hadd(p_2d_C);
        hadd(p_2d_1R); hadd(p_2d_2R);

        // ── B6/S567 規則判斷 ──────────────────────────────────────────────────
        // count == 5: 00101
        // count == 6: 00110
        // count == 7: 00111
        uint32_t eq5 = ~sixteens & ~eights &  fours & ~twos &  ones;
        uint32_t eq6 = ~sixteens & ~eights &  fours &  twos & ~ones;
        uint32_t eq7 = ~sixteens & ~eights &  fours &  twos &  ones;

        uint32_t survive = eq5 | eq6 | eq7;  // S567
        uint32_t born    = eq6;               // B6

        uint32_t center = w_m_C;
        next[gy * packed_width + gx] = (center & survive) | (~center & born);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc != 7) {
        cout << "Usage: " << argv[0] << " WIDTH HEIGHT ITERATIONS BLOCK_X BLOCK_Y COARSEN\n";
        return 1;
    }

    int WIDTH      = stoi(argv[1]);
    int HEIGHT     = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X    = stoi(argv[4]);
    int BLOCK_Y    = stoi(argv[5]);
    int coarsen    = stoi(argv[6]);

    int packed_width = (WIDTH + 31) / 32;

    string outputFolder = "output_gpu_v4_5x5";
    fs::create_directory(outputFolder);

    vector<uint8_t> flat(WIDTH * HEIGHT, 0);

    // 預設：隨機初始（約 10% 存活）
    mt19937 rng(67);
    uniform_int_distribution<int> dist(0, 12);
    for (auto& cell : flat) cell = (dist(rng) == 0) ? 1 : 0;

    // 圓形初始（註解掉上面的隨機初始後使用）
    // int cx = WIDTH / 2, cy = HEIGHT / 2, r = min(WIDTH, HEIGHT) / 4;
    // for (int y = 0; y < HEIGHT; y++)
    //     for (int x = 0; x < WIDTH; x++)
    //         flat[y * WIDTH + x] = ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= r*r) ? 1 : 0;

    vector<uint32_t> packed;
    packGrid(flat, packed, WIDTH, HEIGHT);
    saveGrid(flat, 0, outputFolder, WIDTH, HEIGHT);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t *d_cur, *d_nxt;
    size_t bytes = (size_t)packed_width * HEIGHT * sizeof(uint32_t);
    cudaMalloc(&d_cur, bytes);
    cudaMalloc(&d_nxt, bytes);
    cudaMemcpy(d_cur, packed.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((packed_width + BLOCK_X * coarsen - 1) / (BLOCK_X * coarsen),
              (HEIGHT       + BLOCK_Y             - 1) /  BLOCK_Y);

    cudaEventRecord(start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        conwayKernel5x5<<<grid, block>>>(d_cur, d_nxt, packed_width, HEIGHT, coarsen);

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
    cout << "Throughput: "
         << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6
         << " Gcells/s\n";

    cudaFree(d_cur);
    cudaFree(d_nxt);
    return 0;
}

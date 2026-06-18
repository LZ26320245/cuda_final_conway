#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include <filesystem>
#include <random>
#include <iomanip>
#include <sstream>
using namespace std;
namespace fs = std::filesystem;

// ─── Grid layout ─────────────────────────────────────────────────────────────
// Two arrays per grid:
//   energy[y*W+x]   : 1~15 = alive, 0 = dead
//   potential[y*W+x]: dead cell's accumulated potential (only meaningful when energy==0)

// ─── Host helpers ─────────────────────────────────────────────────────────────

void saveGrid(const vector<uint8_t>& energy,
              int iteration, const string& folder,
              int width, int height)
{
    ostringstream filename;
    filename << folder << "/iter_" << setw(3) << setfill('0') << iteration << ".txt";
    ofstream out(filename.str());
    const char hex[] = "0123456789abcdef";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            out << hex[energy[y * width + x] & 0xF];
        out << '\n';
    }
}

// ─── Kernel ───────────────────────────────────────────────────────────────────
//
// One thread per cell.
// All steps run in a single kernel pass using only the INPUT arrays (cur_energy,
// cur_potential) to compute the OUTPUT arrays (nxt_energy, nxt_potential).
//
// Step ordering (all based on cur_* values, no mid-step races):
//   1. Maintenance cost        : alive energy -= 1
//   2. Neighbour energy bonus  : based on alive-neighbour count
//   3. Energy cap              : min(energy, 15)
//   4. Death + scatter         : energy <= 0 → dead
//                                  scatter is applied ADDITIONALLY from each
//                                  dying neighbour of this cell (we read
//                                  cur_energy to detect who dies this step)
//   5. Potential decay         : dead cell potential -= 1 (but NOT for cells
//                                that just died this step — they start fresh)
//   6. Birth                   : dead cell checks potential energy = neighbours*2 + potential
//                                  8 <= pe <= 14 → born, energy = min(pe-2, 15)

// HALO = 5:
// tentative(dx,dy) 內部讀 se(dx+dx2, dy+dy2)，dx2 最大 1，所以最遠讀到 dx+1
// alive_in_5x5 計算時呼叫 tentative(dx+dx2, dy+dy2)，dx 最大 2，dx2 最大 2，
// tentative 內部再讀 dx3 最大 1，所以最遠讀到 2+2+1 = 5
#define HALO 5

__global__ void energyLifeKernel(
    const uint8_t* __restrict__ cur_energy,
    const uint8_t* __restrict__ cur_potential,
          uint8_t* __restrict__ nxt_energy,
          uint8_t* __restrict__ nxt_potential,
    int width, int height)
{
    // ── Shared memory ─────────────────────────────────────────────────────────
    // tile 大小 = (BLOCK_X + 2*HALO) × (BLOCK_Y + 2*HALO)
    // 兩個 tile 放在同一塊 shared memory，energy 在前，potential 在後
    extern __shared__ uint8_t smem[];

    int sw = blockDim.x + 2 * HALO;
    int sh = blockDim.y + 2 * HALO;
    int tileSize = sw * sh;

    uint8_t* tile_e = smem;
    uint8_t* tile_p = smem + tileSize;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    // ── 載入兩個 tile（含 halo，循環邊界）────────────────────────────────────
    for (int i = tid; i < tileSize; i += nthreads)
    {
        int sx = i % sw;
        int sy = i / sw;

        int col = blockIdx.x * blockDim.x + sx - HALO;
        int row = blockIdx.y * blockDim.y + sy - HALO;

        if (col < 0)           col += width;
        else if (col >= width)  col -= width;
        if (row < 0)           row += height;
        else if (row >= height) row -= height;

        int gidx = row * width + col;
        tile_e[i] = cur_energy[gidx];
        tile_p[i] = cur_potential[gidx];
    }

    __syncthreads();

    // ── 全域座標 ─────────────────────────────────────────────────────────────
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // shared memory 內的中心座標（加上 halo offset）
    int sx0 = tx + HALO;
    int sy0 = ty + HALO;

    // shared memory 讀取 helper：offset (dx,dy) 相對於當前 cell
    auto se = [&](int dx, int dy) -> int {
        return tile_e[(sy0 + dy) * sw + (sx0 + dx)];
    };

    uint8_t my_energy    = (uint8_t)se(0, 0);
    uint8_t my_potential = tile_p[sy0 * sw + sx0];

    // ── tentative：預測鄰居 (dx,dy) 跑完 Step1~3 後的能量 ───────────────────
    // 最遠存取 se(dx+1, dy+1)，dx 最大 4（由 alive_in_5x5 呼叫時帶入 dx+dx2），
    // +1 = 5，剛好在 HALO=5 的範圍內
    auto tentative = [&](int dx, int dy) -> int {
        int e = se(dx, dy);
        if (e == 0) return 0;

        // 計算那個鄰居自己的 3×3 活鄰居數
        int nc = 0;
        for (int dy2 = -1; dy2 <= 1; dy2++)
        for (int dx2 = -1; dx2 <= 1; dx2++) {
            if (dx2 == 0 && dy2 == 0) continue;
            if (se(dx + dx2, dy + dy2) > 0) nc++;
        }

        int te = e - 4;
        if      (nc == 0 || nc >= 7) te /= 2;
        else if (nc >= 1 && nc <= 4) te += 2;
        else                          te += 5;  // 5~6
        if (te > 15) te = 15;
        return te;
    };

    // ── Step 2 用的 3×3 活鄰居數 ─────────────────────────────────────────────
    int alive_count = 0;
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;
        if (se(dx, dy) > 0) alive_count++;
    }

    // ── 5×5 death scatter ─────────────────────────────────────────────────────
    int scatter_energy = 0;
    int potential_gain = 0;

    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
        if (dx == 0 && dy == 0) continue;
        if (se(dx, dy) == 0) continue;  // already dead

        int te = tentative(dx, dy);
        if (te > 0) continue;           // not dying this step

        // 計算這個死亡鄰居的 5×5 內還活著的 cell 數
        int alive_in_5x5 = 0;
        for (int dy2 = -2; dy2 <= 2; dy2++)
        for (int dx2 = -2; dx2 <= 2; dx2++) {
            if (dx2 == 0 && dy2 == 0) continue;
            if (tentative(dx + dx2, dy + dy2) > 0) alive_in_5x5++;
        }

        if (alive_in_5x5 > 0 && my_energy > 0)
            scatter_energy += 4 / alive_in_5x5;

        if (abs(dx) <= 1 && abs(dy) <= 1 && my_energy == 0)
            potential_gain += 2;
    }

    // ── 計算下一狀態 ──────────────────────────────────────────────────────────
    if (my_energy > 0) {
        int e = my_energy - 4;

        if      (alive_count == 0 || alive_count >= 7) e /= 2;
        else if (alive_count >= 1 && alive_count <= 4)  e += 2;
        else                                             e += 5;

        if (e > 15) e = 15;

        if (e <= 0) {
            nxt_energy[idx]    = 0;
            nxt_potential[idx] = 0;
        } else {
            e += scatter_energy;
            if (e > 15) e = 15;
            nxt_energy[idx]    = (uint8_t)e;
            nxt_potential[idx] = 0;
        }
    } else {
        int p = (int)my_potential + potential_gain;
        p = max(0, p - 1);

        int pe = alive_count * 2 + p;
        if (pe >= 8 && pe <= 14) {
            int born_energy = pe - 2;
            if (born_energy > 15) born_energy = 15;
            nxt_energy[idx]    = (uint8_t)born_energy;
            nxt_potential[idx] = 0;
        } else {
            nxt_energy[idx]    = 0;
            nxt_potential[idx] = (uint8_t)min(p, 255);
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc != 7) {
        cout << "Usage: " << argv[0]
             << " WIDTH HEIGHT ITERATIONS BLOCK_X BLOCK_Y INIT_MODE\n"
             << "  INIT_MODE: 0=random(~8%), 1=random(50%), 2=two blocks\n";
        return 1;
    }

    int WIDTH      = stoi(argv[1]);
    int HEIGHT     = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X    = stoi(argv[4]);
    int BLOCK_Y    = stoi(argv[5]);
    int INIT_MODE  = stoi(argv[6]);

    string outputFolder = "output_energy_shared";
    fs::create_directory(outputFolder);

    size_t N = (size_t)WIDTH * HEIGHT;
    vector<uint8_t> h_energy(N, 0);
    vector<uint8_t> h_potential(N, 0);

    // ── Initialisation ────────────────────────────────────────────────────────
    mt19937 rng(67);

    if (INIT_MODE == 0) {
        // ~8% alive
        uniform_int_distribution<int> dist(0, 12);
        for (int i = 0; i < (int)N; i++)
            if (dist(rng) == 0) h_energy[i] = 10;
    }
    else if (INIT_MODE == 1) {
        // ~50% alive
        uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < (int)N; i++)
            if (dist(rng) == 0) h_energy[i] = 10;
    }
    else if (INIT_MODE == 2){
        int cx = WIDTH / 2, cy = HEIGHT / 2, r = min(WIDTH, HEIGHT) / 4;
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                h_energy[y * WIDTH + x] = ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= r*r) ? 1 : 0;
    }
    else {
        // Two solid blocks: top-left and bottom-right
        int bs = min(WIDTH, HEIGHT) / 4;
        int gap = 15;  // 兩個方塊之間的間距，可調整

        // 兩個方塊沿對角線排列，以畫面中心為基準往兩側偏移
        // 左上方塊：右下角在中心左上方 gap/2 處
        int tl_x = WIDTH  / 2 - gap / 2 - bs;
        int tl_y = HEIGHT / 2 - gap / 2 - bs;

        // 右下方塊：左上角在中心右下方 gap/2 處
        int br_x = WIDTH  / 2 + gap / 2;
        int br_y = HEIGHT / 2 + gap / 2;

        for (int y = tl_y; y < tl_y + bs; y++)
            for (int x = tl_x; x < tl_x + bs; x++)
                h_energy[y * WIDTH + x] = 10;

        for (int y = br_y; y < br_y + bs; y++)
            for (int x = br_x; x < br_x + bs; x++)
                h_energy[y * WIDTH + x] = 10;
    }

    saveGrid(h_energy, 0, outputFolder, WIDTH, HEIGHT);

    // ── CUDA setup ────────────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    uint8_t *d_energy_cur, *d_energy_nxt;
    uint8_t *d_potential_cur, *d_potential_nxt;

    cudaMalloc(&d_energy_cur,    N);
    cudaMalloc(&d_energy_nxt,    N);
    cudaMalloc(&d_potential_cur, N);
    cudaMalloc(&d_potential_nxt, N);

    cudaMemcpy(d_energy_cur,    h_energy.data(),    N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_potential_cur, h_potential.data(), N, cudaMemcpyHostToDevice);
    cudaMemset(d_energy_nxt,    0, N);
    cudaMemset(d_potential_nxt, 0, N);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((WIDTH  + BLOCK_X - 1) / BLOCK_X,
              (HEIGHT + BLOCK_Y - 1) / BLOCK_Y);

    // tile = (BLOCK_X + 2*5) * (BLOCK_Y + 2*5)，兩個 tile：energy + potential
    // 以 16x16 block 為例：2 * 26 * 26 = 1352 bytes，遠小於 48KB 上限
    size_t sharedBytes = 2 * (size_t)(BLOCK_X + 2*HALO) * (BLOCK_Y + 2*HALO) * sizeof(uint8_t);

    // ── Simulation loop ───────────────────────────────────────────────────────
    cudaEventRecord(ev_start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        energyLifeKernel<<<grid, block, sharedBytes>>>(d_energy_cur, d_potential_cur, d_energy_nxt, d_potential_nxt, WIDTH, HEIGHT);

        cudaDeviceSynchronize();

        swap(d_energy_cur,    d_energy_nxt);
        swap(d_potential_cur, d_potential_nxt);

        // cudaMemcpy(h_energy.data(), d_energy_cur, N, cudaMemcpyDeviceToHost);
        // saveGrid(h_energy, iter, outputFolder, WIDTH, HEIGHT);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, ev_start, ev_stop);
    cout << "\nExecution Time: " << elapsed << " ms\n";
    cout << "Throughput: " << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6 << " Gcells/s\n";

    cudaFree(d_energy_cur);    cudaFree(d_energy_nxt);
    cudaFree(d_potential_cur); cudaFree(d_potential_nxt);
    return 0;
}

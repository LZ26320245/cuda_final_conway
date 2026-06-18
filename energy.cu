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

__global__ void energyLifeKernel(
    const uint8_t* __restrict__ cur_energy,
    const uint8_t* __restrict__ cur_potential,
          uint8_t* __restrict__ nxt_energy,
          uint8_t* __restrict__ nxt_potential,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    uint8_t my_energy    = cur_energy[idx];
    uint8_t my_potential = cur_potential[idx];

    // ── Count alive neighbours & gather death-scatter from 3×3 and 5×5 ──────

    int alive_count = 0;   // alive neighbours in 3×3

    // Count for step 4 scatter: how many cells in 5×5 around me are dying?
    // A cell dies if: it is alive now (cur_energy > 0) AND
    //                 its net energy after steps 1+2 would be <= 0.
    // We need to compute that tentative energy for each neighbour.

    int scatter_energy = 0;   // total 9-pt scatter I receive from dying 5×5 neighbours
    int potential_gain = 0;   // 3 pts per dying 3×3 neighbour (for dead cells)

    // Helper: tentative energy of a neighbour after steps 1+2+3
    // (used to decide if it dies this step)
    // Returns 0 if the neighbour is already dead.
    auto tentative = [&](int nx, int ny) -> int {
        // wrap
        if (nx < 0) nx += width;  else if (nx >= width)  nx -= width;
        if (ny < 0) ny += height; else if (ny >= height) ny -= height;
        int nidx = ny * width + nx;
        int e = cur_energy[nidx];
        if (e == 0) return 0;  // already dead

        // count that neighbour's alive neighbours (its own 3x3, not ours)
        int nc = 0;
        for (int dy2 = -1; dy2 <= 1; dy2++)
        for (int dx2 = -1; dx2 <= 1; dx2++) {
            if (dx2 == 0 && dy2 == 0) continue;
            int nnx = nx + dx2, nny = ny + dy2;
            if (nnx < 0) nnx += width;  else if (nnx >= width)  nnx -= width;
            if (nny < 0) nny += height; else if (nny >= height) nny -= height;
            if (cur_energy[nny * width + nnx] > 0) nc++;
        }

        // step 1
        int te = e - 4;
        // step 2
        if      (nc == 0 || nc >= 7) te /= 2;
        else if (nc >= 1 && nc <= 4) te += 2;
        else                          te += 5;  // 4~6
        // step 3
        if (te > 15) te = 15;
        return te;
    };

    // ── 3×3 alive count (for this cell's own step 2) ─────────────────────────
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;
        int nx = x + dx, ny = y + dy;
        if (nx < 0) nx += width;  else if (nx >= width)  nx -= width;
        if (ny < 0) ny += height; else if (ny >= height) ny -= height;
        if (cur_energy[ny * width + nx] > 0) alive_count++;
    }

    // ── 5×5 death scatter (step 4) ────────────────────────────────────────────
    // Count dying cells in 5×5; each one contributes floor(9 / dying_count_in_its_5x5)
    // to surviving neighbours.  For simplicity (and GPU friendliness) we use the
    // fixed split: each dying cell gives floor(9 / alive_neighbours_in_5x5) to ME,
    // where alive_neighbours_in_5x5 is the count of alive cells in the dying cell's
    // own 5×5 that are still alive (i.e. not dying themselves).
    //
    // This is an approximation that avoids a two-pass algorithm while keeping
    // the spirit of the rule.  The exact "平分" would require knowing how many
    // of the 5×5 survive, which requires the same tentative computation for all
    // 24 neighbours — expensive but doable.  We use it here for correctness.

    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
        if (dx == 0 && dy == 0) continue;
        int nx = x + dx, ny = y + dy;
        if (nx < 0) nx += width;  else if (nx >= width)  nx -= width;
        if (ny < 0) ny += height; else if (ny >= height) ny -= height;
        int nidx = ny * width + nx;

        int ne = cur_energy[nidx];
        if (ne == 0) continue;  // already dead, skip

        int te = tentative(nx, ny);
        if (te > 0) continue;   // not dying this step

        // This neighbour is dying.  Count how many alive (non-dying) cells are
        // in its 5×5 to determine the share.
        int alive_in_5x5 = 0;
        for (int dy2 = -2; dy2 <= 2; dy2++)
        for (int dx2 = -2; dx2 <= 2; dx2++) {
            if (dx2 == 0 && dy2 == 0) continue;
            int nnx = nx + dx2, nny = ny + dy2;
            if (nnx < 0) nnx += width;  else if (nnx >= width)  nnx -= width;
            if (nny < 0) nny += height; else if (nny >= height) nny -= height;
            int te2 = tentative(nnx, nny);
            if (te2 > 0) alive_in_5x5++;
        }

        if (alive_in_5x5 > 0 && my_energy > 0) {
            // I am alive and this dying cell scatters to me
            scatter_energy += 4 / alive_in_5x5;  // floor division
        }

        // 3×3 potential gain for dead cells
        bool in_3x3 = (abs(dx) <= 1 && abs(dy) <= 1);
        if (in_3x3 && my_energy == 0) {
            potential_gain += 2;
        }
    }

    // ── Now compute this cell's next state ────────────────────────────────────

    if (my_energy > 0) {
        // ── ALIVE cell ────────────────────────────────────────────────────────

        // Step 1
        int e = my_energy - 4;

        // Step 2
        if      (alive_count == 0 || alive_count >= 7) e /= 2;
        else if (alive_count >= 1 && alive_count <= 4)  e += 2;
        else                                             e += 5;  // 5~6

        // Step 3
        if (e > 15) e = 15;

        // Step 4: death check (before scatter)
        if (e <= 0) {
            // Dies this step — start as dead, potential = 0 (not decayed)
            nxt_energy[idx]    = 0;
            nxt_potential[idx] = 0;
        } else {
            // Survives: add scatter from dying 5×5 neighbours
            e += scatter_energy;
            if (e > 15) e = 15;
            nxt_energy[idx]    = (uint8_t)e;
            nxt_potential[idx] = 0;  // alive cells don't have potential
        }

    } else {
        // ── DEAD cell ─────────────────────────────────────────────────────────

        // Step 5: potential decay (only for cells that were ALREADY dead)
        int p = (int)my_potential + potential_gain;
        p = max(0, p - 1);  // decay by 1 this step

        // Step 6: birth check
        int pe = alive_count * 2 + p;
        if (pe >= 8 && pe <= 14) {
            // Born
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

    string outputFolder = "output_energy";
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
        int tl_x = WIDTH  / 4 - bs / 2, tl_y = HEIGHT / 4 - bs / 2;
        int br_x = 3*WIDTH/ 4 - bs / 2, br_y = 3*HEIGHT/4 - bs / 2;
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

    // ── Simulation loop ───────────────────────────────────────────────────────
    cudaEventRecord(ev_start);

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        energyLifeKernel<<<grid, block>>>(
            d_energy_cur, d_potential_cur,
            d_energy_nxt, d_potential_nxt,
            WIDTH, HEIGHT);

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

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

// ── 計算鄰居的 tentative energy（跑完 Step1~3 後的預測值）────────────────────
int tentative(const vector<uint8_t>& cur_energy,
              int nx, int ny, int width, int height)
{
    if (nx < 0) nx += width;  else if (nx >= width)  nx -= width;
    if (ny < 0) ny += height; else if (ny >= height) ny -= height;

    int e = cur_energy[ny * width + nx];
    if (e == 0) return 0;

    // 計算那個鄰居自己的 3×3 活鄰居數
    int nc = 0;
    for (int dy2 = -1; dy2 <= 1; dy2++)
    for (int dx2 = -1; dx2 <= 1; dx2++) {
        if (dx2 == 0 && dy2 == 0) continue;
        int nnx = nx + dx2, nny = ny + dy2;
        if (nnx < 0) nnx += width;  else if (nnx >= width)  nnx -= width;
        if (nny < 0) nny += height; else if (nny >= height) nny -= height;
        if (cur_energy[nny * width + nnx] > 0) nc++;
    }

    int te = e - 4;
    if      (nc == 0 || nc >= 7) te /= 2;
    else if (nc >= 1 && nc <= 4) te += 2;
    else                          te += 5;  // 5~6
    if (te > 15) te = 15;
    return te;
}

int main(int argc, char* argv[])
{
    if (argc != 7) {
        cout << "Usage: " << argv[0]
             << " WIDTH HEIGHT ITERATIONS INIT_MODE\n"
             << "  INIT_MODE: 0=random(~8%), 1=random(50%), 2=circle, 3=two blocks\n";
        return 1;
    }

    int WIDTH      = stoi(argv[1]);
    int HEIGHT     = stoi(argv[2]);
    int ITERATIONS = stoi(argv[3]);
    int BLOCK_X    = stoi(argv[4]);  // CPU 版不用，保留讓參數一致
    int BLOCK_Y    = stoi(argv[5]);  // CPU 版不用，保留讓參數一致
    int INIT_MODE  = stoi(argv[6]);

    (void)BLOCK_X; (void)BLOCK_Y;

    string outputFolder = "output_energy_cpu";
    fs::create_directory(outputFolder);

    size_t N = (size_t)WIDTH * HEIGHT;
    vector<uint8_t> cur_energy(N, 0);
    vector<uint8_t> cur_potential(N, 0);
    vector<uint8_t> nxt_energy(N, 0);
    vector<uint8_t> nxt_potential(N, 0);

    // ── 初始化（和 GPU 版本完全相同）────────────────────────────────────────
    mt19937 rng(67);

    if (INIT_MODE == 0) {
        uniform_int_distribution<int> dist(0, 12);
        for (int i = 0; i < (int)N; i++)
            if (dist(rng) == 0) cur_energy[i] = 10;
    }
    else if (INIT_MODE == 1) {
        uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < (int)N; i++)
            if (dist(rng) == 0) cur_energy[i] = 10;
    }
    else if (INIT_MODE == 2) {
        int cx = WIDTH / 2, cy = HEIGHT / 2, r = min(WIDTH, HEIGHT) / 4;
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                cur_energy[y * WIDTH + x] =
                    ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= r*r) ? 10 : 0;
    }
    else {
        int bs = min(WIDTH, HEIGHT) / 4;
        int tl_x = WIDTH  / 4 - bs / 2, tl_y = HEIGHT / 4 - bs / 2;
        int br_x = 3*WIDTH/ 4 - bs / 2, br_y = 3*HEIGHT/4 - bs / 2;
        for (int y = tl_y; y < tl_y + bs; y++)
            for (int x = tl_x; x < tl_x + bs; x++)
                cur_energy[y * WIDTH + x] = 10;
        for (int y = br_y; y < br_y + bs; y++)
            for (int x = br_x; x < br_x + bs; x++)
                cur_energy[y * WIDTH + x] = 10;
    }

    saveGrid(cur_energy, 0, outputFolder, WIDTH, HEIGHT);

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= ITERATIONS; iter++)
    {
        for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH;  x++)
        {
            int idx        = y * WIDTH + x;
            int my_energy    = cur_energy[idx];
            int my_potential = cur_potential[idx];

            // ── Step 2 用的 3×3 活鄰居數 ─────────────────────────────────────
            int alive_count = 0;
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny2 = y + dy;
                if (nx < 0) nx += WIDTH;  else if (nx >= WIDTH)  nx -= WIDTH;
                if (ny2 < 0) ny2 += HEIGHT; else if (ny2 >= HEIGHT) ny2 -= HEIGHT;
                if (cur_energy[ny2 * WIDTH + nx] > 0) alive_count++;
            }

            // ── 5×5 death scatter ─────────────────────────────────────────────
            int scatter_energy = 0;
            int potential_gain = 0;

            for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny2 = y + dy;
                if (nx < 0) nx += WIDTH;  else if (nx >= WIDTH)  nx -= WIDTH;
                if (ny2 < 0) ny2 += HEIGHT; else if (ny2 >= HEIGHT) ny2 -= HEIGHT;

                if (cur_energy[ny2 * WIDTH + nx] == 0) continue;

                int te = tentative(cur_energy, nx, ny2, WIDTH, HEIGHT);
                if (te > 0) continue;  // not dying

                // 計算這個死亡鄰居的 5×5 內還活著的 cell 數
                int alive_in_5x5 = 0;
                for (int dy2 = -2; dy2 <= 2; dy2++)
                for (int dx2 = -2; dx2 <= 2; dx2++) {
                    if (dx2 == 0 && dy2 == 0) continue;
                    int nnx = nx + dx2, nny = ny2 + dy2;
                    if (nnx < 0) nnx += WIDTH;  else if (nnx >= WIDTH)  nnx -= WIDTH;
                    if (nny < 0) nny += HEIGHT; else if (nny >= HEIGHT) nny -= HEIGHT;
                    if (tentative(cur_energy, nnx, nny, WIDTH, HEIGHT) > 0)
                        alive_in_5x5++;
                }

                if (alive_in_5x5 > 0 && my_energy > 0)
                    scatter_energy += 4 / alive_in_5x5;

                if (abs(dx) <= 1 && abs(dy) <= 1 && my_energy == 0)
                    potential_gain += 2;
            }

            // ── 計算下一狀態 ──────────────────────────────────────────────────
            if (my_energy > 0) {
                // Step 1
                int e = my_energy - 4;
                // Step 2
                if      (alive_count == 0 || alive_count >= 7) e /= 2;
                else if (alive_count >= 1 && alive_count <= 4)  e += 2;
                else                                             e += 5;
                // Step 3
                if (e > 15) e = 15;
                // Step 4
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
                // Step 5
                int p = my_potential + potential_gain;
                p = max(0, p - 1);
                // Step 6
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

        // saveGrid(nxt_energy, iter, outputFolder, WIDTH, HEIGHT);

        swap(cur_energy,    nxt_energy);
        swap(cur_potential, nxt_potential);
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double, milli>(end - start).count();
    cout << "\nExecution Time: " << elapsed << " ms\n";
    cout << "Throughput: " << (double)WIDTH * HEIGHT * ITERATIONS / elapsed / 1e6 << " Gcells/s\n";

    return 0;
}
// open-physicell computation kernels.
//
// Stateless C-ABI functions over caller-owned, C-contiguous double arrays.
// Python owns all state and the time loop; these are the inner loops only.
//
// Parallelism: OpenMP when the compiler provides it, otherwise an equivalent
// std::thread fan-out, so the kernels are parallel on every platform (Apple
// clang ships no OpenMP runtime).
//
// Numerics follow PhysiCell 1.14.2 / BioFVM exactly:
//   - diffusion: locally one-dimensional (LOD) implicit split, Thomas algorithm
//     per line  (BioFVM_solvers.cpp)
//   - mechanics: Cell::add_potentials  (PhysiCell_cell.cpp)
//   - positions: second-order Adams-Bashforth  (Cell::update_position)
//   - secretion/uptake: the BioFVM semi-implicit cell source/sink solver

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#else
#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#endif

#if defined(_MSC_VER)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {

#if !defined(_OPENMP)
// A persistent worker pool.  The kernels are called hundreds of thousands of
// times in a run, so threads are created once for the process, not per call:
// spawning a pool per call costs more than the work and eventually exhausts the
// process thread limit.
class Pool {
 public:
  static Pool& get() {
    static Pool pool;
    return pool;
  }

  int size() const { return static_cast<int>(workers_.size()) + 1; }  // + the caller

  void run(long n, const std::function<void(long, long)>& chunk) {
    const long nthreads = size();
    const long block = (n + nthreads - 1) / nthreads;
    {
      std::lock_guard<std::mutex> lock(m_);
      task_ = &chunk;
      next_ = block;        // the caller takes [0, block)
      end_ = n;
      block_ = block;
      pending_ = static_cast<long>(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    work(0, std::min(block, n));                     // the caller works too
    std::unique_lock<std::mutex> lock(m_);
    done_cv_.wait(lock, [this] { return pending_ == 0; });
    task_ = nullptr;
  }

 private:
  Pool() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw < 2) hw = 1;
    // An ensemble run (a sweep, a replicate set) wants one thread per process,
    // not one per core per process.  PHYSICELL_THREADS caps the pool.
    if (const char* cap = std::getenv("PHYSICELL_THREADS")) {
      long want = std::strtol(cap, nullptr, 10);
      if (want > 0 && (unsigned)want < hw) hw = (unsigned)want;
    }
    for (unsigned i = 1; i < hw; ++i) workers_.emplace_back([this] { loop(); });
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
  }

  void work(long lo, long hi) const {
    if (lo < hi) (*task_)(lo, hi);
  }

  void loop() {
    unsigned seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(m_);
      cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      for (;;) {
        const long lo = next_;
        if (lo >= end_) break;
        next_ = lo + block_;
        const long hi = std::min(lo + block_, end_);
        lock.unlock();
        (*task_)(lo, hi);
        lock.lock();
      }
      if (--pending_ == 0) {
        lock.unlock();
        done_cv_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  mutable std::mutex m_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(long, long)>* task_ = nullptr;
  long next_ = 0, end_ = 0, block_ = 1, pending_ = 0;
  unsigned generation_ = 0;
  bool stop_ = false;
};
#endif

// parallel_for over [0, n): OpenMP where available, the pool above otherwise.
template <typename F>
void parallel_for(long n, const F& body) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) body(i);
#else
  if (n < 64 || Pool::get().size() < 2) {      // not worth waking the pool
    for (long i = 0; i < n; ++i) body(i);
    return;
  }
  std::function<void(long, long)> chunk = [&body](long lo, long hi) {
    for (long i = lo; i < hi; ++i) body(i);
  };
  Pool::get().run(n, chunk);
#endif
}

inline long vox(long i, long j, long k, long nx, long ny) { return (k * ny + j) * nx + i; }

}  // namespace

API int pc_has_openmp(void) {
#if defined(_OPENMP)
  return 1;
#else
  return 0;
#endif
}

API int pc_max_threads(void) {
#if defined(_OPENMP)
  if (const char* cap = std::getenv("PHYSICELL_THREADS")) {
    long want = std::strtol(cap, nullptr, 10);
    if (want > 0 && want < omp_get_max_threads()) { omp_set_num_threads((int)want); }
  }
  return omp_get_max_threads();
#else
  return Pool::get().size();
#endif
}

// ---------------------------------------------------------------------------
// Diffusion-decay: one LOD sweep along one axis.
//
// rho       : (nz, ny, nx, ns) C-contiguous densities, updated in place
// denom, cc : (n_axis, ns) Thomas coefficients for this axis (built in Python,
//             following BioFVM's setup so they can be checked there)
// c1        : (ns) dt*D/h^2 for this axis
// axis      : 0 = x, 1 = y, 2 = z
// ---------------------------------------------------------------------------
API void pc_thomas_sweep(double* rho, long nx, long ny, long nz, long ns, int axis,
                         const double* denom, const double* cc, const double* c1) {
  long nline, n_outer_a, n_outer_b, stride;
  if (axis == 0) { nline = nx; n_outer_a = ny; n_outer_b = nz; stride = ns; }
  else if (axis == 1) { nline = ny; n_outer_a = nx; n_outer_b = nz; stride = nx * ns; }
  else { nline = nz; n_outer_a = nx; n_outer_b = ny; stride = nx * ny * ns; }

  parallel_for(n_outer_a * n_outer_b, [&](long t) {
    const long a = t % n_outer_a, b = t / n_outer_a;
    long base;
    if (axis == 0) base = vox(0, a, b, nx, ny) * ns;
    else if (axis == 1) base = vox(a, 0, b, nx, ny) * ns;
    else base = vox(a, b, 0, nx, ny) * ns;

    // forward elimination with precomputed coefficients
    for (long s = 0; s < ns; ++s) rho[base + s] /= denom[0 * ns + s];
    for (long m = 1; m < nline; ++m) {
      double* cur = rho + base + m * stride;
      const double* prev = cur - stride;
      for (long s = 0; s < ns; ++s) {
        cur[s] += c1[s] * prev[s];
        cur[s] /= denom[m * ns + s];
      }
    }
    // back substitution
    for (long m = nline - 2; m >= 0; --m) {
      double* cur = rho + base + m * stride;
      const double* next = cur + stride;
      for (long s = 0; s < ns; ++s) cur[s] -= cc[m * ns + s] * next[s];
    }
  });
}

// Overwrite Dirichlet voxels. idx: (n) voxel indices, val: (n, ns) values,
// mask: (n, ns) 1 where the condition is active for that substrate.
API void pc_apply_dirichlet(double* rho, long ns, const long* idx, const double* val,
                            const unsigned char* mask, long n) {
  parallel_for(n, [&](long m) {
    double* cur = rho + idx[m] * ns;
    const double* v = val + m * ns;
    const unsigned char* mk = mask + m * ns;
    for (long s = 0; s < ns; ++s)
      if (mk[s]) cur[s] = v[s];
  });
}

// Central-difference gradients, one-sided at the faces.  grad: (nz,ny,nx,ns,3)
API void pc_gradients(const double* rho, long nx, long ny, long nz, long ns,
                      double dx, double dy, double dz, double* grad) {
  const long sx = ns, sy = nx * ns, sz = nx * ny * ns;
  parallel_for(nz * ny, [&](long t) {
    const long j = t % ny, k = t / ny;
    for (long i = 0; i < nx; ++i) {
      const long n = vox(i, j, k, nx, ny);
      const double* c = rho + n * ns;
      double* g = grad + n * ns * 3;
      for (long s = 0; s < ns; ++s) {
        double gxv = 0.0, gyv = 0.0, gzv = 0.0;
        if (nx > 1) {
          if (i == 0)            gxv = (c[s + sx] - c[s]) / dx;
          else if (i == nx - 1)  gxv = (c[s] - c[s - sx]) / dx;
          else                   gxv = (c[s + sx] - c[s - sx]) / (2.0 * dx);
        }
        if (ny > 1) {
          if (j == 0)            gyv = (c[s + sy] - c[s]) / dy;
          else if (j == ny - 1)  gyv = (c[s] - c[s - sy]) / dy;
          else                   gyv = (c[s + sy] - c[s - sy]) / (2.0 * dy);
        }
        if (nz > 1) {
          if (k == 0)            gzv = (c[s + sz] - c[s]) / dz;
          else if (k == nz - 1)  gzv = (c[s] - c[s - sz]) / dz;
          else                   gzv = (c[s + sz] - c[s - sz]) / (2.0 * dz);
        }
        g[s * 3 + 0] = gxv;
        g[s * 3 + 1] = gyv;
        g[s * 3 + 2] = gzv;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Cell secretion / uptake, BioFVM's semi-implicit solver at the cell's voxel:
//     rho <- (rho + c1) / c2 + export/W,      c1 = (dt V/W) S T,
//                                             c2 = 1 + (dt V/W)(S + U)
// Serial over cells on purpose: several cells share a voxel, and PhysiCell's
// own parallel loop over cells races there.  This is cheap and deterministic.
// ---------------------------------------------------------------------------
API void pc_secretion_uptake(double* rho, long ns, const long* cell_voxel, const double* cell_volume,
                             const double* S, const double* U, const double* T, const double* net_export,
                             long ncells, double voxel_volume, double dt, double* internalized) {
  for (long c = 0; c < ncells; ++c) {
    const double f = dt * cell_volume[c] / voxel_volume;
    double* r = rho + cell_voxel[c] * ns;
    const double* Sc = S + c * ns;
    const double* Uc = U + c * ns;
    const double* Tc = T + c * ns;
    const double* Ec = net_export + c * ns;
    for (long s = 0; s < ns; ++s) {
      const double c1 = f * Sc[s] * Tc[s];
      const double c2 = 1.0 + f * (Sc[s] + Uc[s]);
      const double before = r[s];
      double after = (before + c1) / c2;
      const double exported = dt * Ec[s] / voxel_volume;
      after += exported;
      if (internalized) internalized[c * ns + s] -= (after - before) * voxel_volume;
      r[s] = after;
    }
  }
}

// Nearest-voxel index for each cell (uniform Cartesian mesh).
API void pc_voxel_of(const double* pos, long n, double x0, double y0, double z0,
                     double dx, double dy, double dz, long nx, long ny, long nz, long* out) {
  parallel_for(n, [&](long c) {
    long i = static_cast<long>((pos[c * 3 + 0] - x0) / dx);
    long j = static_cast<long>((pos[c * 3 + 1] - y0) / dy);
    long k = static_cast<long>((pos[c * 3 + 2] - z0) / dz);
    i = i < 0 ? 0 : (i >= nx ? nx - 1 : i);
    j = j < 0 ? 0 : (j >= ny ? ny - 1 : j);
    k = k < 0 ? 0 : (k >= nz ? nz - 1 : k);
    out[c] = vox(i, j, k, nx, ny);
  });
}

// ---------------------------------------------------------------------------
// Mechanics.  Cell list over a uniform grid of side h >= the largest
// interaction distance, then Cell::add_potentials over the 27-voxel
// Moore neighborhood.  Each cell accumulates only its own velocity, so the
// loop is race-free and bitwise-reproducible whatever the thread count.
// ---------------------------------------------------------------------------
API void pc_build_cell_list(const double* pos, long n, double x0, double y0, double z0,
                            double h, long gx, long gy, long gz, long* head, long* next) {
  for (long m = 0; m < gx * gy * gz; ++m) head[m] = -1;
  for (long c = 0; c < n; ++c) {
    long i = static_cast<long>((pos[c * 3 + 0] - x0) / h);
    long j = static_cast<long>((pos[c * 3 + 1] - y0) / h);
    long k = static_cast<long>((pos[c * 3 + 2] - z0) / h);
    i = i < 0 ? 0 : (i >= gx ? gx - 1 : i);
    j = j < 0 ? 0 : (j >= gy ? gy - 1 : j);
    k = k < 0 ? 0 : (k >= gz ? gz - 1 : k);
    const long g = (k * gy + j) * gx + i;
    next[c] = head[g];
    head[g] = c;
  }
}

API void pc_forces(const double* pos, const double* radius, const double* rel_max_adh,
                   const double* adhesion, const double* repulsion, const long* type,
                   const double* affinity, long ntypes, const unsigned char* movable,
                   long n, double x0, double y0, double z0, double h, long gx, long gy, long gz,
                   const long* head, const long* next, int is2d,
                   double* velocity, double* pressure, long* neighbors,
                   long* neighbor_idx, long cap) {
  const double simple_pressure_scale = 0.027288820670331;
  parallel_for(n, [&](long c) {
    velocity[c * 3 + 0] = 0.0; velocity[c * 3 + 1] = 0.0; velocity[c * 3 + 2] = 0.0;
    pressure[c] = 0.0; neighbors[c] = 0;
    if (!movable[c]) return;

    const double xi = pos[c * 3 + 0], yi = pos[c * 3 + 1], zi = pos[c * 3 + 2];
    long i = static_cast<long>((xi - x0) / h), j = static_cast<long>((yi - y0) / h),
         k = static_cast<long>((zi - z0) / h);
    i = i < 0 ? 0 : (i >= gx ? gx - 1 : i);
    j = j < 0 ? 0 : (j >= gy ? gy - 1 : j);
    k = k < 0 ? 0 : (k >= gz ? gz - 1 : k);

    double vx = 0.0, vy = 0.0, vz = 0.0, press = 0.0;
    long nb = 0;
    const long ti = type[c];

    for (long kk = (k > 0 ? k - 1 : 0); kk <= (k + 1 < gz ? k + 1 : gz - 1); ++kk)
      for (long jj = (j > 0 ? j - 1 : 0); jj <= (j + 1 < gy ? j + 1 : gy - 1); ++jj)
        for (long ii = (i > 0 ? i - 1 : 0); ii <= (i + 1 < gx ? i + 1 : gx - 1); ++ii) {
          for (long o = head[(kk * gy + jj) * gx + ii]; o != -1; o = next[o]) {
            if (o == c) continue;
            double ddx = xi - pos[o * 3 + 0];
            double ddy = yi - pos[o * 3 + 1];
            double ddz = zi - pos[o * 3 + 2];
            double d = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            if (d < 1e-5) d = 1e-5;

            const double R = radius[c] + radius[o];
            double temp_r = 0.0;
            if (d <= R) {
              temp_r = 1.0 - d / R;
              temp_r *= temp_r;
              press += temp_r / simple_pressure_scale;
            }
            temp_r *= std::sqrt(repulsion[c] * repulsion[o]);

            const double Smax = rel_max_adh[c] * radius[c] + rel_max_adh[o] * radius[o];
            if (d < Smax) {
              if (neighbor_idx && nb < cap) neighbor_idx[c * cap + nb] = o;
              double temp_a = 1.0 - d / Smax;
              temp_a *= temp_a;
              const long to = type[o];
              // affinity is per cell (a row per cell), so a rule can change it
              const double a_ij = adhesion[c] * affinity[c * ntypes + to];
              const double a_ji = adhesion[o] * affinity[o * ntypes + ti];
              temp_a *= std::sqrt(a_ij * a_ji);
              temp_r -= temp_a;
              ++nb;
            }
            if (std::fabs(temp_r) < 1e-16) continue;
            temp_r /= d;
            vx += temp_r * ddx;
            vy += temp_r * ddy;
            vz += temp_r * ddz;
          }
        }
    velocity[c * 3 + 0] = vx;
    velocity[c * 3 + 1] = vy;
    velocity[c * 3 + 2] = is2d ? 0.0 : vz;
    pressure[c] = press;
    neighbors[c] = nb;
  });
}

// Second-order Adams-Bashforth, as Cell::update_position.
API void pc_update_positions(double* pos, double* vel, double* prev_vel,
                             const unsigned char* movable, long n, double dt, int is2d) {
  const double d1 = 1.5 * dt, d2 = -0.5 * dt;
  parallel_for(n, [&](long c) {
    if (!movable[c]) return;
    for (int a = 0; a < 3; ++a) {
      if (is2d && a == 2) { vel[c * 3 + 2] = 0.0; }
      pos[c * 3 + a] += d1 * vel[c * 3 + a] + d2 * prev_vel[c * 3 + a];
      prev_vel[c * 3 + a] = vel[c * 3 + a];
    }
  });
}

// Sample the field (and its gradient) at each cell's voxel.
API void pc_sample_at_cells(const double* rho, const double* grad, long ns, const long* cell_voxel,
                            long n, double* out_rho, double* out_grad) {
  parallel_for(n, [&](long c) {
    const long v = cell_voxel[c];
    for (long s = 0; s < ns; ++s) {
      out_rho[c * ns + s] = rho[v * ns + s];
      if (out_grad) {
        for (int a = 0; a < 3; ++a) out_grad[(c * ns + s) * 3 + a] = grad[(v * ns + s) * 3 + a];
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Fibres: extended agents (line segments), as PhysiMeSS models the ECM.
//
// A fibre is a centre, a unit orientation, a half-length and a radius.  The
// force law is PhysiCell's own, with the cell-cell distance replaced by the
// distance from the cell to the nearest point on the segment.  Cells and
// fibres are accumulated in separate passes so each output array is written by
// exactly one thread, which keeps the result independent of the thread count.
// ---------------------------------------------------------------------------

namespace {

// Nearest point on the segment [c - L u, c + L u] to p; writes p - nearest.
inline double nearest_on_fibre(const double* p, const double* centre, const double* dir,
                               double half_length, double* disp) {
    double t = 0.0;
    for (int a = 0; a < 3; ++a) t += (p[a] - centre[a]) * dir[a];
    if (t > half_length) t = half_length;
    if (t < -half_length) t = -half_length;
    double d2 = 0.0;
    for (int a = 0; a < 3; ++a) {
        disp[a] = p[a] - (centre[a] + t * dir[a]);
        d2 += disp[a] * disp[a];
    }
    double d = std::sqrt(d2);
    return d < 1e-5 ? 1e-5 : d;
}

inline void grid_cell(const double* p, double x0, double y0, double z0, double h,
                      long gx, long gy, long gz, long& i, long& j, long& k) {
    i = static_cast<long>((p[0] - x0) / h);
    j = static_cast<long>((p[1] - y0) / h);
    k = static_cast<long>((p[2] - z0) / h);
    i = i < 0 ? 0 : (i >= gx ? gx - 1 : i);
    j = j < 0 ? 0 : (j >= gy ? gy - 1 : j);
    k = k < 0 ? 0 : (k >= gz ? gz - 1 : k);
}

}  // namespace

// Pass 1: fibres push cells.  Parallel over cells; `head`/`next` index fibres.
API void pc_fibre_push_cells(const double* cpos, const double* cradius, const double* crepulsion,
                             const unsigned char* cmovable, long ncells,
                             const double* fpos, const double* fdir, const double* fhalf,
                             const double* frad, const double* frep, long nfibres,
                             double x0, double y0, double z0, double h, long gx, long gy, long gz,
                             const long* head, const long* next, int is2d,
                             double* velocity, double* pressure, long* contacts) {
  const double simple_pressure_scale = 0.027288820670331;
  parallel_for(ncells, [&](long c) {
    if (!cmovable[c]) return;
    long i, j, k;
    grid_cell(cpos + c * 3, x0, y0, z0, h, gx, gy, gz, i, j, k);
    double vx = 0.0, vy = 0.0, vz = 0.0, press = 0.0;
    long touched = 0;
    for (long kk = (k > 0 ? k - 1 : 0); kk <= (k + 1 < gz ? k + 1 : gz - 1); ++kk)
      for (long jj = (j > 0 ? j - 1 : 0); jj <= (j + 1 < gy ? j + 1 : gy - 1); ++jj)
        for (long ii = (i > 0 ? i - 1 : 0); ii <= (i + 1 < gx ? i + 1 : gx - 1); ++ii)
          for (long f = head[(kk * gy + jj) * gx + ii]; f != -1; f = next[f]) {
            double disp[3];
            const double d = nearest_on_fibre(cpos + c * 3, fpos + f * 3, fdir + f * 3,
                                              fhalf[f], disp);
            const double R = cradius[c] + frad[f];
            if (d > R) continue;
            double temp_r = 1.0 - d / R;
            temp_r *= temp_r;
            press += temp_r / simple_pressure_scale;
            temp_r *= std::sqrt(crepulsion[c] * frep[f]);
            if (std::fabs(temp_r) < 1e-16) continue;
            temp_r /= d;
            vx += temp_r * disp[0];
            vy += temp_r * disp[1];
            vz += temp_r * disp[2];
            ++touched;
          }
    velocity[c * 3 + 0] += vx;
    velocity[c * 3 + 1] += vy;
    if (!is2d) velocity[c * 3 + 2] += vz;
    pressure[c] += press;
    contacts[c] += touched;
  });
}

// Pass 2: cells push and turn fibres.  Parallel over fibres; `head`/`next`
// index cells.  `push` and `rotate` are per-fibre flags, `sticky` per cell.
API void pc_cells_push_fibres(const double* cpos, const double* cradius, const double* crepulsion,
                              const double* cspeed, const double* csticky, long ncells,
                              const double* fpos, const double* fdir, const double* fhalf,
                              const double* frad, const double* frep,
                              const unsigned char* fpush, const unsigned char* frotate,
                              const long* fcrosslinks, long nfibres,
                              double x0, double y0, double z0, double h, long gx, long gy, long gz,
                              const long* head, const long* next, int is2d,
                              double* fvelocity, double* fangle, double* fpressure) {
  const double simple_pressure_scale = 0.027288820670331;
  parallel_for(nfibres, [&](long f) {
    long i, j, k;
    grid_cell(fpos + f * 3, x0, y0, z0, h, gx, gy, gz, i, j, k);
    double vx = 0.0, vy = 0.0, vz = 0.0, angle = 0.0, press = 0.0;
    const bool crosslinked = fcrosslinks[f] != 0;
    for (long kk = (k > 0 ? k - 1 : 0); kk <= (k + 1 < gz ? k + 1 : gz - 1); ++kk)
      for (long jj = (j > 0 ? j - 1 : 0); jj <= (j + 1 < gy ? j + 1 : gy - 1); ++jj)
        for (long ii = (i > 0 ? i - 1 : 0); ii <= (i + 1 < gx ? i + 1 : gx - 1); ++ii)
          for (long c = head[(kk * gy + jj) * gx + ii]; c != -1; c = next[c]) {
            double disp[3];
            const double d = nearest_on_fibre(cpos + c * 3, fpos + f * 3, fdir + f * 3,
                                              fhalf[f], disp);
            const double R = cradius[c] + frad[f];
            if (d > R) continue;
            double temp_r = 1.0 - d / R;
            temp_r *= temp_r;
            press += temp_r / simple_pressure_scale;

            if (fpush[f] && !crosslinked) {
              double push = temp_r * std::sqrt(crepulsion[c] * frep[f]);
              if (std::fabs(push) >= 1e-16) {
                push /= d;
                vx -= push * disp[0];           // the fibre moves away from the cell
                vy -= push * disp[1];
                vz -= push * disp[2];
              }
            }
            if (frotate[f] && !crosslinked) {
              // the impact point is the cell's position minus the displacement
              const double ix = cpos[c * 3 + 0] - disp[0] - fpos[f * 3 + 0];
              const double iy = cpos[c * 3 + 1] - disp[1] - fpos[f * 3 + 1];
              const double arm = std::sqrt(ix * ix + iy * iy);
              const double full = 2.0 * fhalf[f];
              angle += csticky[c] * cspeed[c] * arm / (0.5 * full * full);
            }
          }
    fvelocity[f * 3 + 0] += vx;
    fvelocity[f * 3 + 1] += vy;
    if (!is2d) fvelocity[f * 3 + 2] += vz;
    fangle[f] += angle;
    fpressure[f] += press;
  });
}

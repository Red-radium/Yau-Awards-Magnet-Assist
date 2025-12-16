#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

const double g = 9.80665;
const double pi = 3.141592653589793;
const double mu_0 = 4*pi*1e-7;

struct Vector3 {
    double x, y, z;
    Vector3 operator+(const Vector3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vector3 operator-(const Vector3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vector3 operator*(double s)         const { return {x*s,   y*s,   z*s  }; }
    Vector3 operator/(double s)         const { return {x/s,   y/s,   z/s  }; }
    Vector3& operator+=(const Vector3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
};
double dot(const Vector3& a, const Vector3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
double norm(const Vector3& v) {
    return sqrt(dot(v,v));
}
struct Magnet {
    Vector3 pos;
    Vector3 ma;
};

struct ExperimentParams {
    double line_length;
    Vector3 hanging_point;
    double m;
    Vector3 ma;
    Vector3 v;
    double damping;
    Vector3 pos1;
    Vector3 pos2;
    Vector3 ma1;
    Vector3 ma2;
};
Vector3 dipole_force(const Vector3& ma1, const Vector3& ma2, const Vector3& r1, const Vector3& r2) {
    Vector3 R = r2 - r1;
    double Rmag = norm(R);
    if (Rmag == 0.0) return {0,0,0};
    Vector3 R_hat = R / Rmag;
    double term1 = dot(ma2, R_hat);
    double term2 = dot(ma1, ma2);
    double term3 = dot(ma1, R_hat);
    double term4 = dot(ma2, R_hat);

    Vector3 v1 = ma1 * term1;
    Vector3 v2 = R_hat * term2;
    Vector3 v3 = ma2 * term3;
    Vector3 v4 = R_hat * (5 * term3 * term4);

    return (v1 + v2 + v3 - v4) * (3 * mu_0 / (4*pi) / pow(Rmag, 4));
}

class Experiment {
public:
    explicit Experiment(const ExperimentParams& p) : params(p) {}
    double dt = 0.01; // 这是dt不要改！！！！！！！！
    double line_length;
    Vector3 hanging_point;
    double m;
    Vector3 ma;
    Vector3 pos;
    Vector3 v;
    double damping;
    vector<Magnet> magnets; // must be 2 elements!
private:
    ExperimentParams params;

public:
    void init(double x, double y) {
        line_length = params.line_length;        
        pos = {x, y, 0.0};
        if (x*x + y*y > line_length*line_length) {
            cout << x << " " << y << endl;
            throw runtime_error("line length too short for initial (x,y)");
        }
        double dz = sqrt(line_length*line_length - x*x - y*y);
        hanging_point = params.hanging_point;
        pos.z = hanging_point.z - dz;
        ma = params.ma;
        m = params.m;
        v = params.v;
        damping = params.damping;
        magnets.clear();
        Magnet m1 = {params.pos1, params.ma1};
        magnets.emplace_back(m1);
        Magnet m2 = {params.pos2, params.ma2};
        magnets.emplace_back(m2);
    }

    Vector3 motion_with_tension() {
        Vector3 Fg = {0, 0, -m * g};
        Vector3 Fm = {0,0,0};
        for (auto& mag : magnets)
            Fm += dipole_force(mag.ma, ma, mag.pos, pos);
        // cout << Fm.x << endl;
        Vector3 Fd = v * -damping;
        Vector3 Fext = Fg + Fm + Fd;

        Vector3 radial = pos - hanging_point;
        double rnorm = norm(radial);
        Vector3 radial_hat = radial / rnorm;
        double v2 = dot(v,v);
        double radial_force = dot(Fext, radial_hat);
        double tension = m * v2 / line_length + radial_force;

        Vector3 Ftotal = Fext - radial_hat * tension;
        return Ftotal / m;
    }

    pair<vector<double>, vector<Vector3>>
    simulation(double x=0.0, double y=0.0,
               double max_time=80.0,
               double terminate_acc=0.01,
               double terminate_vel=0.02)
    {
        init(x, y);
        double t = 0.0;
        vector<double> T;
        vector<Vector3> Pos;

        while (t < max_time) {
            Vector3 acc = motion_with_tension();
            v += acc * dt;
            pos += v * dt;

            Vector3 rel = pos - hanging_point;
            Vector3 rhat = rel / norm(rel);
            pos = hanging_point + rhat * line_length;

            double radialComponent = dot(v,rhat);
            v = v - rhat * radialComponent;

            t += dt;
            T.push_back(t);
            Pos.push_back(pos);

            if (norm(acc) <= terminate_acc && norm(v) <= terminate_vel)
                break;
        }
        return {T, Pos};
    }
};

int main() {
    struct ExperimentCase {
        string idx;
        ExperimentParams params;
        double Sx;
        double Sy;
        int resolution;
    };
    vector<ExperimentCase> cases = {
        // {"Length_1", {0.206, {0, 0, 0.246}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Length_2", {0.256, {0, 0, 0.296}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Length_3", {0.306, {0, 0, 0.346}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Length_4", {0.356, {0, 0, 0.396}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Height_1", {0.286, {0, 0, 0.316}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Height_2", {0.286, {0, 0, 0.346}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Height_3", {0.286, {0, 0, 0.376}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Height_4", {0.286, {0, 0, 0.406}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Damp_1", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Damp_2", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0005, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Damp_3", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.001, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Damp_4", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.005, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Damp_5", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.01, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"MagneticMoment_1", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.5}, {0, 0, 0.5}}, 0.12, 0.12, 500},
        // {"MagneticMoment_2", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 1.0}, {0, 0, 1.0}}, 0.12, 0.12, 500},
        // {"MagneticMoment_3", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 1.5}, {0, 0, 1.5}}, 0.12, 0.12, 500},
        // {"MagneticMoment_4", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 2.0}, {0, 0, 2.0}}, 0.12, 0.12, 500},
        // {"Vx_1", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0.1, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vx_2", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0.2, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vx_3", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0.3, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vx_4", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0.4, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vy_1", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0.1, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vy_2", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0.2, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vy_3", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0.3, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Vy_4", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0.4, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Mass_1", {0.286, {0, 0, 0.326}, 0.003, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Mass_2", {0.286, {0, 0, 0.326}, 0.006, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Mass_3", {0.286, {0, 0, 0.326}, 0.009, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Mass_4", {0.286, {0, 0, 0.326}, 0.012, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Separation_1", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.02, 0, 0.001}, {0.02, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Separation_2", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.03, 0, 0.001}, {0.03, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Separation_3", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.04, 0, 0.001}, {0.04, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Separation_4", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0002, {-0.05, 0, 0.001}, {0.05, 0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Experiment_1", {0.263, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0434, -0.001, 0.001}, {0.0400, 0.01, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        // {"Experiment_2", {0.274, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0367, -0.001, 0.001}, {0.0348, 0.002, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500},
        {"Experiment_3", {0.287, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0281, -0.001, 0.001}, {0.0285, 0.02, 0.001}, {0, 0, 0.557}, {0, 0, 0.557}}, 0.12, 0.12, 500},
        {"Experiment_4", {0.287, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0184, -0.0, 0.001}, {0.0186, 0.0, 0.001}, {0, 0, 0.557}, {0, 0, 0.557}}, 0.12, 0.12, 500},
        {"Experiment_5", {0.28,  {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0342, -0.0, 0.001}, {0.0182, 0.0, 0.001}, {0, 0, 0.557}, {0, 0, 0.557}}, 0.12, 0.12, 500},
        // {"Experiment_6", {0.286, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0430, -0.0, 0.001}, {0.0400, 0.0, 0.001}, {0, 0, 0.557}, {0, 0, 0.557}}, 0.12, 0.12, 500},
        // {"Experiment_7", {0.298, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0176, -0.0, 0.001}, {0.0290, 0.0, 0.001}, {0, 0, 0.557}, {0, 0, 0.557}}, 0.12, 0.12, 500},
        // {"Experiment_8", {0.293, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0198, -0.0, 0.001}, {0.0199, 0.0, 0.001}, {0, 0, 0.381}, {0, 0, 0.381}}, 0.12, 0.12, 500},
        // {"Experiment_9", {0.296, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0272, -0.0, 0.001}, {0.0289, 0.0, 0.001}, {0, 0, 0.381}, {0, 0, 0.381}}, 0.12, 0.12, 500},
        // {"Experiment_10", {0.296, {0, 0, 0.326}, 0.00530, {0, 0, 0.762}, {0, 0, 0}, 0.0008, {-0.0369, -0.0, 0.001}, {0.0369, 0.0, 0.001}, {0, 0, 0.762}, {0, 0, 0.762}}, 0.12, 0.12, 500}
    };

    for (const auto& c : cases) {
        const double Sx = c.Sx;
        const double Sy = c.Sy;
        const int resolution = c.resolution;
        // if (c.idx >= 5) {cout << c.idx << " skipped" << endl; continue;}

        string filename = "fractal_basin_" + c.idx + ".txt";

        ofstream f(filename);
        cout << filename << endl;

        f << Sx << " " << Sy << endl;
        double dx = Sx*2/resolution;
        double dy = Sy*2/resolution;

        vector<string> rows(resolution);
        atomic<int> next_row{0};
        atomic<int> rows_done{0};
        atomic<int> last_print{-1};
        unsigned int thread_count = thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 4;
        if (thread_count > static_cast<unsigned int>(resolution)) {
            thread_count = static_cast<unsigned int>(resolution);
        }

        auto worker = [&]() {
            Experiment exp_local(c.params);
            pair<vector<double>, vector<Vector3>> local_result;
            while (true) {
                int row = next_row.fetch_add(1);
                if (row >= resolution) break;
                double y = -Sy + row * dy;
                string line;
                line.reserve(resolution);
                for (int col = 0; col < resolution; ++col) {
                    double x = -Sx + col * dx;
                    local_result = exp_local.simulation(x, y);
                    auto Pos = local_result.second;
                    auto p = Pos.back();
                    double distance1 = norm(p - exp_local.magnets[0].pos);
                    double distance2 = norm(p - exp_local.magnets[1].pos);
                    Vector3 delta_height = {0, 0, exp_local.line_length};
                    double distance_to_origin = norm(p - (exp_local.hanging_point - delta_height));
                    char out;
                    if (distance_to_origin <= 0.005) {
                        out = '1';
                    } else if (distance1 < distance2) {
                        out = '0';
                    } else {
                        out = '2';
                    }
                    line.push_back(out);
                }
                rows[row] = std::move(line);

                int done = rows_done.fetch_add(1) + 1;
                int percent = done * 100 / resolution;
                if (percent % 5 == 0) {
                    int prev = last_print.load();
                    while (percent > prev && !last_print.compare_exchange_weak(prev, percent)) {}
                    if (percent > prev) {
                        cout << "\rProgress: " << percent << "%" << flush;
                    }
                }
            }
        };

        vector<thread> workers;
        workers.reserve(thread_count);
        for (unsigned int i = 0; i < thread_count; ++i) {
            workers.emplace_back(worker);
        }
        for (auto& th : workers) th.join();
        cout << "\rProgress: 100%" << endl;

        for (const auto& line : rows) f << line << endl;
        f.close();
        cout << endl;
    }
    return 0;
}

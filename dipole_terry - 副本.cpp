#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

const double g = 9.80665;
const double pi = 3.141592653589793;
const double mu_0 = 4*pi*1e-7;

struct Vector3 {
    double x, y, z;
    Vector3 operator+(const Vector3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vector3 operator-(const Vector3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vector3 operator*(double s)       const { return {x*s,   y*s,   z*s  }; }
    Vector3 operator/(double s)       const { return {x/s,   y/s,   z/s  }; }
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
    double line_length = 0.4;
    Vector3 hanging_point = {0.0, 0.0, 0.45};
    double m = 0.04;
    Vector3 ma = {0.0, 0.0, 4.0};
    double dt = 0.01;
    Vector3 pos;
    Vector3 v;
    vector<Magnet> magnets; // must be 2 elements!

    void init(double x, double y) {
        if (x*x + y*y > line_length*line_length)
            throw runtime_error("line length too short for initial (x,y)");
        pos = {x, y, 0.0};
        double dz = sqrt(line_length*line_length - x*x - y*y);
        pos.z = hanging_point.z - dz;
        v = {0.0, 0.0, 0.0};
        magnets.clear();
        Magnet m1 = {Vector3{-0.1,0,0}, Vector3{0,0,4}};
        magnets.emplace_back(m1);
        Magnet m2 = {Vector3{0.1,0,0}, Vector3{0,0,4}};
        magnets.emplace_back(m2);
    }

    Vector3 motion_with_tension() {
        Vector3 Fg = {0, 0, -m * g};
        Vector3 Fm = {0,0,0};
        for (auto& mag : magnets)
            Fm += dipole_force(mag.ma, ma, mag.pos, pos);
        // cout << Fm.x << endl;
        Vector3 Fd = v * -0.01;
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
        init(x,y);
        double t = 0.0;
        vector<double> T;
        vector<Vector3> Pos;

        while (t < max_time) {
            Vector3 acc = motion_with_tension();
            v += acc * dt;
            pos += v * dt;

            Vector3 rel = pos - hanging_point;
            rel = rel * (line_length / norm(rel));
            pos = hanging_point + rel;

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
    ofstream f("D:\\Terry\\School\\Grade 11\\Phy\\ffy physics\\Magnetic Assist\\fractal_basin.txt");
    Experiment exp;
    pair<vector<double>, vector<Vector3>> result;
    double resolution = 500;
    double Sx = 0.2; // number in each direction
    double Sy = 0.2;
    f << Sx << " " << Sy << endl;
    double dx = Sx*2/resolution;
    double dy = Sy*2/resolution;
    for (double y = -Sy; y < Sy; y += dy) {
        cout << 1;
        for (double x = -Sx; x < Sx; x += dx) {
            result = exp.simulation(x, y);
            auto T = result.first;
            auto Pos = result.second;
            auto p = Pos.back();
            double distance1 = norm(p - exp.magnets[0].pos);
            double distance2 = norm(p - exp.magnets[1].pos);
            Vector3 delta_height = {0, 0, exp.line_length};
            double distance_to_origin = norm(p - (exp.hanging_point - delta_height));
            double out;
            if (distance_to_origin <= 0.02) {
                out = 1;
            } else if (distance1 < distance2) {
                out = 0;
            } else {
                out = 2;
            }
            f << out;
        }
        f << endl;
    }
    return 0;
}
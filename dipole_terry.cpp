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

// parameter for experiment
Vector3 pos1_ = {-0.04,0,0}; // {-0.1,0,0}
Vector3 pos2_ = {0.04,0,0}; // {0.1,0,0}
Vector3 ma_ = {0,0,4.4}; // {0,0,3}
Vector3 ma1_ = {0,0,4.4}; // {0,0,3}
Vector3 ma2_ = {0,0,0}; // {0,0,3}
double m_ = 0.0075; // 0.02
Vector3 v_ = {0,0,0}; // {0,0,0}
double line_length_ = 0.27; // 0.4
Vector3 hanging_point_ = {0,0,0.315}; // {0,0,0.45}
double damping_ = 0.01; // 0.01

class Experiment {
public:
    double dt = 0.01; // 这是dt不要改！！！！！！！！
    double line_length;
    Vector3 hanging_point;
    double m;
    Vector3 ma;
    Vector3 pos;
    Vector3 v;
    double damping;
    vector<Magnet> magnets; // must be 2 elements!

    void init(double x, double y) {
        line_length = line_length_;        
        pos = {x, y, 0.0};
        if (x*x + y*y > line_length*line_length) {
            cout << x << " " << y << endl;
            throw runtime_error("line length too short for initial (x,y)");
        }
        double dz = sqrt(line_length*line_length - x*x - y*y);
        pos.z = hanging_point.z - dz;
        ma = ma_;
        m = m_;
        v = v_;
        damping = damping_;
        hanging_point = hanging_point_;
        magnets.clear();
        Magnet m1 = {pos1_, ma1_};
        magnets.emplace_back(m1);
        Magnet m2 = {pos2_, ma2_};
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
    cout << "Hello";
    // ofstream f("D:\\Terry\\School\\Grade 11\\Phy\\ffy physics\\Magnetic Assist\\fractal_basin.txt");
    Experiment exp;
    pair<vector<double>, vector<Vector3>> result;
    double resolution = 500;
    double Sx = 0.15; // 0.2
    double Sy = 0.15; // 0.2
    // f << Sx << " " << Sy << endl;
    double dx = Sx*2/resolution;
    double dy = Sy*2/resolution;
    for (double y = -Sy; y < Sy; y += dy) {
        cout << "|";
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
            // f << out;
        }
        // f << endl;
    }
    // f.close();
    return 0;
}
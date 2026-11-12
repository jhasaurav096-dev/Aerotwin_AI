#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>

struct EngineTelemetry {
    int cycle;
    double egt_celsius;
    double n2_rpm_pct;
    double fuel_flow_pph;
    double vibration_ips;
    double oil_pressure_psi;
};

class DigitalTwinStateEstimator {
private:
    double alpha;
    double filtered_egt;
    double filtered_vib;
    bool initialized;

public:
    DigitalTwinStateEstimator(double smoothing_factor = 0.25)
        : alpha(smoothing_factor), filtered_egt(0.0), filtered_vib(0.0), initialized(false) {}

    EngineTelemetry parseFrame(const std::string& raw_csv) {
        std::stringstream ss(raw_csv);
        std::string item;
        std::vector<double> vals;
        while (std::getline(ss, item, ',')) {
            vals.push_back(std::stod(item));
        }
        if (vals.size() < 6) {
            throw std::runtime_error("Malformed telemetry frame");
        }
        return EngineTelemetry{
            static_cast<int>(vals[0]),
            vals[1], vals[2], vals[3], vals[4], vals[5]
        };
    }

    double computePhysicsExpectedEGT(double n2_pct, double fuel_flow) const {
        return 310.0 + (4.15 * n2_pct) + (0.048 * fuel_flow);
    }

    void processFrame(const EngineTelemetry& raw) {
        auto start = std::chrono::high_resolution_clock::now();

        if (!initialized) {
            filtered_egt = raw.egt_celsius;
            filtered_vib = raw.vibration_ips;
            initialized = true;
        } else {
            filtered_egt = alpha * raw.egt_celsius + (1.0 - alpha) * filtered_egt;
            filtered_vib = alpha * raw.vibration_ips + (1.0 - alpha) * filtered_vib;
        }

        double expected_egt = computePhysicsExpectedEGT(raw.n2_rpm_pct, raw.fuel_flow_pph);
        double egt_residual = filtered_egt - expected_egt;

        double penalty = std::max(0.0, egt_residual * 0.65) + std::max(0.0, (filtered_vib - 0.8) * 18.0);
        double health_index = std::max(0.0, std::min(100.0, 100.0 - penalty));

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();

        std::string status = (health_index > 80.0) ? "NOMINAL" : (health_index > 55.0 ? "ADVISORY" : "CRITICAL_SURGE");

        std::cout << std::fixed << std::setprecision(2)
                  << "[Cycle " << std::setw(3) << raw.cycle << "] "
                  << "EGT: " << filtered_egt << "C (Expected: " << expected_egt << "C, Residual: +" << egt_residual << "C) | "
                  << "Health: " << health_index << "% | "
                  << "Status: " << status << " | "
                  << "Latency: " << latency_us << " us\n";
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  AeroTwin AI - Low-Latency C++17 Engine Telemetry Processor (macOS)\n";
    std::cout << "====================================================================\n";

    std::vector<std::string> sample_frames = {
        "10,645.2,91.5,2450.0,0.42,62.0",
        "50,652.8,92.0,2480.0,0.51,61.2",
        "120,678.4,92.1,2510.0,0.88,58.5",
        "185,715.9,92.5,2560.0,1.45,54.0",
        "210,758.3,93.0,2610.0,2.15,48.2"
    };

    DigitalTwinStateEstimator estimator(0.35);
    for (const auto& frame : sample_frames) {
        auto parsed = estimator.parseFrame(frame);
        estimator.processFrame(parsed);
    }

    return 0;
}

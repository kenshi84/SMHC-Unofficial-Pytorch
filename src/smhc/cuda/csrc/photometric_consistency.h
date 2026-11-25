#ifndef SMHC_PHOTOMETRIC_CONSISTENCY_H
#define SMHC_PHOTOMETRIC_CONSISTENCY_H

#include <cmath>

#ifdef __CUDACC__
#define SMHC_PHOTOMETRIC_HD __host__ __device__
#else
#define SMHC_PHOTOMETRIC_HD
#endif

namespace solution::lpmvs
{

struct ChannelCorrelation
{
    int count = 0;
    float mean_reference = 0.f;
    float mean_neighbor = 0.f;
    float reference_energy = 0.f;
    float neighbor_energy = 0.f;
    float cross_energy = 0.f;
    bool finite = true;

    SMHC_PHOTOMETRIC_HD void add(float reference, float neighbor, bool use_zncc)
    {
        if (!std::isfinite(reference) || !std::isfinite(neighbor))
        {
            finite = false;
            return;
        }
        ++count;
        if (use_zncc)
        {
            // Welford updates avoid subtracting nearly equal raw moments in
            // bright, low-contrast patches. Count only accepted sample pairs.
            const float delta_reference = reference - mean_reference;
            const float delta_neighbor = neighbor - mean_neighbor;
            mean_reference += delta_reference / count;
            mean_neighbor += delta_neighbor / count;
            reference_energy += delta_reference * (reference - mean_reference);
            neighbor_energy += delta_neighbor * (neighbor - mean_neighbor);
            cross_energy += delta_reference * (neighbor - mean_neighbor);
        }
        else
        {
            cross_energy += reference * neighbor;
            reference_energy += reference * reference;
            neighbor_energy += neighbor * neighbor;
        }
    }

    SMHC_PHOTOMETRIC_HD bool correlation(
        bool use_zncc, float minimum_variance, float& result) const
    {
        result = 0.f;
        if (!finite || !std::isfinite(reference_energy)
            || !std::isfinite(neighbor_energy) || !std::isfinite(cross_energy))
            return false;

        if (use_zncc)
        {
            // A constant patch carries no evidence about a correspondence.
            if (count < 2 || reference_energy <= minimum_variance * count
                || neighbor_energy <= minimum_variance * count)
                return false;
            const float denominator = std::sqrt(reference_energy) * std::sqrt(neighbor_energy);
            result = cross_energy / denominator;
        }
        else
        {
            // Preserve the previous uncentered cosine score for comparisons.
            result = cross_energy / (std::sqrt(reference_energy * neighbor_energy) + 1e-8f);
        }

        if (!std::isfinite(result))
            return false;
        result = result < -1.f ? -1.f : (result > 1.f ? 1.f : result);
        return true;
    }
};

// RGB samples are centered per channel. Ignore alpha and channels without
// sufficient variance rather than diluting the evidence from textured ones.
template<typename T>
struct PhotometricCorrelation
{
    ChannelCorrelation channels[3];

    SMHC_PHOTOMETRIC_HD void add(const T& reference, const T& neighbor, bool use_zncc)
    {
        channels[0].add(reference.x, neighbor.x, use_zncc);
        channels[1].add(reference.y, neighbor.y, use_zncc);
        channels[2].add(reference.z, neighbor.z, use_zncc);
    }

    SMHC_PHOTOMETRIC_HD bool correlation(
        bool use_zncc, float minimum_variance, float& result) const
    {
        result = 0.f;
        int valid_channels = 0;
        for (const auto& channel : channels)
        {
            float value;
            if (channel.correlation(use_zncc, minimum_variance, value))
            {
                result += value;
                ++valid_channels;
            }
        }
        if (valid_channels == 0)
            return false;
        result /= valid_channels;
        return true;
    }
};

template<>
struct PhotometricCorrelation<float> : ChannelCorrelation {};

}

#undef SMHC_PHOTOMETRIC_HD
#endif

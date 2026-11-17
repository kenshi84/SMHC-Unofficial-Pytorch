#ifndef GROWING_DIRECTION_H_INCLUDED
#define GROWING_DIRECTION_H_INCLUDED

#include <limits>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace solution::multiview_hair_growing
{

// Solve the multi-view plane constraints, then reweight once to suppress outliers.
// A zero vector indicates invalid or insufficiently independent constraints.
inline Eigen::Vector3d solve_growing_direction(Eigen::MatrixX3d A)
{
    if (A.rows() < 2 || !A.allFinite())
        return Eigen::Vector3d::Zero();

    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        // Common scaling preserves the objective and makes the residual floor
        // independent of the coordinate units used by the camera geometry.
        const double scale = A.cwiseAbs().maxCoeff();
        if (!(scale > 0.0))
            return Eigen::Vector3d::Zero();
        A /= scale;

        const Eigen::Matrix3d normal_matrix = A.transpose() * A;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(normal_matrix);
        if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
            return Eigen::Vector3d::Zero();

        // Rank two is the ideal case: its nullspace is the growing direction.
        // Reject an ambiguous minimum (including rank zero or one), not a zero
        // smallest eigenvalue.
        const auto& eigenvalues = solver.eigenvalues();
        const double gap_tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * eigenvalues(2);
        if (eigenvalues(2) <= 0.0 || eigenvalues(1) - eigenvalues(0) <= gap_tolerance)
            return Eigen::Vector3d::Zero();

        direction = solver.eigenvectors().col(0);
        if (!direction.allFinite())
            return Eigen::Vector3d::Zero();

        if (iteration == 0)
        {
            // Cap inverse-squared residual weights at 1e6. Exact agreements
            // remain usable without infinities or overwhelming all other views.
            constexpr double minimum_squared_residual = 1e-6;
            Eigen::VectorXd weights = (A * direction).array().square()
                .max(minimum_squared_residual).inverse();
            weights /= weights.maxCoeff();
            A.array().colwise() *= weights.array();
        }
    }

    return direction;
}

}

#endif

#ifndef TENSIONFIELDSTVKMATERIAL_H
#define TENSIONFIELDSTVKMATERIAL_H

#include "MaterialModel.h"

/*
 * Tension field theory implementation of a St. Venant-Kirchhoff material with 
 * energy density (in the pure-tension case)
 * W = alpha/2.0 tr(S)^2 + beta tr(S^2),
 * for strain tensor S = gbar^{-1}(g-gbar), where g and gbar are the current 
 * and rest metrics of the shell volume (which vary in the thickness direction
 * as defined by the surface fundamental forms).
 *
 * This material has no bending energy.
 */

template <class SFF>
class TensionFieldStVKMaterial : public MaterialModel<SFF>
{
public:
    TensionFieldStVKMaterial(double lameAlpha, double lameBeta) : lameAlpha_(lameAlpha), lameBeta_(lameBeta) {}

    /*
     * Lame parameters of the material (as in the energy density written above)
     */
    double lameAlpha_, lameBeta_;

    virtual double stretchingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        double thickness,
        const Eigen::Matrix2d &abar,
        int face,
        Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
        Eigen::Matrix<double, 9, 9> *hessian) const;

    virtual double bendingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        const Eigen::VectorXd &extraDOFs,
        double thickness,
        const Eigen::Matrix2d &abar, const Eigen::Matrix2d &bbar,
        int face,
        Eigen::Matrix<double, 1, 18 + 3*SFF::numExtraDOFs> *derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
        Eigen::Matrix<double, 18 + 3*SFF::numExtraDOFs, 18 + 3*SFF::numExtraDOFs> *hessian) const;
};


#endif
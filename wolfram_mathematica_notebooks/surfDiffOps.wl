(* ::Package:: *)

BeginPackage["SurfDiffOps`"];


gammaGradScalar::usage = "gammaGradScalar[P, q, xyz] computes Gamma-gradient of a scalar field q.";
gammaCovariantDerivative::usage = "gammaCovariantDerivative[P, v, xyz] computes covariant derivative of a vector field v.";
gammaGradVector::usage = "gammaGradVector[P, v, xyz] computes Gamma-gradient of a vector field v.";
gammaStressTensor::usage = "gammaStressTensor[P, v, xyz] computes Cauchy stress tensor a vector field v.";
gammaDivergenceVector::usage = "gammaDivergenceVector[P, v, xyz] computes Gamma-divergence of a vector field v";
gammaLapl::usage = "gammaLapl[P, q, xyz] computes Gamma-Laplacian of a scalar field q.";
gammaDivergenceMatrix::usage = "gammaDivergenceMatrix[P, A, xyz] computes Gamma-Laplacian of a vector field v.";
gammaCurlScalar::usage = "gammaCurlScalar[n, P, psi, xyz] computes Gamma-curl of a scalar field psi.";


Begin["`Private`"];


gammaGradScalar[Q_,q_,xyz_] :=
    Dot[Q, Grad[q, xyz]];

gammaCovariantDerivative[Q_,v_,xyz_] :=
    Dot[Grad[v, xyz],Q];

gammaGradVector[Q_,v_,xyz_] :=
    Dot[Dot[Q, Grad[v, xyz]], Q];

gammaStressTensor[Q_,v_,xyz_] :=
    1 / 2 * Dot[Dot[Q, Grad[v, xyz] + Transpose[Grad[v, xyz]]], Q];

gammaDivergenceVector[Q_,v_,xyz_] :=
    Tr[gammaCovariantDerivative[Q,v,xyz]];

gammaLapl[Q_,q_,xyz_] :=
    gammaDivergenceVector[Q,gammaGradScalar[Q,q,xyz],xyz];

gammaDivergenceMatrix[Q_,A_,xyz_] :=
    Transpose[{
    gammaDivergenceVector[Q, Dot[Transpose[UnitVector[3, 1]], A],xyz], 
    gammaDivergenceVector[Q, Dot[Transpose[UnitVector[3, 2]], A],xyz], 
    gammaDivergenceVector[Q, Dot[Transpose[UnitVector[3, 3]], A],xyz]}];

gammaCurlScalar[n_,Q_,psi_,xyz_] :=
    Cross[n, gammaGradScalar[Q, psi,xyz]];


End[];


EndPackage[];

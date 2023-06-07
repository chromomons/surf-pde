(* ::Package:: *)

BeginPackage["ExactSolution`"];


sinSolPoi::usage = "Args: x. Non-polynomial function depending only on space variables.";
sinSolDiff::usage = "Args: t, x. Non-polynomial function depending on both space and time variables.";
evolvingSurfDiffLehrenfeldEtAl::usage = "Args:t, x, w. Exact solution to homogeneous evolving-surface diffusion problem where the domain is translating sphere from Lehrenfeld, Olshanskii, Xu, 2018.";

BrandnerEtAlSteadyStokesVel::usage = "Args: n, P, x. Gamma-divergence-free solution for the fixed-surface Stokes problem from Brandner et al, 2022. Velocity.";
BrandnerEtAlSteadyStokesPres::usage = "Args: x. Gamma-divergence-free solution for the fixed-surface Stokes problem from Brandner et al, 2022. Pressure.";
BrandnerEtAlUnsteadyStokesVel::usage = "Args: n, P, t, x. Same as BrandnerEtAlSteadyStokesVel, but each component is multiplied by a time dependent function g(t)=1+sin(pi*t). Velocity.";
BrandnerEtAlUnsteadyStokesPres::usage = "Args: t, x. Same as BrandnerEtAlSteadyStokesPres, but multiplied by a time dependent function g(t)=1+sin(pi*t). Pressure.";

OlshanskiiEtAlSteadyStokesVel::usage = "Args: P, x. Non-solenoidal solution for the fixed-surface Stokes problem from Olshanskii et al, 2018. Velocity.";
OlshanskiiEtAlSteadyStokesPres::usage = "Args: x. Non-solenoidal solution for the fixed-surface Stokes problem from Olshanskii et al, 2018. Pressure.";
OlshanskiiEtAlUnsteadyStokesVel::usage = "Args: P, t, x. Same as OlshanskiiEtAlSteadyStokesVel, but each component is multiplied by a time dependent function g(t)=1+sin(pi*t). Velocity.";
OlshanskiiEtAlUnsteadyStokesPres::usage = "Args: t, x. Same as OlshanskiiEtAlSteadyStokesPres, but multiplied by a time dependent function g(t)=1+sin(pi*t). Pressure.";

OlshanskiiEtAlEvolvingSurfNSVel::usage = "Args: n, P, t, x. Gamma-divergence-free solution for the evolving-surface NS from Olshanskii et al, 2023. Velocity.";
OlshanskiiEtAlEvolvingSurfNSPres::usage = "Args: t, x, w. Gamma-divergence-free solution for the evolving-surface NS from Olshanskii et al, 2023. Pressure.";


Begin["`Private`"];


<<"surfDiffOps.wl";
sinSolPoi[x_]:=Sin[Pi*x[[1]]]*Sin[Pi*x[[2]]*x[[3]]];
sinSolDiff[t_,x_]:=(1+Sin[Pi*t])*Sin[Pi*x[[1]]]*Sin[Pi*x[[2]]*x[[3]]];
evolvingSurfDiffLehrenfeldEtAl[t_,x_,w_]:=1+(x[[1]]+x[[2]]+x[[3]]-w[[1]]*t)*Exp[-2*t];

BrandnerEtAlSteadyStokesVel[n_,P_,x_]:=FullSimplify@gammaCurlScalar[n,P,x[[1]]^2*x[[2]]-5*x[[3]]^3,x];
BrandnerEtAlSteadyStokesPres[x_]:=x[[1]]^3+x[[1]]*x[[2]]*x[[3]];
BrandnerEtAlUnsteadyStokesVel[n_,P_,t_,x_]:=(1+Sin[Pi*t])*FullSimplify@gammaCurlScalar[n,P,x[[1]]^2*x[[2]]-5*x[[3]]^3,x];
BrandnerEtAlUnsteadyStokesPres[t_,x_]:=(1+Sin[Pi*t])*(x[[1]]^3+x[[1]]*x[[2]]*x[[3]]);

OlshanskiiEtAlSteadyStokesVel[P_,x_]:=Dot[P,{-x[[3]]^2,x[[2]],x[[1]]}];
OlshanskiiEtAlSteadyStokesPres[x_]:=x[[1]]*x[[2]]^3+x[[3]];
OlshanskiiEtAlUnsteadyStokesVel[P_,t_,x_]:=(1+Sin[Pi*t])*Dot[P,{-x[[3]]^2,x[[2]],x[[1]]}];
OlshanskiiEtAlUnsteadyStokesPres[t_,x_]:=(1+Sin[Pi*t])*(x[[1]]*x[[2]]^3+x[[3]]);

OlshanskiiEtAlEvolvingSurfNSVel[n_,P_,t_,x_]:=FullSimplify@gammaCurlScalar[n,P,x[[1]]*x[[2]]-2*t,x];
OlshanskiiEtAlEvolvingSurfNSPres[t_,x_,w_]:=(x[[1]]-w[[1]]*t)*x[[2]]*x[[3]];


End[];


EndPackage[];

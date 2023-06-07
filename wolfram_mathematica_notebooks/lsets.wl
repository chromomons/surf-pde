(* ::Package:: *)

BeginPackage["Levelsets`"];


phiSphere::usage = "phiSphere[R, x] returns levelset of a sphere of radius R, correctly scaled.";
phiTorus::usage = "phiTorus[r, R, x] returns levelset of a torus with radii r, R, for computation.";
phiCheese::usage = "phiCheese[x] returns levelset of a cheese surface of genus 6, for computation.";
phiBiconc::usage = "phiBiconc[d,c,x] returns levelset of a biconcave shape with parameters d,c, for computation.";
phiDziuk::usage = "phiDziuk[x] returns levelset of a Dziuk surface, for computation.";

phiTranslSphere::usage = "phiTranslSphere[R, t, x, w] returns levelset of a sphere advected by ambient velocity field w of radius R, correctly scaled.";
phiL4intoL2sphere::usage = "phiL4intoL2sphere[R, t, x] returns levelset of a unit L^4 sphere morphing into a regular unit sphere both of radius R, correctly scaled.";

phiSphereComp::usage = "phiSphereComp[R, x] returns levelset of a sphere of radius R, for computation.";
phiTranslSphereComp::usage = "phiTranslSphereComp[R, t, x, w] returns levelset of a sphere advected by ambient velocity field w of radius R, for computation.";
phiL4intoL2sphereComp::usage = "phiL4intoL2sphereComp[R, t, x] returns levelset of a unit L^4 sphere morphing into a regular unit sphere both of radius R, for computation.";


Begin["`Private`"];


phiSphere[R_,x_]:=Sqrt[x[[1]]^2 + x[[2]]^2 + x[[3]]^2] - R;
phiTorus[r_,R_,x_]:=(Sqrt[x[[1]]^2 + x[[2]]^2] - R)^2 + x[[3]]^2 - r^2;
phiCheese[x_]:=2*x[[2]]*(x[[2]]^2-3x[[1]]^2)*(1-x[[3]]^2)+(x[[1]]^2+x[[2]]^2)^2-(9*x[[3]]^2-1)*(1-x[[3]]^2);
phiBiconc[d_,c_,x_]:=(d^2+x[[1]]^2+x[[2]]^2+x[[3]]^2)^3-8d^2*(x[[2]]^2+x[[3]]^2)-c^4;
phiDziuk[x_]:=1/4*x[[1]]^2+x[[2]]^2+(4*x[[3]]^2)/((1+1/2*Sin[Pi*x[[1]]])^2)-1/2;
phiTranslSphere[R_,t_,x_,w_]:=Sqrt[(x[[1]]-w[[1]]*t)^2 + (x[[2]]-w[[2]]*t)^2 + (x[[3]]-w[[3]]*t)^2] - R;
phiL4intoL2sphere[R_,t_,x_]:=Sqrt[Sqrt[x[[1]]^4+x[[2]]^4+x[[3]]^4+2*t*(x[[1]]^2*x[[2]]^2+x[[2]]^2*x[[3]]^2+x[[1]]^2*x[[3]]^2)]]-R;

phiSphereComp[R_,x_]:=x[[1]]^2 + x[[2]]^2 + x[[3]]^2 - R^2;
phiTranslSphereComp[R_,t_,x_,w_]:=(x[[1]]-w[[1]]*t)^2 + (x[[2]]-w[[2]]*t)^2 + (x[[3]]-w[[3]]*t)^2 - R^2;
phiL4intoL2sphereComp[R_,t_,x_]:=x[[1]]^4+x[[2]]^4+x[[3]]^4+2*t*(x[[1]]^2*x[[2]]^2+x[[2]]^2*x[[3]]^2+x[[1]]^2*x[[3]]^2)-R^4;


End[];


EndPackage[];

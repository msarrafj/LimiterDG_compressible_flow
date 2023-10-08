<center> <h1> Codes for a bound-preserving discontinuous Galerkin solver for compressible two-phase flow problems in porous media </h1> </center>

> Mohammad. S. Joshaghani and Beatrice Riviere,
> ``Bound-preserving discontinuous Galerkin methods for compressible two-phase flows in porous media" Available in [arXiv](https://arxiv.org/abs/2309.01908).
> <details><summary>[Abstract]</summary>
><p> This paper presents a numerical study of immiscible, compressible two-phase flows in porous media, that takes into account heterogeneity, gravity, anisotropy, and injection/production wells. We formulate a fully implicit stable discontinuous Galerkin solver for this system that is accurate, that respects the maximum principle for the approximation of saturation, and that is locally mass conservative. To completely eliminate the overshoot and undershoot phenomena, we construct a flux limiter that produces bound-preserving elementwise average of the saturation. The addition of a slope limiter allows to recover a pointwise bound-preserving discrete saturation. Numerical results show that both maximum principle and monotonicity of the solution are satisfied. The proposed flux limiter does not impact the local mass error and the number of nonlinear solver iterations.
></p>
></details>

Here you can find python codes for the numerical solution of immiscible compressible two-phase flows in porous media. The proposed framework has several interesting features (e.g. high accuracy, robustness, mesh-independence, local mass conservation, and satisfies the maximum principle).
This repo entails several examples of pressure-driven flow and quarter-five spot problems that account for the effect of gravity, anisotropy, heterogeneity, and compressibility factor on the flow. More details are discussed in the paper.





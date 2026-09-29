"""FCX -- Formula-Coordinate Explanations.

Explains TRACE-PD's models in the coordinates of the published Stebbins
formula: the tremor/gait log-ratio  l = log T - log P  and its two cutoffs.

  shapley      exact / Monte-Carlo interventional Shapley (own implementation)
  formula      the Stebbins formula in log-ratio coordinates
  implied_exam C1: explain the subtype classifier as an implied tremor & gait score
  straddle     C2: explain a conformal set as an implied-ratio interval straddling a cutoff
  pdn          C3: explain transition risk as proximity + drift + noise
"""

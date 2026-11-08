# SCET finite-model learning pilot

Decision: **NO_GO**

Learned stopping-time consistency on a finite partially observed competing-risk environment. Mechanism pilot, not geometric navigation or Falcon benchmark.

| Training contexts | Arm | Expected cost | Collision probability | Kernel KL | Train seconds |
|---:|---|---:|---:|---:|---:|
| 128 | step | 13.94598 | 0.04978 | 0.24914 | 0.08 |
| 128 | event | 12.01108 | 0.02067 | 0.42704 | 0.07 |
| 128 | scet | 12.01358 | 0.02075 | 0.42683 | 0.16 |
| 128 | event_long | 12.98969 | 0.04675 | 0.89782 | 0.19 |
| 512 | step | 11.94772 | 0.02133 | 0.20783 | 0.08 |
| 512 | event | 11.65068 | 0.01520 | 0.09804 | 0.07 |
| 512 | scet | 11.65062 | 0.01520 | 0.09799 | 0.15 |
| 512 | event_long | 11.82026 | 0.02003 | 0.13358 | 0.19 |

All methods have 1,313 parameters and receive the same observed history. The step model receives explicit age, so it does not reset the waiting-time distribution.
SCET uses extra consistency passes; event_long receives 3x supervised updates. Recorded training time, not update count alone, determines actual compute comparisons.
Expected costs are evaluated exactly under the simulator distribution, including a closed-loop decision at every survived prefix. These are not robot trajectories or benchmark success rates.

Paired context bootstrap after averaging three training seeds; conditional on these fitted models, not a full training-population confidence interval.

## SCET contrasts at 512 contexts

- Versus step: cost reduction 2.487%; cost delta -0.29710, 99% paired-context interval [-0.3523085666917259, -0.24321529811076562]; collision delta -0.00613; gate False.
- Versus event: cost reduction 0.000%; cost delta -0.00006, 99% paired-context interval [-0.00041519706398937767, 0.00036362065265085983]; collision delta -0.00000; gate False.
- Versus event_long: cost reduction 1.435%; cost delta -0.16964, 99% paired-context interval [-0.2013201120139494, -0.13888906665695142]; collision delta -0.00483; gate False.

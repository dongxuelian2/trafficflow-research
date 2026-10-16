# Directional residual GP: assumptions and falsifiable claims

For a trusted directed mainline chain define x in km by summing released link
lengths in outgoing topology order. Require reciprocal incoming/outgoing edges,
matching node endpoints, no internal branching/cycle, positive lengths, and
monotone public mileage with plausible distance ratios. Latent links remain in
the chain. Disconnected or untrusted cells get zero correction. Mileage is only
a validation measurement and never supplies missing topology edges.

For c in km/hour, let phi_c(x,t)=((x-c t)/ell_x,t/ell_t). The stated propagation
kernel is the unit-amplitude Gaussian RBF under phi_c. Gaussian RBF is PSD by
its nonnegative Fourier transform; pulling it back through any deterministic
map preserves PSD. For lawful congestion probability g(z) in [0,1], define

K(z,z') = sqrt(1-g(z)) sqrt(1-g(z')) K_80(z,z')
          + sqrt(g(z)) sqrt(g(z')) K_-15(z,z').

Each term is D K D for a real diagonal D, so v^T D K D v >= 0. Their sum is PSD.
The square-root gates keep the diagonal at one. Multiplying by a Gaussian RBF
of lawful predicted-state/age features and the component delta kernel preserves
PSD by the Schur product theorem. The component kernel is block diagonal with
all-ones trusted blocks and zeros for untrusted cells, hence PSD. These facts do
not require gates to be constant or independent of lawful observations. Gates
using target truth or assessment residuals would invalidate the experimental
contract even though the resulting numerical matrix could remain PSD.

The implemented GP uses at most 192 inducing points selected from at most 512
same-day lawful visible anchors, without inspecting residual values during
selection. Every sampled anchor has its own speed, flow, occupancy and density
channels withheld before prediction by the frozen strong model, using two
seeded groups. Original target cells stay missing in both groups. Only the
withheld anchor predictions are used to center observed residuals; target
predictions retain the original frozen caches. With C=chol(K_mm+1e-7 I), features are
Phi=K_nm C^-T and the finite-rank kernel is Phi Phi^T, also PSD. For observation
noise variance lambda=.10, posterior mean is
Phi_* (I+Phi^T Phi/lambda)^-1 Phi^T r/lambda. Cholesky solves replace inverses.
The posterior variance is the finite-rank latent variance; discarded GP modes,
base-model uncertainty and interval calibration are not included. No full
N-by-N matrix is formed. At most a 512-by-192 cross kernel is evaluated per
prediction/training batch.

The primary experiment fits a separate same-day posterior. Time t is hours
since that day's start; anchors from different dates are not pooled. Gaussian
output amplitude per channel is estimated from forward-held-out correction-fit
target residual RMS, clipped to [.05,1], and fixed before calibration. The
counterfactual anchor context adds missingness; it does not guarantee unbiased
centering or exchangeability with long R3 gaps. Calibration selects correction
strength separately for ordinary c=0, gated c=0, and directional kernels under
the same samples, states, inducing budget, noise and length scales. The gated
c=0 comparison distinguishes propagation from gating. The graph control uses
the existing state_assimilation.assimilate_residuals implementation with the
same sampled counterfactual anchor residuals and trusted edges, fixing all
three precisions at one. Historical A58/A59 exact settings are not represented
as rerun evidence. The initial cross-date hour-of-day bias migration proposal
was removed before any real GP experiment; it would not directly test same-day
shock propagation.

Base fit and checkpoint selection precede correction-fit dates; this is one
chronological forward cross-fit, not multiple random out-of-fold replicas.
Calibration precedes assessment and strengths are saved with SHA-256 before
assessment cache arrays or raw labels are read. All fixed full-strength kernels
are also assessed to retain unfavorable results. State uses official
eligibility, pooled regime RMSE and family macro; only two directions are in
scope. Inventory-increment error uses public official topology/ramp pair
guards. It does not assert equality with organizer-held boundary flux b*.

The final executable scorer rejects multiple station/link cells, nonfinite
eligible anchor values, missing published endpoints, and date-varying eligible
full-link sets within a regime. Under those checked invariants, every finite
candidate shares the exact same official pair universe: changing only speed
and flow cannot change topology membership, start-ramp validity or dt guards.
Official guards run once per date; indexed inventory differences are then
algebraically identical to rebuilding the full anchored frame. Every candidate
on the first calibration and assessment date of each direction is checked
against the original independent scorer at rtol=1e-12, atol=1e-8. Selected
calendar dates are joined for <=5.1-minute midnight pairs within the same
regime, using start-time ramp validity and the complete selected-date link-set
guard. Midnight pairs are assigned to their start date for reporting. Epoch
seconds are derived only after explicit conversion to nanoseconds, avoiding
the historical microsecond timestamp failure.

The directional hypothesis is falsified for expansion unless its independent
gains over ordinary and gated ordinary controls are stable across I5_N/I210_W
and seasons. State promotion additionally requires >=.002 gain over the strong
base, positive paired bootstrap lower bound, and no direction below -.001.

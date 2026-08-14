# Mathematics of GT models

## Introduction

Suppose our network has n neurons N. We want to run a simulation of the
membrane potential v(t) of each neuron N over time t. Naturally, the
system evolves so that: v(t+1) = v(t) + \left.\frac{\partial v}{\partial
t}\right\|\_{t+1} The hard part, of course, is to compute \partial
v/\partial t. That is, how do membrane potentials evolve over time?

## Power minimization

To tackle this question, [Gangopadhyay and
Chakrabartty](https://doi.org/10.3389/fnins.2020.00425) applied the
[Baum-Eagon inequality](https://doi.org/10.1090/S0002-9904-1967-11751-8)
to the net metabolic power \mathcal{H} of v to derive \partial
v/\partial t as a growth-transform of v. That is, they assumed that
\partial v/\partial t is always such that: \mathcal{H}(v)\leq
\mathcal{H}(v + \partial v/\partial t) This amounts to assuming that
\mathcal{H} is minimized over time: \frac{\partial\mathcal{H}}{\partial
v}\frac{\partial v}{\partial t} \leq 0 Here, we will work directly from
this minimization assumption and use a biologically inspired
interpretation of the variables to derive \partial v/\partial t. For
simplicity, let us assume that \partial\mathcal{H}/\partial t = 0, i.e.,
that the metabolic power is not changing over time.

Also, let us assume that v represents only the *subthreshold* membrane
potential and thus that \partial v/\partial t only *directly* models
subthreshold membrane potential dynamics.[^1] This leaves out any
mechanistic modeling of spikes themselves. Instead, spikes are modeled
as a constant and instantaneous value v\_\mathrm{spike} (variable
spike_potential in the simulation) added to the subthreshold dynamics
when the spike threshold is crossed. We will explain below how spikes
are nevertheless still critical to the evolution of the system.

Where do we begin? It’s plausible that the membrane potential v is a
scalar multiple fv\_\mathrm{rest} of the resting potential
v\_\mathrm{rest},[^2] where the scale factor f is a function of
\partial\mathcal{H}/\partial v. In other words, we assume that the scale
factor f depends on how a small change \partial v in membrane potential
would change net metabolic power \mathcal{H}. Hence, we have that:[^3]
v(t+1) = v(t) + \left.\frac{\partial v}{\partial t}\right\|\_{t+1} =
f\left(\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_{t+1}\right)v\_\mathrm{rest}

## Scale factor

Therefore, if we could find a suitable scale factor f, we’d be able to
compute \partial v/\partial t as: \left.\frac{\partial v}{\partial
t}\right\|\_t = f\left(\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_{t}\right)v\_\mathrm{rest} - v(t-1) Combining the previous
equation with the minimization assumption, we have:
\frac{\partial\mathcal{H}}{\partial v}
\left(f\left(\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_t\right)v\_\mathrm{rest} - v(t-1) \right) = 0 Assuming
\partial\mathcal{H}/\partial v \neq 0, this equation implies that:
v(t-1) = f\left(\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_t\right)v\_\mathrm{rest} And so:
f\left(\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t\right) =
\frac{v(t-1)}{v\_\mathrm{rest}} Hence, we see that f is a function of
the metabolic power gradient \partial\mathcal{H}/\partial v that’s equal
to the ratio of v over the rest potential v\_\mathrm{rest}. An immediate
consequence is that, when v\approx v\_\mathrm{rest}, the scale factor f
is approximately 1. Hence, we are assuming that v\_\mathrm{rest} is an
equilibrium point (as defined by dynamical systems theory) of the
membrane potential.

Notice that this derivation implies that \partial v/\partial t=0, which
may seem obviously wrong. However, we are attempting to derive a formula
for \partial v/\partial t which makes \partial\mathcal{H}/\partial t=0.
The “makes” is aspirational: of course, \partial v/\partial t\neq 0,
but, what general form of \partial v/\partial t will tend to make
\partial\mathcal{H}/\partial t=0 over time? In terms of biology, we can
alternatively think of the system as *trying* to maintain a stable rest
voltage v\_\mathrm{rest}, and the question is: what form of \partial
v/\partial t will best achieve this goal?

## Rest vs spike power

In order to solve for f, it’s helpful to think about the metabolic power
gradient \partial\mathcal{H}/\partial v and how it relates to two
quantities: the metabolic power \mathcal{H}\_\mathrm{rest} of
maintaining rest potential v\_\mathrm{rest} and the metabolic power
\mathcal{H}\_\mathrm{spike} of initiating a spike under the conditions
which hold at time t. As a linear approximation, we have that:
\mathcal{H}\_\mathrm{rest} =
v\_\mathrm{rest}\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t
This equation is justified because the metabolic power
\mathcal{H}\_\mathrm{rest} of maintaining rest potential
v\_\mathrm{rest} is equal to the power used to maintain the negative
potential v\_\mathrm{rest} under the natural positive flow of charge
into the cell. Unit analysis (\mathrm{Watts} = \mathrm{Volts} \times
\mathrm{Ampere}) implies that this latter quantity must be the power
gradient \partial\mathcal{H}/\partial v evaluated at t.

What about the metabolic power \mathcal{H}\_\mathrm{spike} of initiating
a spike? If v\_\mathrm{threshold} is the membrane potential at spike
threshold, then (by the above unit analysis) the change \partial
I\_\mathrm{influx} in current I across the membrane at spike initiation
is: \partial I\_\mathrm{influx} =
\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_{v\_\mathrm{threshold}} and the metabolic power required to
produce a spike is approximately: \mathcal{H}\_\mathrm{spike} =
v\partial I\_\mathrm{influx}

The values \mathcal{H}\_\mathrm{rest} and \mathcal{H}\_\mathrm{spike}
are helpful because, at any moment t, whether there is a spike depends
on how they relate. If \mathcal{H}\_\mathrm{spike} \<
\mathcal{H}\_\mathrm{rest}, then the neuron will be more inclined to
spike. Under this assumption, we can suppose that f is a function of the
difference between \mathcal{H}\_\mathrm{spike} and
\mathcal{H}\_\mathrm{rest}: \begin{aligned}
f\left(\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t\right) &=
f\left(\mathcal{H}\_\mathrm{spike} - \mathcal{H}\_\mathrm{rest}\right)
\\ &= f\left(v\partial I\_\mathrm{influx} -
v\_\mathrm{rest}\left.\frac{\partial\mathcal{H}}{\partial
v}\right\|\_t\right) \end{aligned} A suitable function f follows from
the previously cited lemma that f\approx 1 when v\approx
v\_\mathrm{rest}. A plausible way to achieve this result without f
collapsing into a trivial constant is to set: f = \frac{ v\partial
I\_\mathrm{influx} -
v\_\mathrm{rest}\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t
}{ v\_\mathrm{rest}\partial I\_\mathrm{influx} -
v\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t }

## Interpretation

While the motivation for this definition of f is mathematical (we want
f\approx 1 when v\approx v\_\mathrm{rest}), the equation makes some
biological sense. We have already discussed the numerator:
v\_\mathrm{rest}\left.\partial\mathcal{H}/\partial v\right\|\_t is the
metabolic power \mathcal{H}\_\mathrm{rest} of maintaining rest potential
v\_\mathrm{rest} under current influx at time t, v\partial
I\_\mathrm{influx} is the metabolic power \mathcal{H}\_\mathrm{spike} of
initiating a spike under that same condition, and their difference
signals whether it takes more metabolic power to spike or maintain rest.

In the denominator, v\_\mathrm{rest}\partial I\_\mathrm{influx} is the
metabolic power of initiating a spike from the rest potential, while
v\left.\partial\mathcal{H}/\partial v\right\|\_t is the metabolic power
of maintaining the potential v(t) under the current influx
\left.\partial\mathcal{H}/\partial v\right\|\_t holding at time t.
Hence, the difference between these two quantities is the amount of
additional metabolic power needed to (hypothetically) initiate a spike
from rest, compared to the power needed to maintain the cell’s present
state.

Thus, the denominator provides a kind of upper bound, or normalization.
The scale factor f is plausibly interpreted as the additional power
needed to spike (vs maintain rest potential) as a fraction of the max
possible power needed to initiate a spike. It gives a normalized “cost”
of the power (and hence energy) for a spike.

## Power gradient

Returning to \partial v/\partial t, we have: \left.\frac{\partial
v}{\partial t}\right\|\_t = v\_\mathrm{rest} \left( \frac{ v\partial
I\_\mathrm{influx} -
v\_\mathrm{rest}\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t
}{ v\_\mathrm{rest}\partial I\_\mathrm{influx} -
v\left.\frac{\partial\mathcal{H}}{\partial v}\right\|\_t }\right) -
v(t-1) This is the equation given by [Gangopadhyay and
Chakrabartty](https://doi.org/10.3389/fnins.2020.00425) for GT models.
It ensures the minimization condition, that is, ensures that
\partial\mathcal{H}/\partial t = 0.

How do we determine \partial I\_\mathrm{influx}? Well, \partial
I\_\mathrm{influx} is the change in current I across the membrane at
spike initiation. Following Gangopadhyay and Chakrabartty, we will treat
it as a constant empirical parameter that bounds the power gradient,
i.e., \partial I\_\mathrm{influx} \geq
\left.\partial\mathcal{H}/\partial v\right\|\_t for all t. This
treatment is plausible, as presumably the moment of spike initiation
involves the largest change in membrane current a neuron will
experience.[^4]

Second, how do we determine \partial\mathcal{H}/\partial v? Intuitively
(and, as before, following Gangopadhyay and Chakrabartty), we have:
\frac{\partial\mathcal{H}}{\partial v} =
I\_\mathrm{synaptic\\transmission} - I\_\mathrm{membrane\\current} +
I\_\mathrm{spike} In this equation, I\_\mathrm{synaptic\\transmission}
is the input current induced by synaptic transmission across all
synapses, I\_\mathrm{membrane\\current} gives the current across the
cell membrane due to external stimuli (I\_\mathrm{stim}) and intrinsic
membrane leak (I\_\mathrm{leak}), and I\_\mathrm{spike} gives the spike
current (if any). The stimulus current I\_\mathrm{stim} is what’s
specified by the simulation variable stimulus_current_matrix, while the
membrane leak current I\_\mathrm{leak} is determiend by the cell type
parameter leak_conductance. The spike current I\_\mathrm{spike} is
determined by simple thresholding: I\_\mathrm{spike}=0 if v \<
v\_\mathrm{threshold} and otherwise is equal to the value of the
simulation variable I_spike for the cell type.

The input current induced by synaptic transmission across all synapses,
I\_\mathrm{synaptic\\transmission}, is handled specially to account for
signal transmission lag, as described in [the tutorial on spatial GT
models](https://Oviedo-Lab.org/DACx/articles/tutorial_SGT.md). However,
the basic idea is straightforward: the synaptic current is the synaptic
conductance times the presynaptic membrane potential v.

## Spiking without spikes?

GT models are only ever modelling *subthreshold* dynamics. That is, v
represents only the membrane potential below the spike threshold. This
arrangement leads to an awkward interpretation of
I\_\mathrm{synaptic\\transmission}. Specifically, computing
I\_\mathrm{synaptic\\transmission} as synaptic conductance times v
implies that it’s the subthreshold membrane potential of the presynaptic
neuron which is transduced across the synapse to the postsynaptic
neuron. Biologically, signal transduction at synapses happens primarily
through spikes, of course. So, two questions naturally arise:

1.  If it’s the subthreshold membrane potential v which drives synaptic
    transmission in a GT model, how do the spikes themselves have any
    causal role in signal transmission?
2.  Whatever the answer to the first question, why arrange a spiking
    neural network model in this strange way?

The answer to the first question is that, in GT models, spikes of
presynaptic neurons causally affect the membrane potential of
postsynaptic neurons indirectly, via their effect on the subthreshold
membrane potential of the presynaptic neuron. This is, of course, the
inverse of the causal flow in the real biology. In real neurons, changes
in subthreshold membrane potential cause spikes, which cause synaptic
transmission. In GT models, the energy cost of a future spike causes
changes in the subthreshold membrane potential, which itself is engaged
in continuous synaptic transmission. Mathematically, that future energy
cost appears as the term I\_\mathrm{spike} in the metabolic power
gradient \partial\mathcal{H}/\partial v, and as the term
\mathcal{H}\_\mathrm{spike} in the scale factor f.

As for the second question, the advantage of flipping the causal order
and modelling the energy cost of a spike instead of a spike itself is
that it makes it easier to optimize the synaptic “weights”, i.e., the
matrix Q of synaptic conductances, so that the network model reproduces
a desired response in v for a given stimulus input current
I\_\mathrm{stimulus\\input}. Traditionally, spiking neural networks are
trained by optimizing synaptic weights to minimize the difference
between some time-dependent desired spike rate and the model spike rate.
However, if R(w,t,n) is the spike rate implied by a spiking neural
network for neuron n at time t under synaptic weights w, R will not
usually be differentiable due to spiking being a function of discrete
threshold crossings. This means that spiking neural networks can’t
straightforwardly be trained using gradient descent. In contrast,
\mathcal{H} is a continuous function of v with a well defined derivative
\partial\mathcal{H}/\partial v, and v itself, being only the
subthreshold membrane potential, is also smooth.

## Membrane potential reset

Notice that unlike a leaky integrate-and-fire neuron, \partial
v/\partial t itself produces the reset after a spike. When v is very
near the threshold v\_\mathrm{threshold}, then (by the definition given
above) we have that: \partial I\_\mathrm{influx} =
\left.\frac{\partial\mathcal{H}}{\partial v}
\right\|\_{v\_\mathrm{threshold}} and hence f\approx -1, because, in
general, (x-y)/(y-x)=-1. Thus, when v\approx v\_\mathrm{threshold}: v +
\frac{\partial v}{\partial t }\approx -v\_\mathrm{rest} At first glance,
this seems off by a sign: shouldn’t \$v/t \$ return us to
v\_\mathrm{rest} when near v\_\mathrm{threshold}? Yes, but the issue is
easy enough to fix: simply add a negation to our definition of f. In the
actual DACx code, the relevant multiple for f is not v\_\mathrm{rest},
but \|v\_\mathrm{rest}\| + \epsilon for some small \epsilon, but the
result is the same: the model produces a reset to a value near the rest
potential after a spike, without any explicit reset mechanism.

[^1]: Hereafter we leave the qualifier “subthreshold” implicit.

[^2]: Technically, the resting potential is the equilibrium point
    towards which the membrane potential tends in the absence of any
    input.

[^3]: [Gangopadhyay and
    Chakrabartty](https://doi.org/10.3389/fnins.2020.00425) assume that
    the absolute value of the quantity of which membrane potential is a
    multiple is a bound on membrane potential. Within the simulation
    code, the relevant variable is v_bound, which is given a value
    slightly higher than the absolute value of the rest potential. If
    the value isn’t increased slightly, the rest potential becomes an
    infinite well that can’t be escaped.

[^4]: In the code for DACx, this bound on the power gradient is variable
    dHdv_bound and is set to be slightly more than the absolute value of
    the spike current I_spike.

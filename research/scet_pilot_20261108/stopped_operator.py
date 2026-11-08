"""Exact reference semantics for observable stopping boundaries.

This is a specification layer for learned kernels, not a learned safety
certificate. Information keys denote observations the agent will actually see.
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Boundary:
    ticks: int
    kind: str
    information: str
    probability: Fraction


@dataclass(frozen=True)
class StoppedKernel:
    deadline: int
    age: int
    boundaries: tuple[Boundary, ...]

    def __post_init__(self):
        if not isinstance(self.age,int) or not isinstance(self.deadline,int) or not 0<=self.age<self.deadline:
            raise ValueError('Invalid original deadline/age')
        if not self.boundaries or sum(b.probability for b in self.boundaries)!=1:
            raise ValueError('Kernel must have exact unit mass, including tail')
        for b in self.boundaries:
            if not isinstance(b.probability,Fraction) or b.probability<0:
                raise ValueError('Reference probabilities must be nonnegative Fractions')
            if not isinstance(b.ticks,int) or not 1<=b.ticks<=self.remaining:
                raise ValueError('Boundary outside original remaining deadline')
            if b.kind not in {'goal','collision','observe','timeout'}:
                raise ValueError('Unknown boundary kind')
            if b.kind=='timeout' and b.ticks!=self.remaining:
                raise ValueError('Administrative timeout must be at original deadline')
            if b.kind=='observe' and not b.information:
                raise ValueError('Observation boundary needs an observable information key')

    @property
    def remaining(self):
        return self.deadline-self.age

    def resume(self,elapsed):
        """Condition on absence of ANY boundary; never restart its clock."""
        if not isinstance(elapsed,int) or not 0<=elapsed<self.remaining:
            raise ValueError('Resume must precede original deadline')
        survivors=[b for b in self.boundaries if b.ticks>elapsed]
        mass=sum(b.probability for b in survivors)
        if mass==0:
            raise ValueError('Cannot condition on zero-probability survival')
        return StoppedKernel(self.deadline,self.age+elapsed,tuple(
            Boundary(b.ticks-elapsed,b.kind,b.information,b.probability/mass) for b in survivors))

    def compose(self,continuations):
        """Continue after observed boundary, using only its observable key.

        Dictionary key is (relative boundary time, observed information), never
        a latent component index. Unobservable duplicates share continuation.
        """
        combined=[]
        for b in self.boundaries:
            if b.kind!='observe':
                combined.append(b)
                continue
            if b.ticks==self.remaining:
                combined.append(Boundary(b.ticks,'timeout','',b.probability))
                continue
            nxt=continuations[(b.ticks,b.information)]
            if nxt.deadline!=self.deadline or nxt.age!=self.age+b.ticks:
                raise ValueError('Continuation must preserve original task deadline and attained age')
            for c in nxt.boundaries:
                combined.append(Boundary(b.ticks+c.ticks,c.kind,c.information,b.probability*c.probability))
        return StoppedKernel(self.deadline,self.age,tuple(combined))

    def expected(self,terminal_costs):
        """Elapsed duration plus boundary-dependent continuation/terminal cost."""
        return sum(b.probability*(b.ticks+terminal_costs[(b.kind,b.information)]) for b in self.boundaries)

    def observation_value(self,action_costs,terminal_costs):
        """E[min action cost after observation], keyed by actual information."""
        return sum(b.probability*(b.ticks+(
            min(action_costs[(b.ticks,b.information)].values()) if b.kind=='observe'
            else terminal_costs[(b.kind,b.information)])) for b in self.boundaries)

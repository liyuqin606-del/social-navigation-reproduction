"""Commit and advance a planned option; no simulator or hidden-state access."""
from .planner import Outcome

class PolicyCursor:
    def __init__(self, plan):
        self.plan=plan
        self.closed=plan.time_left==0
        self.invalid=False

    def option(self):
        if self.closed or self.invalid:raise RuntimeError('No executable certified continuation')
        return self.plan.option

    def observe(self, outcome):
        """Called only after executing the currently committed option.
        Unsupported events invalidate the model contract; they do not replan.
        """
        option=self.option()
        support={o for k in option.kernels for o,p in k if p}
        if outcome not in support:
            self.invalid=True
            raise ValueError('Observed outcome outside model support; halt certified execution')
        if outcome.event=='R':
            self.plan=dict(self.plan.children)[outcome]
            self.closed=self.plan.time_left==0
        else:self.closed=True
        return {'closed':self.closed,'budget':self.plan.budget,'scope':self.plan.scope}

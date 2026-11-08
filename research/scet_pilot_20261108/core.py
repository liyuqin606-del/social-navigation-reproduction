"""Finite event model pilot. NumPy only; no hidden simulator state in models."""
import numpy as np

H = 16
N = 2 * H + 1
DETOUR = 17.0
COLLISION = 60.0


def features(cues, ages, modes):
    cues = np.asarray(cues)
    return np.column_stack([cues, np.asarray(ages)/H, np.eye(3)[modes]])


def mask_for(ages):
    # Absolute event times retain original deadline at every resumed prefix.
    times = np.repeat(np.arange(1, H+1), 2)
    return np.column_stack([times[None, :] > np.asarray(ages)[:, None],
                            np.ones(len(ages), dtype=bool)])


def softmax(logits, mask=None):
    if mask is not None:
        logits = np.where(mask, logits, -1e30)
    ex = np.exp(logits-logits.max(axis=1, keepdims=True))
    return ex/ex.sum(axis=1, keepdims=True)


def softmax_backward(p, g):
    return p*(g-(p*g).sum(axis=1, keepdims=True))


class Net:
    def __init__(self, output, hidden, seed):
        rng = np.random.default_rng(seed)
        self.p = [rng.normal(0, 1/np.sqrt(6), (6, hidden)), np.zeros(hidden),
                  rng.normal(0, 1/np.sqrt(hidden), (hidden, output)), np.zeros(output)]
        self.m = [np.zeros_like(v) for v in self.p]
        self.v = [np.zeros_like(v) for v in self.p]
        self.tick = 0

    def forward(self, x):
        h = np.tanh(x @ self.p[0]+self.p[1])
        return h @ self.p[2]+self.p[3], (x, h)

    def backward(self, cache, grad):
        x, h = cache
        gh = (grad @ self.p[2].T)*(1-h*h)
        return [x.T @ gh, gh.sum(axis=0), h.T @ grad, grad.sum(axis=0)]

    def update(self, gradients, lr):
        self.tick += 1
        for p, m, v, g in zip(self.p, self.m, self.v, gradients):
            m *= .9
            m += .1*g
            v *= .999
            v += .001*g*g
            p -= lr*(m/(1-.9**self.tick))/(np.sqrt(v/(1-.999**self.tick))+1e-8)


def prefix_loss(net, x, age, y, step=False):
    logits, cache = net.forward(x)
    p = softmax(logits, None if step else mask_for(age))
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean()
    grad = p.copy()
    grad[np.arange(len(y)), y] -= 1
    return loss, net.backward(cache, grad/len(y))


def consistency_loss(net, x, ages):
    """Conditional-kernel identity when survival is the only new observation.

    r = normalize(K_age over event times >age+1 and horizon tail)
    q = K_age+1. Full gradient through r and q; no teacher or oracle.
    """
    z, cache = net.forward(x)
    xnext = x.copy()
    xnext[:, 2] += 1/H
    znext, cnext = net.forward(xnext)
    mask = mask_for(ages+1)
    r, q = softmax(z, mask), softmax(znext, mask)
    diff = r-q
    loss = .5*np.square(diff).sum(axis=1).mean()
    g = net.backward(cache, softmax_backward(r, diff)/len(x))
    gn = net.backward(cnext, softmax_backward(q, -diff)/len(x))
    return loss, [a+b for a,b in zip(g, gn)]


def components():
    """Simulator-only latent fast/slow and left/right intent kernels."""
    out = np.zeros((4, 3, N))
    for slow in range(2):
        for side in range(2):
            for mode in range(3):
                survive = 1.0
                for t in range(1, H+1):
                    peak = (3 if not slow else 12) + (mode == 0)
                    good = .025+.34*np.exp(-.5*((t-peak)/1.8)**2)
                    if mode:
                        good *= 1.2 if mode-1 == side else .75
                    bad = .002 if mode == 0 else (.005 if mode-1 == side else .055)
                    idx = 2*slow+side
                    out[idx, mode, 2*(t-1)] = survive*good
                    out[idx, mode, 2*(t-1)+1] = survive*bad
                    survive *= 1-good-bad
                out[idx, mode, -1] = survive
    return out


def mixture(cues):
    prob = 1/(1+np.exp(-2.5*np.asarray(cues)))
    slow, side = prob[:, 0], prob[:, 1]
    return np.column_stack([(1-slow)*(1-side), (1-slow)*side,
                            slow*(1-side), slow*side])


def true_kernels(cues):
    return np.einsum('bi,imk->bmk', mixture(cues), components())


def training_data(count, seed):
    rng = np.random.default_rng(seed)
    cues = rng.uniform(-1, 1, (count, 2))
    kernels = true_kernels(cues)
    records = []
    for i in range(count):
        for mode in range(3):
            event = rng.choice(N, p=kernels[i, mode])
            time = H+1 if event == N-1 else event//2+1
            records.append([*cues[i], mode, time, event%2 if time<=H else 2])
    return np.asarray(records, dtype=float)


def prefixes(records):
    rows = []
    for c0, c1, mode, time, kind in records:
        for age in range(min(int(time), H)):
            target = N-1 if time>H else 2*(int(time)-1)+int(kind)
            next_target = int(kind) if time == age+1 else 2
            rows.append([c0, c1, mode, age, target, next_target])
    return np.asarray(rows)


def event_predictions(net, cues):
    n = len(cues)
    ages = np.tile(np.repeat(np.arange(H+1), 3), n)
    modes = np.tile(np.arange(3), n*(H+1))
    x = features(np.repeat(cues, (H+1)*3, axis=0), ages, modes)
    z, _ = net.forward(x)
    return softmax(z, mask_for(ages)).reshape(n, H+1, 3, N)


def step_predictions(net, cues):
    n = len(cues)
    ages = np.tile(np.repeat(np.arange(H), 3), n)
    modes = np.tile(np.arange(3), n*H)
    x = features(np.repeat(cues, H*3, axis=0), ages, modes)
    z, _ = net.forward(x)
    hazard = softmax(z).reshape(n, H, 3, 3)
    out = np.zeros((n, H+1, 3, N))
    for age in range(H+1):
        survival = np.ones((n, 3))
        for t in range(age, H):
            out[:, age, :, 2*t:2*t+2] = survival[:, :, None]*hazard[:, t, :, :2]
            survival *= hazard[:, t, :, 2]
        out[:, age, :, -1] = survival
    return out


def conditional_truth(cues):
    raw = true_kernels(cues)
    out = np.zeros((len(cues), H+1, 3, N))
    for age in range(H+1):
        p = raw*mask_for(np.array([age]))[:, None, :]
        out[:, age] = p/p.sum(axis=-1, keepdims=True)
    return out


def cutoff_cost(kernel, age, cutoff):
    """Expected cost of continuing until absolute cutoff then detouring."""
    duration = cutoff-age
    good = kernel[..., 2*age:2*cutoff:2]
    bad = kernel[..., 2*age+1:2*cutoff:2]
    ticks = np.arange(1, duration+1)
    tail = 1-good.sum(axis=-1)-bad.sum(axis=-1)
    return (good*(ticks+1)).sum(axis=-1)+(bad*(ticks+COLLISION)).sum(axis=-1)+tail*(duration+DETOUR)


def policy(kernel):
    """Replan every survival prefix; keep chosen controller and original deadline."""
    n = len(kernel)
    best = np.zeros((n, H+1, 3), dtype=int)
    values = np.zeros((n, H+1, 3))
    for age in range(H+1):
        costs = np.stack([cutoff_cost(kernel[:, age], age, k) for k in range(age,H+1)],axis=-1)
        best[:, age] = costs.argmin(axis=-1)+age
        values[:, age] = costs.min(axis=-1)
    mode = values[:, 0].argmin(axis=-1)
    cutoff = np.full(n, H, dtype=int)
    for i in range(n):
        for age in range(H+1):
            if best[i, age, mode[i]] == age:
                cutoff[i] = age
                break
    return mode, cutoff


def evaluate(kernel, truth):
    mode, cut = policy(kernel)
    costs, collisions, durations = [], [], []
    for i, (m,k) in enumerate(zip(mode,cut)):
        p = truth[i, 0, m]
        costs.append(float(cutoff_cost(p, 0, k)))
        collisions.append(float(p[1:2*k:2].sum()))
        ticks = np.arange(1,k+1)
        mass = p[:2*k].reshape(k,2).sum(axis=1)
        durations.append(float((mass*ticks).sum()+(1-mass.sum())*k))
    # Exact distributional log score under original context/mode occupancy.
    # This scores every observed survival prefix, normalized by total occupancy.
    raw = truth[:, 0]
    weighted_kl = 0.0
    weight = 0.0
    for age in range(H):
        occupancy = raw[..., 2*age:].sum(axis=-1)
        p = truth[:, age]
        kl = (p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(kernel[:,age],1e-300)))).sum(axis=-1)
        weighted_kl += float((occupancy*kl).sum())
        weight += float(occupancy.sum())
    consistency = []
    for age in range(H-1):
        p = kernel[:, age].copy()
        p[..., :2*(age+1)] = 0
        p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-300)
        consistency.append(np.square(p-kernel[:,age+1]).sum(axis=-1).mean())
    return dict(cost=np.array(costs), collision=np.array(collisions), option_ticks=np.array(durations),
                mode=mode, cutoff=cut, kernel_kl=weighted_kl/weight,
                consistency=float(np.mean(consistency)))

def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )

def retro():
    return atari()

def user():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        save_interval=100,
        ent_coef=0.0,
        vf_coef=0.5,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy',
        # superv_coef=0.0,
        # sil_value=0.01,
        # sil_update=10,
        # sil_alpha=0.6,
        # sil_beta=0.1
    )

def user_defined():
    defaults = mujoco()
    defaults.update(dict(nsteps=128,
                         nminibatches=4,
                         noptepochs=4,))
    return defaults

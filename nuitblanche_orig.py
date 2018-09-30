from stft import *
from pvoc import *
from numpy import *
def tri(x):
    return abs((x%2*pi)-pi)/pi
def squ(x):
    return sign((x%2*pi)-pi)

stft_defs = {
        8: STFT(256, 2, 2),
        9: STFT(512, 2, 2),
        10: STFT(1024, 2, 2),
        11: STFT(2048, 2, 2),
        12: STFT(4096, 2, 2),
        13: STFT(8192, 2, 2),
        14: STFT(16384, 2, 2),
        15: STFT(32768, 2, 2),
        16: STFT(65536, 2, 2),
        17: STFT(65536*2, 2, 2)
        }
stft_defs2 = {
        8: STFT(256, 2, 2),
        9: STFT(512, 2, 2),
        10: STFT(1024, 2, 2),
        11: STFT(2048, 2, 2),
        12: STFT(4096, 2, 2),
        13: STFT(8192, 2, 2),
        14: STFT(16384, 2, 2),
        15: STFT(32768, 2, 2),
        16: STFT(65536, 2, 2),
        17: STFT(65536*2, 2, 2)
        }

pvoc_defs = {}
pvoc_defs2 = {}
def gen_pvocs():
    for size in stft_defs:
        pvoc_defs[size] = PhaseVocoder(stft_defs[size])
def gen_pvocs2():
    for size in stft_defs2:
        pvoc_defs2[size] = PhaseVocoder(stft_defs2[size])
gen_pvocs()
gen_pvocs2()
def pvocwrap(x):
    return (x,)

fademax = 500
fader = 0 # 0 = fns, fademax = fadefns
fadefns = {}
fadetime = 0
direction = 1
def fadeto(fn):
    global fns
    global fadefns
    global fader
    global direction
    global time
    global fadetime
    direction *= -1
    if direction == -1:
        fns = fn
        time = 0
        gen_pvocs()
    else:
        fadefns = fn
        fadetime = 0
        gen_pvocs2()
    print(fns, fadefns)
def fadeto_cont(fn):
    global fns
    global fadefns
    global fader
    global direction
    global time
    global fadetime
    direction *= -1
    if direction == -1:
        fns = fn
    else:
        fadefns = fn
    print(fns, fadefns)

fadeto({
    })

def process(i, o):
    o[:] = 0
    global time
    global fadetime
    global fader
    fader = min(fademax, max(0, fader + direction))
    fo = copy(o)
    for size in stft_defs:
        if fader != fademax and size in fns and len(fns[size]) > 0:
            stft = stft_defs[size]
            for x in stft.forward(o):
                idx = indices(x.shape)
                t = (time*x.shape[1]+idx[1])
                for fn in fns[size]:
                    if type(fn) == tuple:
                        pv = pvoc_defs[size]
                        x[:] = pv.forward(x)
                        fn[0](time, copy(t), idx, x, pv)
                        x[:] = pv.backward(x)
                    else:
                        fn(time, copy(t), idx, x)
                x *= (fademax - fader)/fademax
                stft.backward(x)
                time += 1
            stft.pop(o)
        if fader != 0 and size in fadefns and len(fadefns[size]) > 0:
            stft = stft_defs2[size]
            for x in stft.forward(fo):
                idx = indices(x.shape)
                t = (fadetime*x.shape[1]+idx[1])
                for fn in fadefns[size]:
                    if type(fn) == tuple:
                        pv = pvoc_defs2[size]
                        x[:] = pv.forward(x)
                        fn[0](fadetime, copy(t), idx, x, pv)
                        x[:] = pv.backward(x)
                    else:
                        fn(fadetime, copy(t), idx, x)
                x *= fader/fademax
                stft.backward(x)
                fadetime += 1
            stft.pop(fo)
    o += fo


def f1(time, _, idx, x, pv):
    time += 1<<18
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t//6&t>>12))*4096
    x[:] = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (ph*fq)/4096%22050)
fadeto({
    12: [pvocwrap(f1)],
    })

def f2(time, _, idx, x, pv):
    time += 10000 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t//6&t>>12&t>>13))*4096
    x[:] = sin(z*ph)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (ph*fq)/1024%22050)
fadeto({
    14: [pvocwrap(f2)],
    })

def f15(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = (t//555&t//7777&t>>13+idx[0])%((1<<6))
    x[:] = sin(t)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*((z)%idx[1].shape[1]+1)/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (fq//32*32+z*8)%22050)
    p = ((t//16&t//64|t//512)//1024)^(idx[1]/idx[1].shape[1]*22050).astype(int)
    x[:] = pv.shift(x, lambda fq: (p)%22050)
    x[:].real /= log1p(z)+1
fadeto({
    14: [pvocwrap(f15)],
    })

def f14(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = (t>>4^t>>8^t>>11+idx[0])%(1<<16)
    x[:] = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*((idx[1]+z)%idx[1].shape[1]+1)/idx[1].shape[1]*22050
fadeto({
    14: [pvocwrap(f14)],
    })

def f6(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = ((t//8192&t//1024&t//32))**(1/128)%65536
    x[:] = sin(idx[1]+idx[0])/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: ((fq).astype(int)^(ph).astype(int))%22050)
fadeto({
    14: [pvocwrap(f6)],
    })

def f7(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = ((t//8192^t//2048^t//512&t//32+idx[0]))**(1/128)%65536
    x[:] = sin(ph*64+idx[0])/fmax((1+idx[1]**0.9), 1)*x.shape[1]/256 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: ((fq).astype(int)^(ph).astype(int))/2%22050)
fadeto({
    14: [pvocwrap(f7)],
    })

def f8(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = ((t//32767^t//2049^t//511&t//33&t//7^idx[0]))**(1/128)
    x[:] = sin(ph*64+t)/fmax((1+idx[1]**0.9), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: ((fq).astype(int)|(ph/2).astype(int)&t)%22050+ph*8%64)
fadeto({
    14: [pvocwrap(f8)],
    })

def f9(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    ot = time
    time //= 16
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t>>13&t>>11&t//7))*64
    x[:] = sin(t*ot)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: ((ph/(0.5+ot/256%1)*idx[1]).astype(int)^fq.astype(int))%22050)
fadeto({
    10: [pvocwrap(f9)],
    })

def f10(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    ot = time
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph=(idx[1]&t//3+idx[0])//32
    x[:] = sin(t)/fmax((1+idx[1]**0.9), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (fq*3+sin(ph)*22050/idx[1].shape[1])%22050)
fadeto({
    12: [pvocwrap(f10)],
    })

def f11(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    ot = time
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph=(idx[1]&t//3+idx[0])//128
    tune=(t//9933&t//4993+idx[0])&31
    x[:] = sin((t//4)**1.5)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (tune+(fq))%22050)
fadeto({
    12: [pvocwrap(f11)],
    })

def f12(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    x[:] = sin(t)/fmax((1+idx[1]**1.2), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    ph = ((t|t>>1|t>>2^t>>4&t>>7&t>>11&t>>14)%((1<<14)+idx[1]))*(idx[1])+idx[0]
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**1.2), 1)*x.shape[1]/1024
    oph=ph
    ph = ((t>>6&t>>10&t>>14)**2)+idx[0]*0.1
    #pv.shift(x, lambda fq: ((fq).astype(int)^(ph).astype(int))%22050).real
    #pv.shift(x, lambda fq: (ph)%22050).real
fadeto({
    9: [pvocwrap(f12)],
    })

def f13(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    x[:] = sin(t)/fmax((1+idx[1]**1.2), 1)*x.shape[1]/512 + 1j*idx[1]/idx[1].shape[1]*22050
    ph = ((t|t>>1|t>>2^t>>4&t>>7&t>>11&t>>14)%((1<<14)+idx[1]))*(idx[1])+idx[0]
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**1.2), 1)*x.shape[1]/1024
    oph=ph
    ph = ((t>>4&t>>10&t>>14)**1)+idx[0]*0.1
    #pv.shift(x, lambda fq: ((fq).astype(int)^(ph).astype(int))%22050).real
    x[:]=pv.shift(x, lambda fq: (ph+fq)%22050)
fadeto({
    10: [pvocwrap(f13)],
    })

def f5(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t//512&t//37))*4096
    x[:] = sin(idx[1]+idx[0])/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: fq+(ph)%10)
fadeto({
    15: [pvocwrap(f5)],
    })

def f4(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t//512&t//17))*4096
    x[:] = sin(idx[1]+idx[0])/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: fq+(ph)%10)
fadeto({
    16: [pvocwrap(f4)],
    })

def f3(time, _, idx, x, pv):
    time += 0 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t//512&t//7))*4096
    x[:] = sin(idx[1]+idx[0])/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: fq+(ph)%10)
fadeto({
    16: [pvocwrap(f3)],
    })

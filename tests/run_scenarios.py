#!/usr/bin/env python3
"""Testa o guia de foto (index.html) com câmeras falsas no Chrome headless.

Gera um vídeo .y4m por cenário (folha do gabarito em perspectiva sobre um fundo),
abre a página com --use-file-for-fake-video-capture e anota a mensagem final e
se o botão ficou verde.

  python3 tests/run_scenarios.py                      # todos, retrato + paisagem
  python3 tests/run_scenarios.py good white -o port   # só alguns
  python3 tests/run_scenarios.py --page /tmp/old.html # outra versão da página
  python3 tests/run_scenarios.py --soak 120 good      # 2 min rodando: memória

Requer: opencv-python, numpy, playwright, /usr/bin/google-chrome e o PNG do gabarito.
"""
import argparse, functools, hashlib, http.server, os, shutil, sys, tempfile, threading, time, urllib.request
import cv2, numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
GAB = os.environ.get('GABARITO_PNG', '/home/claude/ampola-mvp/fixtures/gabarito_300dpi.png')
TMP = os.path.join(tempfile.gettempdir(), 'ampola_scen')
CDN = {'https://docs.opencv.org/4.7.0/opencv.js': 'opencv.js',
       'https://cdn.jsdelivr.net/npm/jscanify@1.4.0/src/jscanify.min.js': 'jscanify.min.js'}
rng = np.random.default_rng(7)

# ---------- cenários: (fundo, tamanho da folha no quadro, giro, inclinação, extras) ----------
#   h: 1 = folha inteira cabe justa no quadro;  exp = mensagem esperada (ver EXPECT)
SCEN = {
 'good':     dict(bg='wood',  h=.80, exp='ok'),
 'far':      dict(bg='wood',  h=.42, exp='perto'),
 'close':    dict(bg='wood',  h=1.15, exp='afaste'),
 'tilt':     dict(bg='wood',  h=.78, tilt=.62, exp='endireite'),
 'sideways': dict(bg='wood',  h=.80, rot=90, exp='gire'),       # folha deitada, celular em pé
 'mismatch': dict(bg='wood',  h=.80, exp='afaste|gire', vp='port'),  # vídeo deitado numa tela em pé
 'rot30':    dict(bg='wood',  h=.85, rot=30, exp='alinhe'),
 'rot10':    dict(bg='wood',  h=.80, rot=10, exp='ok'),
 'white':    dict(bg='white', h=.80, exp='ok'),
 'dark':     dict(bg='black', h=.80, exp='ok'),
 'dim':      dict(bg='wood',  h=.80, gain=.22, noise=6, exp='ok|luz'),
 'bright':   dict(bg='wood',  h=.80, gain=2.6, exp='luz|ok'),
 'blur':     dict(bg='wood',  h=.80, motion=19, exp='firme'),
 'blurmild': dict(bg='wood',  h=.80, motion=7, exp='firme|ok'),
 'shadow':   dict(bg='wood',  h=.80, shade=.45, exp='ok'),           # sombra/luz desigual na folha
 'noisy':    dict(bg='wood',  h=.80, gain=.55, noise=12, exp='ok'),   # pouca luz, celular compensou (ruído)
 'good4k':   dict(bg='wood',  h=.80, size=(2160, 3840), exp='ok'),
 'clutter':  dict(bg='glass', h=0, bags=True, exp='procurando'),
 'clutterwood': dict(bg='wood', h=0, bags=True, exp='procurando'),
 'hand':     dict(bg='wood',  h=.80, hand=True, exp='mostra'),
 'ampoule':  dict(bg='wood',  h=.80, ampoule=True, exp='ok'),
 'plain':    dict(bg='wood',  h=.80, plain=True, exp='gabarito'),
 'plainwhite': dict(bg='glass', h=.80, plain=True, bags=True, exp='gabarito|procurando'),
}
EXPECT = {'ok': 'Pode tirar', 'perto': 'Chegue mais perto', 'afaste': 'Afaste', 'endireite': 'Endireite',
          'gire': 'Gire', 'alinhe': 'Alinhe', 'luz': 'luz', 'firme': 'firme', 'procurando': 'Procurando', 'mostra': 'à mostra',
          'gabarito': 'quadradinhos pretos'}

@functools.lru_cache(None)
def sheet(plain=False, ampoule=False):
    g = cv2.imread(GAB)
    g = cv2.resize(g, (1240, 1754), interpolation=cv2.INTER_AREA)  # 150 dpi
    g = (g.astype(np.float32) * 0.94).astype(np.uint8)              # papel ~240, não 255
    if plain: g[:] = 240
    if ampoule:  # ampola de vidro âmbar deitada no tracejado, com rótulo
        cx, cy = 620, 640
        cv2.ellipse(g, (cx, cy), (330, 62), 0, 0, 360, (40, 90, 150), -1)
        cv2.rectangle(g, (cx - 180, cy - 58), (cx + 120, cy + 58), (235, 235, 235), -1)
        cv2.putText(g, 'LIDOCAINA 2%', (cx - 170, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 20, 20), 3)
        cv2.ellipse(g, (cx, cy - 30), (300, 10), 0, 0, 360, (200, 230, 250), -1)  # reflexo
    return g

def background(kind, W, H):
    if kind == 'wood':
        x = np.linspace(0, 40, W)[None, :] + rng.normal(0, 3, (H, 1))
        base = 95 + 18 * np.sin(x)[..., None] * np.ones((1, 1, 3))
        img = base * np.array([0.55, 0.75, 1.0])
    elif kind == 'white':
        img = np.full((H, W, 3), 222.0) + np.linspace(-6, 6, W)[None, :, None]
    elif kind == 'black':
        img = np.full((H, W, 3), 28.0)
    else:  # vidro esverdeado com reflexos
        img = np.full((H, W, 3), (118, 128, 120), np.float64)
        for _ in range(6):
            p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            cv2.line(img, p, (p[0] + 300, p[1] - 200), (175, 185, 180), int(rng.integers(4, 18)))
    img = img + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def place(W, H, h, rot=0, tilt=1.0):
    """Cantos da folha (A4 em pé, girada rot°) no quadro; h=1 → a folha inteira cabe justa."""
    a = np.deg2rad(rot); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[-.5, -.707], [.5, -.707], [.5, .707], [-.5, .707]]) @ R.T
    pts *= h * min(W / np.ptp(pts[:, 0]), H / np.ptp(pts[:, 1]))
    if tilt != 1:  # perspectiva: borda de cima (mais longe) menor e mais curta na vertical
        pts[:2, 0] *= tilt; pts[:2, 1] *= (0.5 + tilt / 2)
    return (pts + [W/2, H/2]).astype(np.float32)

def frame(s, W, H, jitter=0):
    img = background(s['bg'], W, H)
    if s.get('bags'):  # sacolas plásticas brancas (uma quase retangular)
        for i in range(3):
            c = np.array([W * (0.3 + 0.25 * i), H * (0.35 + 0.15 * i)])
            r = min(W, H) * (0.28 if i == 0 else 0.15)
            ang = np.sort(rng.uniform(0, 2*np.pi, 9))
            poly = c + np.stack([np.cos(ang), np.sin(ang) * 1.3], 1) * r * rng.uniform(.75, 1.1, (9, 1))
            cv2.fillPoly(img, [poly.astype(np.int32)], (232, 236, 236))
            for _ in range(5):
                q = (c + rng.normal(0, r/3, 2)).astype(int)
                cv2.line(img, tuple(q), tuple(q + rng.integers(-60, 60, 2)), (190, 196, 196), 3)
    if s['h']:
        g = sheet(s.get('plain', False), s.get('ampoule', False))
        dst = (place(W, H, s['h'], s.get('rot', 0), s.get('tilt', 1.0)) + jitter).astype(np.float32)
        src = np.float32([[0, 0], [g.shape[1], 0], [g.shape[1], g.shape[0]], [0, g.shape[0]]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(g, M, (W, H), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full(g.shape[:2], 255, np.uint8), M, (W, H))
        img[mask > 127] = warped[mask > 127]
        if s.get('hand'):  # mão cobrindo um canto
            x, y = dst[1] + 0.08 * (dst.mean(0) - dst[1])
            cv2.ellipse(img, (int(x), int(y)), (150, 90), 35, 0, 360, (120, 150, 200), -1)
    if s.get('motion'):
        k = np.zeros((s['motion'], s['motion']), np.float32); k[s['motion'] // 2, :] = 1 / s['motion']
        img = cv2.filter2D(img, -1, k)
    if s.get('shade'):
        img = (img * np.linspace(1, s['shade'], W)[None, :, None]).astype(np.uint8)
    f = img.astype(np.float32) * s.get('gain', 1.0) + rng.normal(0, s.get('noise', 1.5), img.shape)
    return np.clip(f, 0, 255).astype(np.uint8)

def make_video(name, orient):
    """port: vídeo 720x1280 numa tela em pé.  land: vídeo 1280x720 numa tela deitada, folha girada 90°
    (o usuário virou o celular).  'mismatch' é sempre vídeo deitado numa tela em pé."""
    s = dict(SCEN[name])
    land = orient == 'land' or name == 'mismatch'
    W, H = (1280, 720) if land else (720, 1280)
    if 'size' in s: W, H = s['size'][::-1] if land else s['size']
    if orient == 'land' and name != 'mismatch': s['rot'] = s.get('rot', 0) + 90
    key = hashlib.md5(repr((sorted(s.items()), W, H)).encode() + open(__file__, 'rb').read()).hexdigest()[:8]
    path = os.path.join(TMP, f'{name}_{orient}_{key}.y4m')
    if not os.path.exists(path):
        os.makedirs(TMP, exist_ok=True)
        with open(path, 'wb') as f:  # y4m cru (I420), 10 qps, 4 quadros em loop
            f.write(f'YUV4MPEG2 W{W} H{H} F10:1 Ip A1:1 C420jpeg\n'.encode())
            for i in range(4):
                f.write(b'FRAME\n' + cv2.cvtColor(frame(s, W, H, rng.normal(0, 0.6, 2)), cv2.COLOR_BGR2YUV_I420).tobytes())
    return path

# ---------- navegador ----------
def serve(page):
    d = tempfile.mkdtemp(); shutil.copy(page, os.path.join(d, 'index.html'))
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    h = functools.partial(Q, directory=d)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{srv.server_address[1]}/index.html#teste'

def cdn_cache():
    d = os.path.join(TMP, 'cdn'); os.makedirs(d, exist_ok=True)
    for url, f in CDN.items():
        p = os.path.join(d, f)
        if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    return d

def run(pw, url, video, vp, secs=6.0, soak=0):
    b = pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=[
        '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        f'--use-file-for-fake-video-capture={video}', '--enable-precise-memory-info'])
    ctx = b.new_context(viewport={'width': 390, 'height': 844} if vp == 'port' else {'width': 844, 'height': 390}, device_scale_factor=2, is_mobile=True, has_touch=True)
    d = cdn_cache()
    for u, f in CDN.items():
        ctx.route(u, (lambda p: lambda r: r.fulfill(path=p, content_type='text/javascript'))(os.path.join(d, f)))
    ctx.route('https://fonts.googleapis.com/**', lambda r: r.fulfill(body='', content_type='text/css'))
    pg = ctx.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(url)
    pg.wait_for_function('!!(window.cv && cv.Mat)', timeout=60000)
    pg.click('#go'); pg.wait_for_selector('#stage:not(.hide)', timeout=15000)
    # wasm: endereço de um bloco de 1 MB recém-alocado (sobe se o OpenCV estiver vazando Mats)
    mem = lambda: pg.evaluate('({js: performance.memory.usedJSHeapSize, wasm: cv._malloc ? (p => (cv._free(p), p))(cv._malloc(1 << 20)) : 0})')
    t0 = time.time(); m0 = None; samples = []; dbgs = []
    while time.time() - t0 < max(secs, soak):
        time.sleep(0.25)
        if m0 is None and time.time() - t0 > 3: m0 = mem()
        samples.append((pg.inner_text('#msg'), pg.eval_on_selector('#shot', 'e => e.classList.contains("ready")')))
        dbg = pg.evaluate('window.__dbg ? (({n, present, lum, clip, sharp, ms, err, edge}) => ({n, present, lum, clip, sharp, ms, err, edge}))(window.__dbg) : null')
        if dbg: dbgs.append(dbg)
    res = pg.evaluate('(v => [v.videoWidth, v.videoHeight])(document.getElementById("v"))')
    m1 = mem(); b.close()
    tail = samples[-8:]
    final = max(set(tail), key=tail.count)
    return dict(msg=final[0], green=final[1], dbg=dbgs[-1] if dbgs else {}, ms=max([d.get('ms') or 0 for d in dbgs] or [0]), stable=tail.count(final) / len(tail), res=res, errs=errs,
                wasm_growth=(m1['wasm'] - m0['wasm']) if m0 else 0, js=m1['js'])

def flow(pw, url):
    """Fluxo completo no cenário 'good': 4 fotos, tamanho das fotos, app em 2º plano e volta, enviar (download)."""
    b = pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=[
        '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
        f'--use-file-for-fake-video-capture={make_video("good", "port")}'])
    ctx = b.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2, is_mobile=True, has_touch=True, accept_downloads=True)
    d = cdn_cache()
    for u, f in CDN.items():
        ctx.route(u, (lambda p: lambda r: r.fulfill(path=p, content_type='text/javascript'))(os.path.join(d, f)))
    pg = ctx.new_page(); errs = []; pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: print('console:', m.text))
    pg.goto(url); pg.wait_for_function('!!(window.cv && cv.Mat)', timeout=60000)
    pg.click('#go'); pg.wait_for_selector('#shot.ready', timeout=15000)
    # app vai para o fundo e volta
    hide = "h => { Object.defineProperty(document, 'hidden', {configurable: true, get: () => h}); document.dispatchEvent(new Event('visibilitychange')) }"
    pg.evaluate(hide, True); time.sleep(.5)
    live = pg.evaluate("!!document.getElementById('v').srcObject")
    pg.evaluate(hide, False); pg.wait_for_selector('#shot.ready', timeout=15000)
    print(f'fundo: câmera solta={not live}, voltou e ficou verde=sim')
    for i in range(4):
        pg.wait_for_selector('#shot:not([disabled])'); t1 = time.time(); pg.click('#shot')
        pg.wait_for_function(f'document.querySelectorAll("#thumbs img").length > {i} || !document.getElementById("done").classList.contains("hide")', timeout=15000)
        print(f'foto {i+1}: {time.time()-t1:.1f}s', pg.inner_text('#msg'))
    pg.wait_for_selector('#done:not(.hide)', timeout=15000)
    dims = pg.evaluate("""Promise.all([...document.querySelectorAll('#grid img')].map(i => i.decode().then(() => i.naturalWidth + 'x' + i.naturalHeight)))""")
    dims = [f"{d} {t}" for d, t in zip(dims, pg.evaluate("""Promise.all([...document.querySelectorAll('#grid img')].map(i => fetch(i.src).then(r => r.blob()).then(b => b.type + ' ' + Math.round(b.size / 1024) + ' kB')))"""))]
    print('fotos:', dims, ' vídeo:', pg.evaluate('(v => v.videoWidth + "x" + v.videoHeight)(document.getElementById("v"))'))
    got = []
    pg.on('download', lambda dl: got.append(dl.suggested_filename))
    pg.click('#share'); time.sleep(4)
    print('enviar (sem Web Share aqui → download):', got, '|', pg.inner_text('#shareNote'), '| erros:', errs or 'nenhum')
    b.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*'); ap.add_argument('-o', '--orient', default='port,land')
    ap.add_argument('--page', default=os.path.join(HERE, '..', 'index.html')); ap.add_argument('--soak', type=int, default=0)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--flow', action='store_true', help='testa tirar 4 fotos, 2º plano e enviar')
    a = ap.parse_args()
    url = serve(a.page); names = a.names or list(SCEN)
    if a.flow:
        with sync_playwright() as pw: flow(pw, url)
        return
    print(f"{'cenário':<16}{'vídeo':<6}{'verde':<7}{'esperado':<22}{'ok?':<5}mensagem")
    bad = 0
    with sync_playwright() as pw:
        for n in names:
            for o in a.orient.split(','):
                if n == 'mismatch' and o == 'land': continue
                r = run(pw, url, make_video(n, o), SCEN[n].get('vp', o), soak=a.soak)
                exp = SCEN[n]['exp']
                good = any(EXPECT[e].lower() in r['msg'].lower() for e in exp.split('|'))
                bad += not good
                extra = f"  [estável {r['stable']:.0%}]" + (f"  ERROS {r['errs']}" if r['errs'] else '')
                if a.verbose: extra += f"  vídeo={r['res'][0]}x{r['res'][1]}  " + ' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in r['dbg'].items())
                if a.soak: extra += f"  wasm +{r['wasm_growth']/1e6:.2f} MB, js {r['js']/1e6:.1f} MB"
                print(f"{n:<16}{o:<6}{'sim' if r['green'] else 'não':<7}{exp:<22}{'✓' if good else '✗':<5}{r['msg']}{extra}", flush=True)
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()

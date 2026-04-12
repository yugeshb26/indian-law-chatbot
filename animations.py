"""
Client-side animation injection for Indian Law Chatbot.

Loads GSAP 3, Three.js, and Anime.js from CDN into the parent Streamlit
document via a 0-height iframe, then sets up:
  • Three.js floating particle field (fixed background)
  • GSAP entrance animations for messages, hero, cards
  • Anime.js ripple micro-interaction on all action buttons
  • CSS scroll-behavior: smooth injected into parent
"""
import streamlit.components.v1 as components

# CDN URLs (pinned versions for stability)
_GSAP_URL  = "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"
_ANIME_URL = "https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"
_THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"

_SCRIPT = """
<script>
(function() {
    'use strict';

    var P  = window.parent;
    var PD = P.document;

    // ── Utility: load a <script> into the PARENT document head ───────────
    function loadInParent(src, id, cb) {
        if (PD.getElementById(id)) { if (cb) cb(); return; }
        var s = PD.createElement('script');
        s.src = src;
        s.id  = id;
        s.onload = cb || function(){};
        PD.head.appendChild(s);
    }

    // ── 1. Ensure programmatic scrolls are always instant ────────────────
    // Do NOT inject scroll-behavior:smooth — it fights the Python-side
    // scroll tracker which needs instant positioning during streaming.
    (function injectScrollCSS() {
        if (PD.getElementById('aether-smooth-scroll')) return;
        var style = PD.createElement('style');
        style.id  = 'aether-smooth-scroll';
        // Anchor-link clicks get smooth scroll; JS scrollTo stays instant
        style.textContent = 'a[href^="#"] { scroll-behavior: smooth; }';
        PD.head.appendChild(style);
    })();

    // ── 2. Three.js particle field ────────────────────────────────────────
    function setupParticles() {
        if (PD.getElementById('aether-canvas')) return;
        var THREE = P.THREE;
        if (!THREE) return;

        var canvas = PD.createElement('canvas');
        canvas.id = 'aether-canvas';
        canvas.style.cssText = [
            'position:fixed', 'top:0', 'left:0',
            'width:100vw', 'height:100vh',
            'pointer-events:none',
            'z-index:0',
            'opacity:0.28',
        ].join(';');
        PD.body.prepend(canvas);

        var W = P.innerWidth, H = P.innerHeight;
        var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: false });
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(P.devicePixelRatio, 1.5));

        var scene  = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
        camera.position.z = 6;

        // Build particle geometry
        var N        = 700;
        var pos      = new Float32Array(N * 3);
        var col      = new Float32Array(N * 3);
        var velY     = new Float32Array(N);   // gentle upward drift

        // Aether Neon palette: gold, cyan, violet
        var palette = [
            [1.0, 0.72, 0.30],   // --gold
            [0.0, 0.90, 1.0 ],   // --cyan
            [0.69, 0.29, 0.93],  // --violet
        ];

        for (var i = 0; i < N; i++) {
            pos[i*3]   = (Math.random() - 0.5) * 22;
            pos[i*3+1] = (Math.random() - 0.5) * 16;
            pos[i*3+2] = (Math.random() - 0.5) * 8;
            velY[i]    = 0.0008 + Math.random() * 0.0012;

            var c = palette[Math.floor(Math.random() * 3)];
            col[i*3]   = c[0];
            col[i*3+1] = c[1];
            col[i*3+2] = c[2];
        }

        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos,  3));
        geo.setAttribute('color',    new THREE.BufferAttribute(col,  3));

        var mat = new THREE.PointsMaterial({
            size: 0.045,
            vertexColors: true,
            transparent: true,
            opacity: 0.85,
            sizeAttenuation: true,
        });

        var points = new THREE.Points(geo, mat);
        scene.add(points);

        // Animate
        var tick = 0;
        var posAttr = geo.attributes.position;
        (function animate() {
            P.requestAnimationFrame(animate);
            tick++;

            // Slow rotation + individual particle drift upward
            points.rotation.y += 0.0002;
            for (var j = 0; j < N; j++) {
                posAttr.array[j*3+1] += velY[j];
                if (posAttr.array[j*3+1] > 8) posAttr.array[j*3+1] = -8;
            }
            posAttr.needsUpdate = true;
            renderer.render(scene, camera);
        })();

        // Resize handler
        P.addEventListener('resize', function() {
            W = P.innerWidth; H = P.innerHeight;
            camera.aspect = W / H;
            camera.updateProjectionMatrix();
            renderer.setSize(W, H);
        });
    }

    // ── 3. GSAP entrance animations + MutationObserver ───────────────────
    function setupGSAP() {
        var gsap = P.gsap;
        if (!gsap) return;

        function animateRow(el) {
            if (!el || el.dataset.animDone) return;
            el.dataset.animDone = '1';
            var isUser = el.classList.contains('user-row');
            gsap.fromTo(el,
                { opacity: 0, x: isUser ? 24 : -24, y: 8, scale: 0.97 },
                { opacity: 1, x: 0, y: 0, scale: 1,
                  duration: 0.42, ease: 'power3.out' }
            );
        }

        function animateHero() {
            var hero = PD.querySelector('.hero-banner:not([data-anim-done])');
            if (!hero) return;
            hero.setAttribute('data-anim-done', '1');
            gsap.fromTo(hero,
                { opacity: 0, y: -18, scale: 0.96 },
                { opacity: 1,  y:   0, scale: 1,
                  duration: 0.65, ease: 'power3.out' }
            );
        }

        function animateDisclaimer() {
            var disc = PD.querySelector('.disclaimer-banner:not([data-anim-done])');
            if (!disc) return;
            disc.setAttribute('data-anim-done', '1');
            gsap.fromTo(disc,
                { opacity: 0, x: -14 },
                { opacity: 1, x: 0, duration: 0.5, delay: 0.12, ease: 'power2.out' }
            );
        }

        function animateCards() {
            var cards = PD.querySelectorAll('[class*="st-key-welcome_"]:not([data-anim-done])');
            if (!cards.length) return;
            cards.forEach(function(c) { c.setAttribute('data-anim-done', '1'); });
            gsap.fromTo(Array.from(cards),
                { opacity: 0, y: 18, scale: 0.93 },
                { opacity: 1, y: 0,  scale: 1,
                  duration: 0.48, stagger: 0.07, ease: 'back.out(1.4)' }
            );
        }

        function animateBars() {
            PD.querySelectorAll('.tricolor-bar:not([data-anim-done])').forEach(function(bar, i) {
                bar.setAttribute('data-anim-done', '1');
                gsap.fromTo(bar,
                    { scaleX: 0, opacity: 0 },
                    { scaleX: 1, opacity: 1, duration: 0.55,
                      delay: 0.2 + i * 0.08, ease: 'power2.out',
                      transformOrigin: 'left center' }
                );
            });
        }

        // Animate what's already in the DOM
        PD.querySelectorAll('.msg-row:not([data-anim-done])').forEach(animateRow);
        animateHero();
        animateDisclaimer();
        animateCards();
        animateBars();

        // Watch for new DOM nodes (new messages added by Streamlit)
        var obs = new P.MutationObserver(function(mutations) {
            mutations.forEach(function(m) {
                m.addedNodes.forEach(function(node) {
                    if (node.nodeType !== 1) return;
                    // Direct .msg-row node
                    if (node.classList && node.classList.contains('msg-row')) {
                        animateRow(node);
                    }
                    // Or rows nested inside a wrapper Streamlit added
                    if (node.querySelectorAll) {
                        node.querySelectorAll('.msg-row:not([data-anim-done])').forEach(animateRow);
                    }
                    // Re-check hero / disclaimer / cards / bars each mutation
                    animateHero();
                    animateDisclaimer();
                    animateCards();
                    animateBars();
                });
            });
        });

        var root = PD.querySelector('[data-testid="stMain"]') || PD.body;
        obs.observe(root, { childList: true, subtree: true });
    }

    // ── 4. Anime.js gold-ripple on action buttons ─────────────────────────
    function setupAnime() {
        var anime = P.anime;
        if (!anime) return;

        function addRipple(btn) {
            if (btn.__aetherRipple) return;
            btn.__aetherRipple = true;
            // Ensure position context for ripple
            var pos = P.getComputedStyle(btn).position;
            if (pos === 'static') btn.style.position = 'relative';
            btn.style.overflow = 'hidden';

            btn.addEventListener('pointerdown', function(e) {
                var rect   = btn.getBoundingClientRect();
                var size   = Math.max(rect.width, rect.height) * 1.6;
                var ripple = PD.createElement('span');
                ripple.style.cssText = [
                    'position:absolute',
                    'border-radius:50%',
                    'background:rgba(255,184,77,0.30)',
                    'pointer-events:none',
                    'width:'  + size + 'px',
                    'height:' + size + 'px',
                    'left:'   + (e.clientX - rect.left  - size/2) + 'px',
                    'top:'    + (e.clientY - rect.top   - size/2) + 'px',
                    'transform:scale(0)',
                    'opacity:0.7',
                ].join(';');
                btn.appendChild(ripple);
                anime({
                    targets:  ripple,
                    scale:    [0, 1],
                    opacity:  [0.7, 0],
                    duration: 560,
                    easing:   'easeOutExpo',
                    complete: function() { ripple.remove(); },
                });
            });
        }

        function wireButtons() {
            PD.querySelectorAll(
                '[data-testid="stMain"] .stButton > button,' +
                'section[data-testid="stSidebar"] .stButton > button'
            ).forEach(addRipple);
        }

        wireButtons();

        // Wire buttons that appear later (sidebar quick topics, regen etc.)
        var obs = new P.MutationObserver(wireButtons);
        obs.observe(PD.body, { childList: true, subtree: true });
    }

    // ── Bootstrap: load libs then initialise ─────────────────────────────
    // Guard against re-running on Streamlit hot-reload
    if (P.__aetherAnimInit) {
        // Libs already loaded — just re-run GSAP for new elements
        if (P.gsap)  setupGSAP();
        if (P.anime) setupAnime();
        return;
    }
    P.__aetherAnimInit = true;

    // Load Three.js first so particles start ASAP
    loadInParent('""" + _THREE_URL + """', 'aether-three', function() {
        setupParticles();
    });

    // GSAP
    loadInParent('""" + _GSAP_URL + """', 'aether-gsap', function() {
        setupGSAP();
    });

    // Anime.js
    loadInParent('""" + _ANIME_URL + """', 'aether-anime', function() {
        setupAnime();
    });

})();
</script>
"""


def inject_animations() -> None:
    """Inject GSAP, Three.js, and Anime.js into the parent Streamlit page.

    Call this once per render pass (Streamlit re-runs are idempotent —
    the guard flag ``window.__aetherAnimInit`` prevents double-init while
    still re-running GSAP on each rerun so new messages get animated).
    """
    components.html(_SCRIPT, height=0)

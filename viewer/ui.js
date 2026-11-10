// Page glue for the 3D viewers. Nothing heavy loads until someone clicks "Load the 3D scene".
(function () {
  var mod = null;
  function loadModule() { return mod || (mod = import('./viewer.js')); }

  document.querySelectorAll('[data-v3d]').forEach(function (el) {
    var scene = el.dataset.v3d;
    var box = el.querySelector('.v3d-box'), start = el.querySelector('.v3d-start'), poster = el.querySelector('.v3d-poster');
    var load = el.querySelector('.v3d-load'), bar = load.querySelector('b'), txt = load.querySelector('span');
    var tag = el.querySelector('.v3d-tag'), err = el.querySelector('.v3d-err');
    var verBtns = el.querySelectorAll('button[data-ver]');
    var viewer = null, current = null, busy = false;

    function mark(ver) { verBtns.forEach(function (b) { b.classList.toggle('on', b.dataset.ver === ver); }); }
    function progress(f, t) {
      load.hidden = f >= 1;
      bar.style.width = Math.round(f * 100) + '%';
      txt.textContent = t || '';
    }
    function show(ver) {
      if (busy || ver === current) return;
      busy = true; err.hidden = true; mark(ver);
      start.hidden = true;
      progress(0, 'Loading…');
      loadModule().then(function (m) {
        if (!viewer) {
          viewer = new m.SceneViewer(box, {
            onProgress: progress,
            onReady: function (key, meta) {
              busy = false; current = key.slice(-2);
              poster.style.opacity = 0;
              tag.hidden = false;
              tag.className = 'v3d-tag' + (current === 'v1' ? ' one' : '');
              tag.textContent = (current === 'v1' ? 'v1 · quick pass' : 'v2 · max effort') + ' · ' + meta.stats.triangles.toLocaleString('en-US') + ' triangles';
              mark(current);
              el.dataset.ready = key;
            },
            onError: function (e) {
              busy = false; load.hidden = true; mark(current);
              err.hidden = false;
              err.textContent = 'Could not load the 3D scene (' + (e && e.message ? e.message : 'unknown error') + '). The renders above and the .blend downloads below still work.';
            },
          });
          el.viewer = viewer;
        }
        return viewer.show(scene + '_' + ver);
      }).catch(function (e) {
        busy = false; load.hidden = true; err.hidden = false;
        err.textContent = 'Your browser could not start the 3D viewer (' + (e && e.message ? e.message : 'no WebGL') + ').';
      });
    }

    start.querySelector('button').addEventListener('click', function () { show('v2'); });
    verBtns.forEach(function (b) { b.addEventListener('click', function () { show(b.dataset.ver); }); });
    el.querySelector('[data-act=reset]').addEventListener('click', function () { if (viewer) viewer.resetView(); });
    el.querySelector('[data-act=full]').addEventListener('click', function () {
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      var fn = box.requestFullscreen || box.webkitRequestFullscreen;
      if (fn) fn.call(box);
    });
  });
})();

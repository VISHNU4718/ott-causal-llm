(function () {
  'use strict';
  const L = window.demoLib;
  const COLORS = { dormant: '#d97757', casual: '#d9a441', core: '#4f8f87' };
  const $ = (id) => document.getElementById(id);
  const pct = (x) => (x * 100).toFixed(1) + '%';
  const f2 = (x) => x.toFixed(2);
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function h(tag, props, ...kids) {
    const e = document.createElement(tag);
    Object.assign(e, props || {});
    kids.forEach((k) => e.append(k));
    return e;
  }

  let data;
  fetch('data.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((d) => { data = d; init(); })
    .catch((e) => {
      document.querySelector('main').prepend(h('p', { className: 'msg', role: 'alert', textContent: 'Could not load data.json (' + e.message + ').' }));
    });

  function init() {
    cards(); gainSection(); deciles(); segments(); map(); users(); limits();
  }

  function cards() {
    const m = data.meta, x = m.metrics['X-Learner'], t = m.metrics['T-Learner'];
    const items = [
      [m.n_users.toLocaleString(), 'simulated users'],
      [m.n_shows.toLocaleString(), 'shows'],
      [m.n_heldout.toLocaleString(), 'held-out users shown below'],
      [pct(m.treated_share), 'randomly assigned the intervention'],
      [pct(m.retention_rate), '30-day retention overall'],
      [f2(x.correlation), 'X-Learner correlation with true effect'],
      [f2(t.correlation), 'T-Learner correlation with true effect'],
      [pct(m.training.hit_rate_at_10), 'transformer hit-rate@10, next show (random: ' + pct(m.training.random_hit_rate_at_10) + ')'],
    ];
    const box = $('cards');
    items.forEach(([n, l]) => box.append(h('div', { className: 'card' }, h('div', { className: 'n', textContent: n }), h('div', { className: 'l', textContent: l }))));
  }

  // ---- targeting ----
  const KEYS = [['uplift_x', 'X-Learner', '#2b5fd9'], ['uplift_t', 'T-Learner', '#7a5bd6'], ['true_effect', 'Oracle (true effect)', '#4f8f87']];
  function gainSection() {
    const total = data.users.length;
    const slider = $('share');
    function render() {
      const share = Number(slider.value);
      $('share-out').textContent = share + '%';
      const n = Math.round((total * share) / 100);
      const rows = KEYS.map(([k, name]) => [name, L.expectedGain(data.users, k, n)]);
      rows.splice(2, 0, ['Random contact', L.randomGain(data.users, n)]);
      const t = $('gain-table');
      t.replaceChildren(
        h('thead', {}, h('tr', {}, h('th', { textContent: 'Ranking' }), h('th', { className: 'num', textContent: 'Extra retained (of ' + n + ' contacted)' }), h('th', { className: 'num', textContent: 'Per 100 contacted' }))),
        h('tbody', {}, ...rows.map(([name, g]) => h('tr', {}, h('td', { textContent: name }), h('td', { className: 'num', textContent: g.toFixed(1) }), h('td', { className: 'num', textContent: n ? ((g / n) * 100).toFixed(1) : '-' }))))
      );
      drawGain(n);
    }
    slider.addEventListener('input', render);
    render();
  }
  function drawGain(n) {
    const svg = $('gain-chart');
    svg.replaceChildren();
    const W = 520, H = 300, m = { l: 44, r: 12, t: 12, b: 34 };
    const total = data.users.length;
    const curves = KEYS.map(([k, name, c]) => ({ name, c, pts: L.gainCurve(data.users, k) }));
    const mean = data.users.reduce((s, u) => s + u.true_effect, 0) / total;
    const ymax = Math.max(...curves.map((c) => c.pts[c.pts.length - 1][1]), mean * total) * 1.08 || 1;
    const X = (v) => m.l + (v / total) * (W - m.l - m.r);
    const Y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    for (let i = 0; i <= 4; i++) {
      const v = (ymax * i) / 4;
      svg.append(el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: '#e6e9ee' }), el('text', { x: m.l - 6, y: Y(v) + 3, 'text-anchor': 'end' }, Math.round(v)));
    }
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => svg.append(el('text', { x: X(f * total), y: H - 16, 'text-anchor': 'middle' }, Math.round(f * total))));
    svg.append(el('text', { x: (W + m.l) / 2, y: H - 3, 'text-anchor': 'middle' }, 'users contacted'));
    svg.append(el('line', { x1: X(0), y1: Y(0), x2: X(total), y2: Y(mean * total), stroke: '#999', 'stroke-dasharray': '4 3' }));
    curves.forEach((c) => svg.append(el('polyline', { points: c.pts.map(([a, b]) => X(a) + ',' + Y(b)).join(' '), fill: 'none', stroke: c.c, 'stroke-width': 2 })));
    svg.append(el('line', { x1: X(n), x2: X(n), y1: m.t, y2: H - m.b, stroke: '#1c2330', 'stroke-width': 1 }));
    const legend = curves.map((c) => [c.name, c.c]).concat([['Random', '#999']]);
    legend.forEach(([name, c], i) => {
      svg.append(el('rect', { x: m.l + 10, y: m.t + 6 + i * 15, width: 10, height: 3, fill: c }), el('text', { x: m.l + 25, y: m.t + 11 + i * 15 }, name));
    });
  }

  // ---- deciles ----
  function deciles() {
    const svg = $('decile-chart');
    const names = Object.keys(data.meta.metrics);
    const colors = ['#7a5bd6', '#2b5fd9', '#9aa5b5'];
    const order = ['T-Learner', 'X-Learner', 'X-Learner on hand-built features'];
    const W = 760, H = 300, m = { l: 44, r: 10, t: 12, b: 40 };
    const ymax = Math.max(...names.flatMap((n) => data.meta.metrics[n].deciles)) * 1.1;
    const Y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    for (let i = 0; i <= 4; i++) {
      const v = (ymax * i) / 4;
      svg.append(el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: '#e6e9ee' }), el('text', { x: m.l - 6, y: Y(v) + 3, 'text-anchor': 'end' }, v.toFixed(2)));
    }
    const gw = (W - m.l - m.r) / 10, bw = (gw - 10) / order.length;
    order.forEach((n, j) => {
      data.meta.metrics[n].deciles.forEach((v, i) => {
        svg.append(el('rect', { x: m.l + i * gw + 5 + j * bw, y: Y(Math.max(v, 0)), width: bw - 1, height: Math.abs(Y(v) - Y(0)), fill: colors[j] }));
      });
    });
    for (let i = 0; i < 10; i++) svg.append(el('text', { x: m.l + i * gw + gw / 2, y: H - 24, 'text-anchor': 'middle' }, i + 1));
    svg.append(el('text', { x: (W + m.l) / 2, y: H - 10, 'text-anchor': 'middle' }, 'predicted-uplift decile (1 = highest predicted uplift)'));
    order.forEach((n, j) => {
      svg.append(el('rect', { x: W - 270, y: m.t + 4 + j * 15, width: 10, height: 10, fill: colors[j] }), el('text', { x: W - 255, y: m.t + 13 + j * 15 }, n));
    });
    const t = $('metric-table');
    t.replaceChildren(
      h('thead', {}, h('tr', {}, h('th', { textContent: 'Model' }), h('th', { className: 'num', textContent: 'Correlation with true effect' }), h('th', { className: 'num', textContent: 'Top decile true effect' }), h('th', { className: 'num', textContent: 'Bottom decile true effect' }))),
      h('tbody', {}, ...order.map((n) => { const r = data.meta.metrics[n]; return h('tr', {}, h('td', { textContent: n }), h('td', { className: 'num', textContent: r.correlation.toFixed(2) }), h('td', { className: 'num', textContent: r.deciles[0].toFixed(3) }), h('td', { className: 'num', textContent: r.deciles[9].toFixed(3) })); }))
    );
  }

  function segments() {
    const m = data.meta;
    const segs = ['dormant', 'casual', 'core'];
    $('seg-table').replaceChildren(
      h('thead', {}, h('tr', {}, ...['Segment', 'Share of users', '30-day retention without intervention', 'True effect of intervention'].map((x, i) => h('th', { className: i ? 'num' : '', textContent: x })))),
      h('tbody', {}, ...segs.map((s) => h('tr', {}, h('td', { textContent: s }), h('td', { className: 'num', textContent: pct(m.segment_share[s]) }), h('td', { className: 'num', textContent: pct(m.segment_control_retention[s]) }), h('td', { className: 'num', textContent: (m.segment_true_effect[s] > 0 ? '+' : '') + m.segment_true_effect[s].toFixed(2) }))))
    );
  }

  // ---- map ----
  function map() {
    const cv = $('map'), ctx = cv.getContext('2d'), tip = $('map-tip');
    const pts = data.tsne;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 16;
    const sx = (v) => pad + ((v - x0) / (x1 - x0)) * (cv.width - 2 * pad);
    const sy = (v) => pad + ((v - y0) / (y1 - y0)) * (cv.height - 2 * pad);
    const colorOf = (seg) => COLORS[seg];
    pts.forEach((p) => { p.px = sx(p[0]); p.py = sy(p[1]); });
    ctx.globalAlpha = 0.65;
    pts.forEach((p) => {
      ctx.fillStyle = colorOf(p[2]);
      ctx.beginPath(); ctx.arc(p.px, p.py, 2.6, 0, 6.3); ctx.fill();
    });
    cv.addEventListener('mousemove', (ev) => {
      const r = cv.getBoundingClientRect(), k = cv.width / r.width;
      const mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
      let best = null, bd = 64;
      for (const p of pts) { const d = (p.px - mx) ** 2 + (p.py - my) ** 2; if (d < bd) { bd = d; best = p; } }
      if (!best) { tip.hidden = true; return; }
      tip.hidden = false;
      tip.textContent = 'user ' + best[4] + ' · ' + best[2] + (best[3] ? ' · held out' : '');
      tip.style.left = (best.px / k + 10) + 'px'; tip.style.top = (best.py / k - 8) + 'px';
    });
    cv.addEventListener('mouseleave', () => { tip.hidden = true; });
    $('legend').replaceChildren(...['dormant', 'casual', 'core'].map((s) => h('span', {}, h('span', { className: 'dot', style: 'background:' + COLORS[s] }), s)));
  }

  // ---- users ----
  const PAGE = 15;
  let page = 0, filtered = [];
  function users() {
    const sel = $('seg-filter');
    function apply() {
      filtered = sel.value ? data.users.filter((u) => u.segment === sel.value) : data.users;
      page = 0; renderUsers();
    }
    sel.addEventListener('change', apply);
    $('prev').addEventListener('click', () => { page = Math.max(0, page - 1); renderUsers(); });
    $('next').addEventListener('click', () => { page = Math.min(Math.ceil(filtered.length / PAGE) - 1, page + 1); renderUsers(); });
    $('lookup').addEventListener('submit', (ev) => {
      ev.preventDefault();
      const r = L.lookupUser(data.users, $('uid').value);
      $('lookup-msg').textContent = r.ok ? '' : r.error;
      if (r.ok) showDetail(r.user);
    });
    apply();
  }
  function renderUsers() {
    const rows = filtered.slice(page * PAGE, page * PAGE + PAGE);
    const cols = ['User', 'Segment (hidden)', 'Items watched', 'X-Learner uplift', 'T-Learner uplift', 'True effect'];
    $('user-table').replaceChildren(
      h('thead', {}, h('tr', {}, ...cols.map((c, i) => h('th', { className: i > 1 ? 'num' : '', textContent: c })))),
      h('tbody', {}, ...rows.map((u) => {
        const tr = h('tr', { tabIndex: 0 }, h('td', { textContent: u.id }), h('td', { textContent: u.segment }), h('td', { className: 'num', textContent: u.n_watched }), h('td', { className: 'num', textContent: u.uplift_x.toFixed(3) }), h('td', { className: 'num', textContent: u.uplift_t.toFixed(3) }), h('td', { className: 'num', textContent: u.true_effect.toFixed(2) }));
        tr.addEventListener('click', () => showDetail(u));
        tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') showDetail(u); });
        return tr;
      }))
    );
    const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
    $('page-info').textContent = 'Page ' + (page + 1) + ' of ' + pages + ' (' + filtered.length + ' users)';
    $('prev').disabled = page === 0; $('next').disabled = page >= pages - 1;
  }
  function showDetail(u) {
    const d = $('detail');
    d.hidden = false;
    const rank = L.rankBy(data.users, 'uplift_x').findIndex((x) => x.id === u.id) + 1;
    d.replaceChildren(
      h('h3', { textContent: 'User ' + u.id }),
      h('p', { textContent: 'Hidden segment: ' + u.segment + '. Ranked ' + rank + ' of ' + data.users.length + ' by X-Learner uplift (' + u.uplift_x.toFixed(3) + '); T-Learner says ' + u.uplift_t.toFixed(3) + '; the simulation\'s true effect is ' + u.true_effect.toFixed(2) + '. In the randomised trial this user was ' + (u.treated ? 'contacted' : 'not contacted') + ' and ' + (u.retained ? 'stayed' : 'did not stay') + ' after 30 days.' }),
      h('div', { textContent: 'Most recent ' + u.recent.length + ' of ' + u.n_watched + ' shows watched (show id, main genre):' }),
      h('div', { className: 'chips' }, ...u.recent.map((id) => h('span', { className: 'chip', textContent: '#' + id + ' · g' + data.shows[id].genre })))
    );
    d.scrollIntoView({ block: 'nearest' });
  }

  function limits() {
    const m = data.meta.metrics, t = data.meta.training;
    const a = m['X-Learner'].correlation, b = m['X-Learner on hand-built features'].correlation;
    $('limit-model').textContent = 'The transformer predicts the next show about as well as chance on this data (hit-rate@10 ' + pct(t.hit_rate_at_10) + ' against ' + pct(t.random_hit_rate_at_10) + ' for random and ' + pct(t.popularity_hit_rate_at_10) + ' for most-popular). Its embeddings still carry useful signal about the history, but hand-built features (length, genre mix, genre concentration) reach a correlation of ' + b.toFixed(2) + ' against ' + a.toFixed(2) + ' for the embeddings with the same X-Learner.';
  }
})();

/* Shared runtime for the Hand Study research console. */
(() => {
  const NAV = [
    ['/capture', 'Capture'],
    ['/dashboard', 'Study dashboard'],
    ['/subjects', 'Subjects'],
    ['/review', 'Review'],
    ['/admin', 'Administration'],
    ['/devices', 'Devices'],
    ['/compare', 'Compare'],
    ['/method', 'Method']
  ];

  const esc = (value) => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const numeric = value => value !== null && value !== '' && value !== undefined && Number.isFinite(Number(value));
  const n = (value, digits = 0) => numeric(value) ? Number(value).toFixed(digits) : '—';
  const ratio = value => numeric(value) ? Number(value).toFixed(4) : '—';
  const byId = id => document.getElementById(id);
  const currentPath = () => location.pathname.replace(/\/$/, '') || '/';

  function persistPreference(key, value) {
    try { localStorage.setItem(key, value); } catch (_) {}
  }
  function preference(key, fallback) {
    try { return localStorage.getItem(key) || fallback; } catch (_) { return fallback; }
  }
  function initialiseDocument() {
    const theme = preference('hs-theme', 'system');
    const rtl = preference('hs-direction', 'ltr');
    applyTheme(theme); applyDirection(rtl);
  }
  function applyTheme(theme) {
    const root = document.documentElement;
    const chosen = theme === 'system' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme;
    root.dataset.theme = chosen;
    persistPreference('hs-theme', theme);
    const button = byId('theme-toggle');
    if (button) button.textContent = chosen === 'dark' ? 'Light' : 'Dark';
  }
  function applyDirection(direction) {
    const root = document.documentElement;
    root.dir = direction;
    root.dataset.direction = direction;
    persistPreference('hs-direction', direction);
    const button = byId('direction-toggle');
    if (button) button.textContent = direction === 'rtl' ? 'LTR' : 'العربية';
  }
  function toolbar() {
    return `<button class="btn btn-quiet" type="button" id="direction-toggle" aria-label="Switch text direction">العربية</button>
      <button class="btn btn-quiet" type="button" id="theme-toggle" aria-label="Switch colour theme">Dark</button>`;
  }
  function bindPreferences() {
    const theme = byId('theme-toggle');
    const direction = byId('direction-toggle');
    if (theme) theme.addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
    if (direction) direction.addEventListener('click', () => applyDirection(document.documentElement.dir === 'rtl' ? 'ltr' : 'rtl'));
    applyTheme(preference('hs-theme', 'system'));
    applyDirection(preference('hs-direction', 'ltr'));
  }
  function shell({active, eyebrow = 'Hand Study', title, subtitle = '', actions = '', content = ''}) {
    const path = active || currentPath();
    const nav = NAV.map(([href, label]) => `<a class="nav-link" href="${href}" ${href === path ? 'aria-current="page"' : ''}><span>${label}</span></a>`).join('');
    return `<div class="console-shell"><aside class="sidebar"><a href="/dashboard" class="brand"><span class="brand-mark">2:4</span><span>Hand Study<small>Measurement console</small></span></a><p class="nav-label">Console</p><nav class="nav-list" aria-label="Primary navigation">${nav}</nav><div class="sidebar-foot"><p class="privacy-note">Research measurement console. A ratio is shown with its uncertainty and quality verdict.</p></div></aside><main class="console-main"><header class="topbar"><div class="page-heading"><p class="eyebrow">${esc(eyebrow)}</p><h1>${esc(title)}</h1>${subtitle ? `<p class="card-note">${esc(subtitle)}</p>` : ''}</div><div class="top-actions">${actions}${toolbar()}</div></header><div class="page-content">${content}</div></main></div>`;
  }
  function quality(qualityValue) {
    const q = qualityValue || {level: 'unknown', reasons: []};
    const level = ['ok','poor','fail'].includes(q.level) ? q.level : 'unknown';
    const labels = {ok:'Quality: acceptable', poor:'Quality: review advised', fail:'Quality: not reliable', unknown:'Quality: unavailable'};
    return `<span class="quality ${level}">${labels[level]}</span>`;
  }
  function reading(data, options = {}) {
    const q = data?.quality || {level:'unknown', reasons:[]};
    const fail = q.level === 'fail' || !numeric(data?.median_ratio);
    const level = ['ok','poor','fail'].includes(q.level) ? q.level : 'unknown';
    const primary = fail ? `<div class="reading-masked">Not reported</div>` : `<div class="reading-value">${ratio(data?.median_ratio)}</div>`;
    const provenance = data?.ratio_from === 'sharp'
      ? `Measured from ${n(data?.n_sharp)} sharp frames of ${n(data?.n_frames)}.`
      : data?.ratio_from === 'all' ? `Measured from all ${n(data?.n_frames)} frames.` : 'Measurement provenance unavailable.';
    const reasons = Array.isArray(q.reasons) && q.reasons.length ? `<ul class="reason-list">${q.reasons.map(item => `<li>${esc(item)}</li>`).join('')}</ul>` : '';
    return `<section class="reading card ${level}" aria-label="Measurement result"><div><div class="reading-label">${esc(options.label || '2D:4D measurement')}</div>${primary}${fail ? `<p class="card-note">The quality gate prevents this value from being presented as a result.</p>` : ''}</div><div class="reading-meta"><div>${quality(q)}</div><div class="reading-sd">SD ${n(data?.sd_ratio, 4)}</div><div class="reading-provenance">${esc(provenance)}</div>${reasons}</div></section>`;
  }
  function state(kind, title, detail, retryLabel = 'Try again') {
    const retry = kind === 'error' ? `<button class="btn" type="button" data-retry>${esc(retryLabel)}</button>` : '';
    return `<div class="state ${kind === 'error' ? 'state-error' : ''}"><div><strong>${esc(title)}</strong><span>${esc(detail || '')}</span>${retry ? `<p style="margin:14px 0 0">${retry}</p>` : ''}</div></div>`;
  }
  function loading(lines = 4, label = 'Loading current study data') { return `<div class="state state-loading" role="status" aria-live="polite"><div><strong>${esc(label)}</strong><span>Waiting for the server response.</span><div class="loading-lines">${Array.from({length:lines}, (_,i)=>`<div class="skeleton" style="width:${90-i*9}%">Loading</div>`).join('')}</div></div></div>`; }
  async function api(url, options = {}) {
    let response;
    try { response = await fetch(url, {cache:'no-store', ...options}); }
    catch (_) { throw new Error('Network error. Check the connection and try again.'); }
    let payload = null;
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok || payload?.error) throw new Error(payload?.error || `Request failed (${response.status})`);
    return payload;
  }
  function iso(value) {
    if (!value) return '—';
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? String(value) : new Intl.DateTimeFormat(undefined, {dateStyle:'medium', timeStyle:'short'}).format(date);
  }
  function sample(values, condition) { return (values || []).filter(condition); }
  function median(values) {
    const nums = values.map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
    if (!nums.length) return null;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  }
  function histogram(values, {min = .88, max = 1.08, bins = 12, label = '2D:4D ratio', refs = true} = {}) {
    const nums = values.map(Number).filter(Number.isFinite);
    if (!nums.length) return '<div class="empty-chart">No measurable readings are available for this view.</div>';
    const counts = Array(bins).fill(0);
    nums.forEach(value => { const index = Math.max(0, Math.min(bins - 1, Math.floor((value-min)/(max-min)*bins))); counts[index]++; });
    const maximum = Math.max(...counts, 1), W = 680, H = 250, L = 42, R = 12, T = 16, B = 31, width = W-L-R, height = H-T-B;
    const x = value => L + (value-min)/(max-min)*width;
    const y = value => T + height - value/maximum*height;
    const bars = counts.map((count,index) => { const bx = L + index*width/bins + 2, bw = width/bins-4; return `<rect class="chart-bar" x="${bx.toFixed(1)}" y="${y(count).toFixed(1)}" width="${bw.toFixed(1)}" height="${(T+height-y(count)).toFixed(1)}" rx="2"/>`; }).join('');
    const band = (centre, spread, cls) => `<rect class="${cls}" x="${x(centre-spread).toFixed(1)}" y="${T}" width="${Math.max(2,x(centre+spread)-x(centre-spread)).toFixed(1)}" height="${height}"/>`;
    const ticks = [min, .92, .96, 1, 1.04, max].filter((v,i,a)=>v>=min&&v<=max&&a.indexOf(v)===i).map(v=>`<text class="chart-label" x="${x(v)}" y="${H-10}" text-anchor="middle">${v.toFixed(2)}</text>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of ${esc(label)} with ${nums.length} readings"><line class="chart-axis" x1="${L}" y1="${T+height}" x2="${W-R}" y2="${T+height}"/>${refs ? band(.964,.030,'chart-ref-male')+band(.975,.028,'chart-ref-female') : ''}${bars}${ticks}<text class="chart-label" x="${L}" y="${T+10}">count</text></svg>${refs ? `<div class="chart-legend"><span><i class="legend-key male"></i> Male reference: 0.964 ± 0.030</span><span><i class="legend-key female"></i> Female reference: 0.975 ± 0.028</span></div>` : ''}</div>`;
  }
  function lineChart(values, {label = 'Captures', threshold = null, thresholdClass = 'chart-line-repeat'} = {}) {
    const rows = values.filter(row => Number.isFinite(Number(row.value)));
    if (!rows.length) return '<div class="empty-chart">No dated readings are available for this view.</div>';
    const W=680,H=220,L=39,R=12,T=16,B=31,w=W-L-R,h=H-T-B;
    const nums=rows.map(row=>Number(row.value)), low=Math.min(...nums, threshold ?? Infinity), high=Math.max(...nums, threshold ?? -Infinity), spread=Math.max(.004,high-low);
    const y=value=>T+h-(value-(low-spread*.15))/(spread*1.3)*h;
    const x=index=>L+(rows.length===1?w/2:index*w/(rows.length-1));
    const points=rows.map((row,index)=>`${x(index).toFixed(1)},${y(Number(row.value)).toFixed(1)}`).join(' ');
    const thresholdLine=Number.isFinite(threshold)?`<line class="${thresholdClass}" x1="${L}" x2="${W-R}" y1="${y(threshold)}" y2="${y(threshold)}"/><text class="chart-label" x="${W-R}" y="${y(threshold)-4}" text-anchor="end">${threshold.toFixed(3)}</text>`:'';
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><line class="chart-axis" x1="${L}" y1="${T+h}" x2="${W-R}" y2="${T+h}"/>${thresholdLine}<polyline points="${points}" fill="none" stroke="var(--teal)" stroke-width="2.5"/>${rows.map((row,index)=>`<circle cx="${x(index)}" cy="${y(Number(row.value))}" r="3.5" fill="var(--teal)"><title>${esc(row.label || '')}: ${Number(row.value).toFixed(4)}</title></circle>`).join('')}<text class="chart-label" x="${L}" y="${T+10}">${low.toFixed(3)}–${high.toFixed(3)}</text></svg></div>`;
  }
  function confirmDialog({title, detail, consequence, confirmLabel = 'Confirm', destructive = true, onConfirm}) {
    const element = document.createElement('div');
    element.className = 'dialog-backdrop';
    element.innerHTML = `<section class="dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><h2 id="confirm-title">${esc(title)}</h2><p>${esc(detail)}</p><div class="notice"><strong>Consequence:</strong> ${esc(consequence)}</div><div class="dialog-actions"><button class="btn" type="button" data-cancel>Cancel</button><button class="btn ${destructive ? 'btn-danger' : 'btn-primary'}" type="button" data-confirm>${esc(confirmLabel)}</button></div></section>`;
    document.body.append(element);
    const close = () => element.remove();
    element.querySelector('[data-cancel]').onclick = close;
    element.addEventListener('click', event => { if (event.target === element) close(); });
    element.querySelector('[data-confirm]').onclick = async () => { const button = element.querySelector('[data-confirm]'); button.disabled = true; try { await onConfirm?.(); close(); } catch (error) { button.disabled = false; element.querySelector('.notice').innerHTML = `<strong>Could not complete:</strong> ${esc(error.message)}`; } };
    element.querySelector('[data-cancel]').focus();
  }
  function canvasReview(canvas, imageUrl, frame, onChange) {
    const ctx = canvas.getContext('2d');
    let image = new Image(), dragging = -1, scale = 1, offset = [0,0], points = (frame.points || []).map(p => [...p]);
    const model = (frame.model_points || []).map(p => [...p]);
    const colours = ['#f09648','#f7c948','#38aa85','#4e86dc'];
    function fit() { const rect=canvas.parentElement.getBoundingClientRect(); canvas.width=Math.max(320,Math.floor(rect.width-32)); canvas.height=Math.max(280,Math.floor(rect.height-24)); if(!image.width) return; scale=Math.min(canvas.width/image.width,canvas.height/image.height); offset=[(canvas.width-image.width*scale)/2,(canvas.height-image.height*scale)/2]; draw(); }
    function pos(point) { return [point[0]*scale+offset[0],point[1]*scale+offset[1]]; }
    function imagePoint(event) { const rect=canvas.getBoundingClientRect(); return [(event.clientX-rect.left-offset[0])/scale,(event.clientY-rect.top-offset[1])/scale]; }
    function drawLine(a,b,col,dashed=false) { const pa=pos(a),pb=pos(b);ctx.save();ctx.strokeStyle=col;ctx.lineWidth=2; if(dashed)ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(...pa);ctx.lineTo(...pb);ctx.stroke();ctx.restore(); }
    function draw() { ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#0c1519';ctx.fillRect(0,0,canvas.width,canvas.height);if(!image.width)return;ctx.drawImage(image,offset[0],offset[1],image.width*scale,image.height*scale);if(model.length===4){drawLine(model[0],model[1],'#ffffff',true);drawLine(model[2],model[3],'#ffffff',true);model.forEach(point=>{const [x,y]=pos(point);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();});}if(points.length===4){drawLine(points[0],points[1],colours[1]);drawLine(points[2],points[3],colours[2]);points.forEach((point,index)=>{const [x,y]=pos(point);ctx.fillStyle=colours[index];ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();ctx.stroke();});} }
    image.onload=fit; image.src=imageUrl;
    canvas.onpointerdown=event=>{ const [x,y]=imagePoint(event);let d=Infinity;points.forEach((point,index)=>{const distance=Math.hypot(point[0]-x,point[1]-y);if(distance<d){d=distance;dragging=index;}});if(d>25)dragging=-1;canvas.setPointerCapture(event.pointerId); };
    canvas.onpointermove=event=>{if(dragging<0)return;points[dragging]=imagePoint(event);draw();onChange?.(points);};
    canvas.onpointerup=()=>{dragging=-1;};
    window.addEventListener('resize',fit,{once:true});
    return {get points(){return points;}, reset(){points=(frame.model_points||[]).map(p=>[...p]);draw();onChange?.(points);}, redraw:fit};
  }
  initialiseDocument();
  window.HandConsole = {esc,n,ratio,numeric,byId,shell,quality,reading,state,loading,api,iso,sample,median,histogram,lineChart,confirmDialog,canvasReview,bindPreferences};
})();

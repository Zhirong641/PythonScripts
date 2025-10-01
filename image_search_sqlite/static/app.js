// Same client-side autocomplete as before
(function(){
  const input = document.getElementById('q');
  const box = document.getElementById('suggestions');

  function getLastToken() {
    const raw = input.value;
    const parts = raw.replace(/,/g, ' ').split(/\s+/);
    let idx = parts.length - 1;
    while (idx >= 0 && parts[idx] === '') idx--;
    const token = idx >= 0 ? parts[idx] : '';
    const before = parts.slice(0, idx).filter(Boolean);
    return {before, token};
  }

  function renderSuggestions(items) {
    box.innerHTML = '';
    if (!items || !items.length) { box.classList.add('hidden'); return; }
    items.forEach(it => {
      const div = document.createElement('div');
      div.className = 'suggestion';
      const left = document.createElement('div');
      left.className = 'left';
      left.textContent = it.tag;
      const right = document.createElement('div');
      right.className = 'right';
      right.textContent = (it.ja ? it.ja + ' · ' : '') + (it.freq || 0);
      div.appendChild(left);
      div.appendChild(right);
      div.addEventListener('click', () => {
        const {before, token} = getLastToken();
        let score = '';
        if (token.includes(':')) {
          const parts = token.split(':');
          score = ':' + (parts[1] || '');
        }
        const newVal = [...before, it.tag + score].join(' ');
        input.value = newVal + ' ';
        box.classList.add('hidden');
        input.focus();
      });
      box.appendChild(div);
    });
    box.classList.remove('hidden');
  }

  let lastQ = '';
  let t = null;
  input.addEventListener('input', () => {
    const {token} = getLastToken();
    const probe = (token.includes(':') ? token.split(':')[0] : token).trim().toLowerCase();
    if (!probe) { box.classList.add('hidden'); return; }
    if (probe === lastQ) return;
    lastQ = probe;
    if (t) clearTimeout(t);
    t = setTimeout(async () => {
      try {
        const res = await fetch(SUGGEST_API + '?q=' + encodeURIComponent(probe) + '&limit=30');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const items = await res.json();
        renderSuggestions(items);
      } catch (e) { console.error(e); }
    }, 120);
  });

  document.addEventListener('click', (ev) => {
    if (!box.contains(ev.target) && ev.target !== input) {
      box.classList.add('hidden');
    }
  });
})();

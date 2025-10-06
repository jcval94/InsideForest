document.addEventListener('DOMContentLoaded', function() {
  const LANG_KEY = 'docs-lang';
  const defaultLang = 'EN';
  const currentLang = localStorage.getItem(LANG_KEY) || defaultLang;

  function createSelector() {
    const select = document.createElement('select');
    select.innerHTML = '<option value="EN">EN</option><option value="ES">ES</option>';
    select.value = currentLang;
    select.addEventListener('change', () => {
      const lang = select.value;
      localStorage.setItem(LANG_KEY, lang);
      const base = window.location.pathname.replace(/_es\.html$/, '.html');
      const target = lang === 'ES' ? base.replace(/\.html$/, '_es.html') : base;
      if (window.location.pathname !== target) {
        window.location.pathname = target;
      } else {
        filterNav(lang);
      }
    });
    return select;
  }

  function filterNav(lang) {
    document.querySelectorAll('nav a').forEach(a => {
      const href = a.getAttribute('href');
      const isSpanish = /_es\.html$/.test(href);
      const li = a.closest('li');
      if (!li) return;
      if ((lang === 'EN' && isSpanish) || (lang === 'ES' && !isSpanish)) {
        li.style.display = 'none';
      } else {
        li.style.display = '';
      }
    });
  }

  const container = document.querySelector('.lang-switch');
  if (container) {
    container.innerHTML = '';
    container.appendChild(createSelector());
  }
  filterNav(currentLang);

  document.querySelectorAll('pre > code').forEach(codeBlock => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.textContent = 'Copy';
    const pre = codeBlock.parentNode;
    pre.style.position = 'relative';
    pre.appendChild(button);
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeBlock.innerText);
        button.textContent = 'Copied!';
        setTimeout(() => { button.textContent = 'Copy'; }, 2000);
      } catch (err) {
        button.textContent = 'Error';
      }
    });
  });
});

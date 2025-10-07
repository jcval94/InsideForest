document.addEventListener('DOMContentLoaded', function() {
  const LANG_KEY = 'docs-lang';
  const htmlLang = (document.documentElement.lang || 'en').toLowerCase();
  const defaultLang = htmlLang.startsWith('es') ? 'ES' : 'EN';
  const storedLang = localStorage.getItem(LANG_KEY);
  const currentLang = storedLang || defaultLang;

  if (!storedLang) {
    localStorage.setItem(LANG_KEY, currentLang);
  }

  function normalizeHref(href) {
    if (!href) return href;
    const trimmed = href.trim();
    if (!trimmed || /^https?:/i.test(trimmed) || /^mailto:/i.test(trimmed) || /^tel:/i.test(trimmed) || trimmed.startsWith('#')) {
      return trimmed;
    }

    const hashIndex = trimmed.indexOf('#');
    const queryIndex = trimmed.indexOf('?');

    let pathEnd = trimmed.length;
    if (hashIndex !== -1) pathEnd = hashIndex;
    if (queryIndex !== -1 && (hashIndex === -1 || queryIndex < hashIndex)) {
      pathEnd = queryIndex;
    }

    const path = trimmed.slice(0, pathEnd);
    const suffix = trimmed.slice(pathEnd);

    return { path, suffix };
  }

  function transformPathForLang(path, lang) {
    if (path === '/' || path === '') {
      return lang === 'ES' ? 'index_es.html' : 'index.html';
    }

    if (!/\.html$/.test(path)) {
      return path;
    }

    if (lang === 'ES') {
      return path.endsWith('_es.html') ? path : path.replace(/\.html$/, '_es.html');
    }

    return path.endsWith('_es.html') ? path.replace(/_es\.html$/, '.html') : path;
  }

  function updateNavLinks(lang) {
    const navLinks = document.querySelectorAll('nav.sidebar a');
    navLinks.forEach((link) => {
      const original = normalizeHref(link.getAttribute('href'));
      if (!original || typeof original === 'string') {
        if (typeof original === 'string') {
          link.setAttribute('href', original);
        }
        return;
      }

      const { path, suffix } = original;
      const transformed = transformPathForLang(path, lang);
      link.setAttribute('href', `${transformed}${suffix}`);
    });
  }

  function updateActiveNav() {
    const navItems = document.querySelectorAll('nav.sidebar li');
    navItems.forEach((item) => item.classList.remove('active'));

    const currentPath = (() => {
      const path = window.location.pathname.replace(/^\//, '');
      return path || 'index.html';
    })();

    const links = document.querySelectorAll('nav.sidebar a');
    links.forEach((link) => {
      const temp = document.createElement('a');
      temp.href = link.getAttribute('href') || '';
      const linkPath = temp.pathname.replace(/^\//, '') || 'index.html';
      if (linkPath === currentPath) {
        const li = link.closest('li');
        if (li) {
          li.classList.add('active');
          const parentLi = li.parentElement && li.parentElement.closest('li');
          if (parentLi) {
            parentLi.classList.add('active');
          }
        }
      }
    });
  }

  function applyLanguage(lang) {
    updateNavLinks(lang);
    updateActiveNav();
    document.dispatchEvent(new CustomEvent('docs:language-applied', { detail: { lang } }));
  }

  function createSelector(selectedLang) {
    const select = document.createElement('select');
    select.innerHTML = '<option value="EN">EN</option><option value="ES">ES</option>';
    select.value = selectedLang;
    select.addEventListener('change', () => {
      const lang = select.value;
      localStorage.setItem(LANG_KEY, lang);
      const base = window.location.pathname.replace(/_es\.html$/, '.html');
      const target = lang === 'ES' ? base.replace(/\.html$/, '_es.html') : base;
      if (window.location.pathname !== target) {
        window.location.pathname = target;
      } else {
        applyLanguage(lang);
      }
    });
    return select;
  }

  const container = document.querySelector('.lang-switch');
  if (container) {
    container.innerHTML = '';
    container.appendChild(createSelector(currentLang));
  }

  applyLanguage(currentLang);

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

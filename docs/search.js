(function () {
  const input = document.getElementById('doc-search-input');
  const resultsContainer = document.getElementById('doc-search-results');
  if (!input || !resultsContainer) {
    return;
  }

  let index = [];
  let loaded = false;
  let pendingQuery = '';
  const indexUrl = window.__DOC_SEARCH_INDEX__ || 'search-data.json';

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const highlight = (text, tokens) => {
    if (!tokens.length) return text;
    let highlighted = text;
    tokens.forEach((token) => {
      if (!token) return;
      const pattern = new RegExp(`(${escapeRegExp(token)})`, 'gi');
      highlighted = highlighted.replace(pattern, '<mark>$1</mark>');
    });
    return highlighted;
  };

  const buildSnippet = (entry, tokens) => {
    if (!tokens.length) {
      return entry.excerpt;
    }
    const contentLower = entry.contentLower;
    let bestIndex = -1;
    tokens.forEach((token) => {
      const idx = contentLower.indexOf(token);
      if (idx !== -1 && (bestIndex === -1 || idx < bestIndex)) {
        bestIndex = idx;
      }
    });

    if (bestIndex === -1) {
      return entry.excerpt;
    }

    const window = 120;
    const start = Math.max(0, bestIndex - window);
    const end = Math.min(entry.content.length, bestIndex + window);
    let snippet = entry.content.slice(start, end).trim();
    if (start > 0) {
      snippet = `…${snippet}`;
    }
    if (end < entry.content.length) {
      snippet = `${snippet}…`;
    }
    return highlight(snippet, tokens);
  };

  const renderResults = (matches, tokens) => {
    if (!matches.length) {
      resultsContainer.innerHTML = '<div class="search-result">No results found / Sin resultados.</div>';
      resultsContainer.classList.add('active');
      return;
    }

    resultsContainer.innerHTML = '';

    const fragment = document.createDocumentFragment();
    matches.slice(0, 12).forEach((entry, index) => {
      const item = document.createElement('div');
      item.className = 'search-result';
      item.setAttribute('role', 'option');
      item.setAttribute('id', `doc-search-result-${index}`);

      const link = document.createElement('a');
      link.href = entry.url;
      link.innerHTML = highlight(entry.title, tokens);
      item.appendChild(link);

      const snippet = document.createElement('p');
      snippet.innerHTML = buildSnippet(entry, tokens);
      item.appendChild(snippet);

      fragment.appendChild(item);
    });

    resultsContainer.appendChild(fragment);
    resultsContainer.classList.add('active');
  };

  const computeScore = (entry, tokens) => {
    let score = 0;
    for (const token of tokens) {
      if (!token) continue;
      const inTitle = entry.titleLower.includes(token);
      const inContent = entry.contentLower.includes(token);
      if (!inTitle && !inContent) {
        return null;
      }
      score += inTitle ? 5 : 0;
      if (inContent) {
        score += 1;
      }
    }
    return score;
  };

  const search = (query) => {
    const tokens = query
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);

    if (!tokens.length) {
      resultsContainer.innerHTML = '';
      resultsContainer.classList.remove('active');
      return;
    }

    const matches = [];
    index.forEach((entry) => {
      const score = computeScore(entry, tokens);
      if (score !== null) {
        matches.push({ ...entry, score });
      }
    });

    matches.sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return a.title.localeCompare(b.title);
    });

    renderResults(matches, tokens);
  };

  const loadIndex = () => {
    if (loaded) return;
    loaded = true;
    fetch(indexUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Failed to load search index');
        }
        return response.json();
      })
      .then((data) => {
        index = data.map((entry) => ({
          ...entry,
          titleLower: entry.title.toLowerCase(),
          contentLower: entry.content.toLowerCase(),
        }));
        if (pendingQuery) {
          search(pendingQuery);
        }
      })
      .catch((error) => {
        console.error(error);
        resultsContainer.innerHTML = '<div class="search-result">Unable to load the search index.</div>';
        resultsContainer.classList.add('active');
      });
  };

  input.addEventListener('focus', loadIndex, { once: true });
  input.addEventListener('input', (event) => {
    const query = event.target.value.trim();
    pendingQuery = query;
    if (!loaded) {
      loadIndex();
      return;
    }
    search(query);
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.activeElement === input) {
      input.blur();
      resultsContainer.innerHTML = '';
      resultsContainer.classList.remove('active');
      return;
    }

    if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey) {
      const active = document.activeElement;
      const activeTag = active && active.tagName ? active.tagName.toLowerCase() : '';
      if (active === input || activeTag === 'input' || activeTag === 'textarea' || (active && active.isContentEditable)) {
        return;
      }
      event.preventDefault();
      input.focus();
    }
  });
})();

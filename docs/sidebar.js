function filterNav() {
  const query = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('#nav-list li').forEach(li => {
    const text = li.textContent.toLowerCase();
    li.style.display = text.includes(query) ? '' : 'none';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const parts = window.location.pathname.split('/');
  const docsIndex = parts.lastIndexOf('docs');
  let rel = '';
  if (docsIndex !== -1) {
    const depth = parts.length - docsIndex - 1;
    for (let i = 0; i < depth; i++) rel += '../';
  }

  sidebar.innerHTML = `
    <input type="text" id="search" placeholder="Buscar..." />
    <ul id="nav-list">
      <li><a href="${rel}index.html">InsideForest</a></li>
      <li><a href="${rel}getting-started/index.html">Guía rápida</a></li>
      <li><a href="${rel}tutorials/index.html">Tutoriales</a></li>
      <li><a href="${rel}guides/index.html">Guías de uso</a></li>
      <li><a href="${rel}api/index.html">Referencia API</a></li>
      <li><a href="${rel}benchmarks.html">Benchmarks</a></li>
      <li><a href="${rel}faq.html">FAQ</a></li>
      <li><a href="${rel}glossary.html">Glosario</a></li>
      <li><a href="${rel}resources.html">Recursos</a></li>
    </ul>`;

  const search = document.getElementById('search');
  if (search) {
    search.addEventListener('input', filterNav);
  }
});

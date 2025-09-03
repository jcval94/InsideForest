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
    <ul id="nav-list">
      <li class="collapsible">
        <a href="#">InsideForest</a>
        <ul class="nested">
          <li><a href="${rel}index.html">Inicio</a></li>
          <li><a href="${rel}getting-started/index.html">Guía rápida</a></li>
          <li><a href="${rel}tutorials/index.html">Tutoriales</a></li>
          <li><a href="${rel}guides/index.html">Guías de uso</a></li>
          <li><a href="${rel}api/index.html">Referencia API</a></li>
          <li><a href="${rel}benchmarks.html">Benchmarks</a></li>
          <li><a href="${rel}faq.html">FAQ</a></li>
          <li><a href="${rel}glossary.html">Glosario</a></li>
          <li><a href="${rel}resources.html">Recursos</a></li>
        </ul>
      </li>
    </ul>`;

  const coll = sidebar.querySelector('.collapsible > a');
  const nested = sidebar.querySelector('.nested');
  coll.addEventListener('click', (e) => {
    e.preventDefault();
    nested.classList.toggle('open');
  });

});
